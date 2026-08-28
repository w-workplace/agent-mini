"""Minimal OpenAI-compatible chat-completions client (standard library only).

We implement the HTTP request, JSON payload, transient-error retry with
exponential backoff, and response parsing ourselves. The only external service
consumed is a model vendor's OpenAI-compatible chat-completions endpoint; no
agent framework or hosted code-execution/file tools are involved.
"""

from __future__ import annotations

import gzip
import http.client
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

# HTTP statuses worth retrying (rate limits and transient server faults).
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
# Hard cap on a single completion response body (non-streaming and streaming).
MAX_RESPONSE_BYTES = 10 * 1024 * 1024


class LLMError(Exception):
    """Raised when the model API call fails in a way we cannot retry away."""

    def __init__(self, status: int | None, message: str, body: str = ""):
        self.status = status
        self.message = message
        self.body = body
        super().__init__(message)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantMessage:
    role: str = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize back to the wire format for the conversation history."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


class _StreamResponse:
    """Small wrapper around an ``http.client.HTTPResponse``.

    Adds transparent gzip decompression and connection-return bookkeeping so
    the response object can still be used with ``with ... as resp`` and
    ``for raw in resp`` exactly like the previous ``urllib`` object.
    """

    def __init__(self, resp: Any, release: Callable[[bool], None]):
        self._resp = resp
        self._release = release
        self._eof = False
        content_encoding = (resp.getheader("Content-Encoding") or "").lower()
        self._gzip_file = (
            gzip.GzipFile(fileobj=resp) if content_encoding == "gzip" else None
        )

    def _read_raw(self, n: int = -1) -> bytes:
        if self._gzip_file is not None:
            data = self._gzip_file.read(n)
        else:
            data = self._resp.read(n)
        if data == b"":
            self._eof = True
        return data

    def read(self, n: int = -1) -> bytes:
        return self._read_raw(n)

    def readline(self) -> bytes:
        if self._gzip_file is not None:
            data = self._gzip_file.readline()
        else:
            data = self._resp.readline()
        if data == b"":
            self._eof = True
        return data

    def __iter__(self) -> "_StreamResponse":
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if line == b"":
            raise StopIteration
        return line

    def __enter__(self) -> "_StreamResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        # Only return the connection to the pool when the body was consumed to
        # EOF; a partial read (e.g. SSE [DONE] followed by break) must close.
        reusable = self._eof and not getattr(self._resp, "will_close", False)
        self._release(reusable)


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_tokens: int = 0,
        temperature: float | None = 0.2,
        retries: int = 3,
        verbose: bool = False,
        extra_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.verbose = verbose
        self.extra_headers = dict(extra_headers or {})
        self._local = threading.local()

    def _payload(self, messages: list[dict[str, Any]], tools: list[dict] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens and self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def _build_request(self, payload: dict[str, Any]) -> urllib.request.Request:
        url = self.base_url + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        for name, value in self.extra_headers.items():
            headers[name] = value
        return urllib.request.Request(url, data=data, method="POST", headers=headers)

    # -- persistent HTTP/1.1 transport -------------------------------------
    def _connection_parts(self, req: urllib.request.Request) -> tuple[tuple[str, str, int], str]:
        parts = urllib.parse.urlsplit(req.full_url)
        host = parts.hostname or ""
        if not host:
            raise LLMError(None, f"invalid model API URL: {req.full_url!r}")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        key = (parts.scheme, host, port)
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query
        return key, target

    def _new_connection(self, key: tuple[str, str, int]):
        scheme, host, port = key
        if scheme == "https":
            return http.client.HTTPSConnection(host, port, timeout=self.timeout)
        return http.client.HTTPConnection(host, port, timeout=self.timeout)

    def _get_connection(self, req: urllib.request.Request):
        """Return a reusable per-thread connection for ``req``.

        Reusing one HTTP/1.1 keep-alive connection per thread avoids a fresh
        TCP+TLS handshake for every chat completion — the dominant avoidable
        latency source in a multi-step agent loop.
        """
        key, target = self._connection_parts(req)
        info = getattr(self._local, "conn_info", None)
        conn = info[0] if info and info[1] == key else None
        if conn is None or getattr(conn, "sock", None) is None:
            conn = self._new_connection(key)
        self._local.conn_info = (conn, key)
        return conn, target, key

    def _discard_connection(self) -> None:
        info = getattr(self._local, "conn_info", None)
        conn = info[0] if info else None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        self._local.conn_info = None

    def _release_connection(self, key: tuple[str, str, int], reusable: bool) -> None:
        info = getattr(self._local, "conn_info", None)
        conn = info[0] if info and info[1] == key else None
        if not reusable or conn is None or getattr(conn, "sock", None) is None:
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
            self._local.conn_info = None

    def _open_proxied(self, req: urllib.request.Request):
        """Fallback transport that honours ``HTTP(S)_PROXY`` environment vars.

        Proxied requests cannot use the keep-alive pool here; correctness
        (corporate gateways, local debugging proxies) is more important.
        """
        try:
            req.add_header("Accept-Encoding", "gzip")
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            raise LLMError(exc.code, f"HTTP {exc.code} from the model API", raw) from exc
        except urllib.error.URLError as exc:
            raise LLMError(None, f"connection error: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise LLMError(None, f"connection error: {exc}") from exc
        return _StreamResponse(resp, lambda reusable: None)

    def _open(self, req: urllib.request.Request):
        """Open a completion request through the pooled connection."""
        parts = urllib.parse.urlsplit(req.full_url)
        if urllib.request.getproxies().get(parts.scheme):
            return self._open_proxied(req)
        conn, target, key = self._get_connection(req)
        headers = {k: v for k, v in req.header_items()}
        headers.setdefault("Accept-Encoding", "gzip")
        headers.setdefault("Connection", "keep-alive")

        for attempt in range(2):
            try:
                conn.request(req.get_method(), target, body=req.data, headers=headers)
                raw = conn.getresponse()
                break
            except (http.client.HTTPException, TimeoutError, OSError) as exc:
                self._discard_connection()
                if attempt == 0:
                    conn, target, key = self._get_connection(req)
                    continue
                raise LLMError(None, f"connection error: {exc}") from exc

        if raw.status >= 400:
            body = ""
            try:
                wrapper = _StreamResponse(raw, lambda reusable: None)
                body = wrapper.read(MAX_RESPONSE_BYTES + 1).decode("utf-8", "replace")
            except (OSError, EOFError) as exc:
                body = f"(could not read error body: {exc})"
            finally:
                self._discard_connection()
            raise LLMError(
                raw.status, f"HTTP {raw.status} from the model API", body
            )
        return _StreamResponse(raw, lambda reusable: self._release_connection(key, reusable))

    @staticmethod
    def _read_response(resp: Any) -> str:
        """Read a non-streaming response body, enforcing a size cap."""
        raw = resp.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise LLMError(None, f"model API response exceeded {MAX_RESPONSE_BYTES} bytes")
        # Drain to EOF so the connection can be reused by the next request.
        extra = resp.read(1)
        if extra:
            raw += extra
        return raw.decode("utf-8", "replace")

    def _retry(self, exc: LLMError, attempt: int) -> bool:
        transient = exc.status in _TRANSIENT_STATUS or exc.status is None
        if not transient or attempt >= self.retries:
            return False
        delay = (2 ** attempt) + random.uniform(0, 0.5)
        if self.verbose:
            print(
                f"[llm] transient error (status={exc.status}); retrying in {delay:.1f}s",
                file=sys.stderr,
            )
        time.sleep(delay)
        return True

    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None) -> AssistantMessage:
        """Send the conversation and tool schemas, return the parsed reply."""
        payload = self._payload(messages, tools)
        last_error: LLMError | None = None
        for attempt in range(self.retries + 1):
            try:
                with self._open(self._build_request(payload)) as resp:
                    raw = self._read_response(resp)
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise LLMError(
                        None, "invalid JSON from the model API", raw[:2000]
                    ) from exc
                return self._parse(data)
            except LLMError as exc:
                last_error = exc
                if self._retry(exc, attempt):
                    continue
                raise
            except (OSError, ValueError) as exc:
                last_error = LLMError(None, f"failed to read model response: {exc}")
                if self._retry(last_error, attempt):
                    continue
                raise
        raise last_error  # type: ignore[misc]

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        on_text: Any = None,
    ) -> AssistantMessage:
        """Stream a completion, calling ``on_text(delta)`` for content deltas.

        Handles both a real SSE stream and (for compatibility with gateways and
        tests) a plain non-streaming JSON response. Returns the accumulated
        :class:`AssistantMessage` (including any tool calls).
        """
        payload = self._payload(messages, tools)
        payload["stream"] = True
        last_error: LLMError | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._read_stream(self._build_request(payload), on_text)
            except LLMError as exc:
                last_error = exc
                if self._retry(exc, attempt):
                    continue
                raise
            except (OSError, ValueError) as exc:
                last_error = LLMError(None, f"failed to read model stream: {exc}")
                if self._retry(last_error, attempt):
                    continue
                raise
        raise last_error  # type: ignore[misc]

    def _read_stream(self, req: urllib.request.Request, on_text: Any) -> AssistantMessage:
        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        finish_reason = ""
        total_bytes = 0
        first = True
        with self._open(req) as resp:
            for raw in resp:
                total_bytes += len(raw)
                if total_bytes > MAX_RESPONSE_BYTES:
                    raise LLMError(
                        None, f"model API stream exceeded {MAX_RESPONSE_BYTES} bytes"
                    )
                line = raw.decode("utf-8", "replace")
                if first:
                    first = False
                    if line.lstrip().startswith("data:"):
                        is_sse = True
                    else:
                        is_sse = False
                if not is_sse:
                    # A gateway may ignore `stream` and return plain JSON.
                    try:
                        rest = resp.read(MAX_RESPONSE_BYTES - total_bytes + 1)
                    except TypeError:  # test/mock response objects without size arg
                        rest = resp.read()
                    total_bytes += len(rest)
                    if total_bytes > MAX_RESPONSE_BYTES:
                        raise LLMError(
                            None, f"model API response exceeded {MAX_RESPONSE_BYTES} bytes"
                        )
                    body = (line + rest.decode("utf-8", "replace")).strip()
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise LLMError(
                            None, "invalid JSON from the model API", body[:2000]
                        ) from exc
                    msg = self._parse(data)
                    if on_text and msg.content:
                        on_text(msg.content)
                    # Drain any trailing byte so the connection can be reused.
                    try:
                        resp.read(1)
                    except TypeError:  # mock responses without a size argument
                        resp.read()
                    return msg

                s = line.strip()
                if not s or s.startswith(":"):
                    continue
                if s.startswith("data:"):
                    data_str = s[len("data:"):].strip()
                    if data_str == "[DONE]":
                        # Keep reading to EOF so the HTTP connection can be
                        # returned to the pool instead of re-handshaking.
                        continue
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                        if on_text:
                            on_text(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        entry = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            entry["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            entry["name"] = fn["name"]
                        if fn.get("arguments"):
                            entry["arguments"] += fn["arguments"]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

        content = "".join(content_parts)
        parsed_calls: list[ToolCall] = []
        for idx in sorted(tool_calls):
            entry = tool_calls[idx]
            try:
                args = json.loads(entry["arguments"]) if entry["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_raw": entry["arguments"], "_error": "invalid JSON arguments"}
            parsed_calls.append(ToolCall(entry["id"], entry["name"], args))
        return AssistantMessage(content=content, tool_calls=parsed_calls, finish_reason=finish_reason)

    def _parse(self, data: dict[str, Any]) -> AssistantMessage:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(None, "unexpected API response shape", json.dumps(data)) from exc

        message = choice.get("message") or {}
        raw_content = message.get("content") or ""
        if isinstance(raw_content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        elif isinstance(raw_content, str):
            content = raw_content
        else:
            content = str(raw_content)
        tool_calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            name = fn.get("name", "")
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw)
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": str(args_raw), "_error": "invalid JSON arguments"}
            if not isinstance(args, dict):
                args = {"_value": args}
            tool_calls.append(ToolCall(raw.get("id", ""), name, args))

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", ""),
            raw=data,
        )
