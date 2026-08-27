"""Minimal OpenAI-compatible chat-completions client (standard library only).

We implement the HTTP request, JSON payload, transient-error retry with
exponential backoff, and response parsing ourselves. The only external service
consumed is a model vendor's OpenAI-compatible chat-completions endpoint; no
agent framework or hosted code-execution/file tools are involved.
"""

from __future__ import annotations

import json
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# HTTP statuses worth retrying (rate limits and transient server faults).
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


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
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.verbose = verbose

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            raise LLMError(exc.code, f"HTTP {exc.code} from the model API", raw) from exc
        except urllib.error.URLError as exc:
            raise LLMError(None, f"connection error: {exc.reason}") from exc

    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None) -> AssistantMessage:
        """Send the conversation and tool schemas, return the parsed reply."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens and self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error: LLMError | None = None
        for attempt in range(self.retries + 1):
            try:
                data = self._request(payload)
                return self._parse(data)
            except LLMError as exc:
                last_error = exc
                transient = exc.status in _TRANSIENT_STATUS or exc.status is None
                if not transient or attempt >= self.retries:
                    raise
                delay = (2 ** attempt) + random.uniform(0, 0.5)
                if self.verbose:
                    print(
                        f"[llm] transient error (status={exc.status}); "
                        f"retrying in {delay:.1f}s",
                        file=sys.stderr,
                    )
                time.sleep(delay)
        # Unreachable, but keeps type checkers happy.
        raise last_error  # type: ignore[misc]

    def _parse(self, data: dict[str, Any]) -> AssistantMessage:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(None, "unexpected API response shape", json.dumps(data)) from exc

        message = choice.get("message") or {}
        content = message.get("content") or ""
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
