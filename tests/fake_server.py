"""A tiny, scriptable OpenAI-compatible mock server for testing the agent.

It serves ``POST /v1/chat/completions`` and returns responses produced by a
user-supplied *scenario* function, which receives the parsed request body and
decides the next assistant message. This lets us exercise the full agent loop
(tool calling -> local execution -> result -> final answer) deterministically,
with no network or real API key.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

Scenario = Callable[[BaseHTTPRequestHandler, dict[str, Any]], dict[str, Any]]


def tool_call_response(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    """Build an assistant message carrying a single tool call."""
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def final_response(content: str) -> dict[str, Any]:
    """Build a plain assistant message with no tool calls."""
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


class FakeOpenAIServer:
    """Starts an HTTP server on an ephemeral port; ``base_url`` includes /v1."""

    def __init__(self, scenario: Scenario, responses_before_error: int = 0, error_status: int = 500):
        self.scenario = scenario
        self.responses_before_error = responses_before_error
        self.error_status = error_status
        self._count = 0
        handler = self._make_handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}/v1"

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path.endswith("/chat/completions"):
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length)
                    body = json.loads(raw)
                    outer._count += 1
                    if outer._count <= outer.responses_before_error:
                        self._send_json(
                            {"error": {"message": "injected error"}},
                            status=outer.error_status,
                        )
                        return
                    try:
                        response = outer.scenario(self, body)
                        self._send_json(response, status=200)
                    except Exception as exc:  # noqa: BLE001
                        self._send_json({"error": {"message": str(exc)}}, status=500)
                else:
                    self.send_error(404)

            def _send_json(self, payload: dict[str, Any], status: int) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True

            def log_message(self, *args: Any) -> None:  # silence request logs
                pass

        return Handler

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
