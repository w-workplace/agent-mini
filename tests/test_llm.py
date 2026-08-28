"""Tests for the LLM client: response parsing, streaming, and retry."""

import unittest
from unittest import mock

from coding_agent.llm import AssistantMessage, LLMClient, LLMError


class LLMParseTestCase(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient("https://example.com/v1", "key", "model")

    def test_parse_final_answer(self):
        data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ]
        }
        msg = self.client._parse(data)
        self.assertIsInstance(msg, AssistantMessage)
        self.assertEqual(msg.content, "done")
        self.assertEqual(msg.tool_calls, [])

    def test_parse_tool_calls(self):
        data = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "a.txt"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        msg = self.client._parse(data)
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0].name, "read_file")
        self.assertEqual(msg.tool_calls[0].arguments, {"path": "a.txt"})

    def test_parse_malformed_arguments(self):
        data = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "x", "arguments": "{not json"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        msg = self.client._parse(data)
        self.assertIn("_error", msg.tool_calls[0].arguments)

    def test_parse_bad_shape_raises(self):
        with self.assertRaises(LLMError):
            self.client._parse({"not": "a completion"})

    def test_to_api_dict_roundtrip(self):
        msg = AssistantMessage(content="hi", tool_calls=[])
        self.assertEqual(msg.to_api_dict()["role"], "assistant")
        self.assertEqual(msg.to_api_dict()["content"], "hi")


class LLMRetryTestCase(unittest.TestCase):
    def test_retries_on_transient_then_succeeds(self):
        from tests.fake_server import FakeOpenAIServer, final_response

        server = FakeOpenAIServer(
            lambda handler, body: final_response("ok"),
            responses_before_error=2,
            error_status=500,
        )
        client = LLMClient(server.base_url, "key", "model", retries=3)
        try:
            msg = client.chat([{"role": "user", "content": "hi"}])
            self.assertEqual(msg.content, "ok")
        finally:
            server.shutdown()

    def test_raises_after_retries_exhausted(self):
        from tests.fake_server import FakeOpenAIServer, final_response

        server = FakeOpenAIServer(
            lambda handler, body: final_response("ok"),
            responses_before_error=100,
            error_status=500,
        )
        client = LLMClient(server.base_url, "key", "model", retries=2)
        try:
            with self.assertRaises(LLMError):
                client.chat([{"role": "user", "content": "hi"}])
        finally:
            server.shutdown()


class _FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return b""


class StreamingTestCase(unittest.TestCase):
    def test_chat_stream_sse(self):
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b"data: [DONE]\n",
        ]
        client = LLMClient("https://x/v1", "k", "m")
        got = []
        with mock.patch("coding_agent.llm.urllib.request.urlopen", return_value=_FakeResponse(lines)):
            msg = client.chat_stream([{"role": "user", "content": "hi"}], on_text=got.append)
        self.assertEqual(msg.content, "Hello world")
        self.assertEqual("".join(got), "Hello world")

    def test_chat_stream_non_streaming_fallback(self):
        body = b'{"choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}'
        client = LLMClient("https://x/v1", "k", "m")
        got = []
        with mock.patch("coding_agent.llm.urllib.request.urlopen", return_value=_FakeResponse([body])):
            msg = client.chat_stream([{"role": "user", "content": "hi"}], on_text=got.append)
        self.assertEqual(msg.content, "done")
        self.assertEqual(got, ["done"])


    def test_chat_stream_non_streaming_fallback_invalid_json(self):
        client = LLMClient("https://x/v1", "k", "m")
        with mock.patch(
            "coding_agent.llm.urllib.request.urlopen",
            return_value=_FakeResponse([b"not-json"]),
        ):
            with self.assertRaises(LLMError):
                client.chat_stream([{"role": "user", "content": "hi"}])

    def test_extra_headers_are_sent(self):
        client = LLMClient(
            "https://x/v1", "k", "m", extra_headers={"X-Trace": "abc"}
        )
        req = client._build_request({"model": "m", "messages": []})
        self.assertEqual(req.get_header("X-trace"), "abc")


if __name__ == "__main__":
    unittest.main()
