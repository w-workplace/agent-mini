"""End-to-end and unit tests for the agent loop."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from coding_agent.agent import (
    SUMMARY_MARKER,
    Agent,
    AgentError,
    MaxIterationsExceeded,
)
from coding_agent.config import Config
from coding_agent.llm import AssistantMessage, LLMClient
from coding_agent.tools import ToolRunner
from tests.fake_server import FakeOpenAIServer, final_response, tool_call_response


def _scenario_create_and_verify():
    """A scripted model: list -> write -> run -> finish.

    Each chat call advances one step, so this exercises the whole loop:
    tool calling, local execution, results fed back, and termination on a
    final answer with no tool calls.
    """
    state = {"n": 0}

    def scenario(handler, body):
        n = state["n"]
        state["n"] += 1
        if n == 0:
            return tool_call_response("list_files", {"pattern": "*"}, "call_0")
        if n == 1:
            return tool_call_response(
                "write_file",
                {"path": "greeting.txt", "content": "hello from coding-agent\n"},
                "call_1",
            )
        if n == 2:
            return tool_call_response(
                "run_command", {"command": "cat greeting.txt"}, "call_2"
            )
        return final_response("Done: created greeting.txt and verified its contents.")

    return scenario


class AgentEndToEndTestCase(unittest.TestCase):
    def test_full_loop(self):
        server = FakeOpenAIServer(_scenario_create_and_verify())
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(
                base_url=server.base_url,
                api_key="test-key",
                model="fake-model",
                workdir=tmp,
                max_iterations=10,
            )
            agent = Agent(config)
            try:
                answer = agent.run("Create a greeting file and verify it.")
            finally:
                server.shutdown()
            self.assertEqual(answer, "Done: created greeting.txt and verified its contents.")
            self.assertTrue(Path(tmp, "greeting.txt").exists())
            self.assertEqual(
                Path(tmp, "greeting.txt").read_text(), "hello from coding-agent\n"
            )

    def test_max_iterations_exceeded(self):
        state = {"n": 0}

        def infinite_tools(handler, body):
            # Always ask for a (distinct) tool call so the loop never reaches
            # a final answer, but the repeat-detection guard is not triggered.
            state["n"] += 1
            return tool_call_response(
                "list_files", {"pattern": f"pattern_{state['n']}"}, f"call_{state['n']}"
            )

        server = FakeOpenAIServer(infinite_tools)
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(
                base_url=server.base_url,
                api_key="k",
                model="m",
                workdir=tmp,
                max_iterations=3,
            )
            agent = Agent(config)
            try:
                with self.assertRaises(MaxIterationsExceeded):
                    agent.run("never finish")
            finally:
                server.shutdown()

    def test_error_fed_back_to_model(self):
        """A tool that fails should return its error to the model and continue."""
        state = {"n": 0}

        def scenario(handler, body):
            n = state["n"]
            state["n"] += 1
            if n == 0:
                return tool_call_response("read_file", {"path": "missing.txt"}, "call_0")
            return final_response("recovered after seeing the error")

        server = FakeOpenAIServer(scenario)
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(base_url=server.base_url, api_key="k", model="m", workdir=tmp)
            agent = Agent(config)
            try:
                answer = agent.run("read a missing file")
            finally:
                server.shutdown()
            self.assertEqual(answer, "recovered after seeing the error")


class AgentUnitTestCase(unittest.TestCase):
    def _agent(self, **overrides):
        config = Config(api_key="k", model="m", **overrides)
        return Agent(config)

    def test_is_stuck(self):
        sig = ((("list_files", '{"pattern":"*"}'),),)
        self.assertTrue(Agent._is_stuck([sig, sig, sig]))
        self.assertFalse(Agent._is_stuck([sig, sig]))
        self.assertFalse(Agent._is_stuck([sig, (("read_file", '{"path":"x"}'),), sig]))

    def test_estimate_tokens(self):
        agent = self._agent()
        msgs = [{"role": "user", "content": "x" * 40}]  # ~10 tokens
        self.assertGreaterEqual(agent._estimate_tokens(msgs), 10)

    def test_trim_context_keeps_system_and_latest(self):
        agent = self._agent(context_limit_tokens=100)
        messages = [{"role": "system", "content": "sys"}]
        for i in range(20):
            messages.append({"role": "user", "content": f"turn {i} " + "y" * 100})
        trimmed = agent._trim_context(messages)
        self.assertEqual(trimmed[0]["role"], "system")
        self.assertEqual(trimmed[-1]["role"], "user")
        self.assertLessEqual(agent._estimate_tokens(trimmed), 100)
        self.assertIn("turn 19", trimmed[-1]["content"])

    def test_trim_context_keeps_tool_pairing(self):
        agent = self._agent(context_limit_tokens=200)
        messages = [
            {"role": "system", "content": "sys " + "z" * 800},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "c", "content": "result"},
        ]
        trimmed = agent._trim_context(messages)
        # The system message alone exceeds budget; tool/assistant pair may drop
        # but must never leave an orphan tool message without its assistant.
        roles = [m["role"] for m in trimmed]
        for i, role in enumerate(roles):
            if role == "tool":
                self.assertEqual(roles[i - 1], "assistant")

    def test_history_injected_after_system(self):
        agent = Agent(
            Config(api_key="k", model="m"),
            history=[{"role": "user", "content": "past"}],
        )
        # System prompt stays first and constant; history follows verbatim.
        self.assertEqual(agent.messages[0]["role"], "system")
        self.assertEqual(agent.messages[1], {"role": "user", "content": "past"})

    def test_system_prompt_is_constant(self):
        a = Agent(Config(api_key="k", model="m"))
        b = Agent(Config(api_key="k", model="m"))
        self.assertEqual(a.messages[0]["content"], b.messages[0]["content"])

    def test_result_summary(self):
        self.assertEqual(
            Agent._result_summary({"ok": True, "exit_code": 0}), "ok (exit_code=0)"
        )
        self.assertEqual(
            Agent._result_summary({"ok": True, "bytes_written": 24}), "ok (bytes_written=24)"
        )
        self.assertEqual(
            Agent._result_summary({"ok": False, "error": "boom"}), "error: boom"
        )

    def test_progress_gating(self):
        quiet = Agent(Config(api_key="k", model="m", quiet=True))
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            quiet._progress("P")
            quiet._detail("D")
        self.assertEqual(buf.getvalue(), "")

        default = Agent(Config(api_key="k", model="m"))
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            default._progress("P")
            default._detail("D")
        self.assertIn("P", buf.getvalue())
        self.assertNotIn("D", buf.getvalue())

        verbose = Agent(Config(api_key="k", model="m", verbose=True))
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            verbose._detail("D")
        self.assertIn("D", buf.getvalue())


    def test_trim_keeps_latest_turn_even_when_over_budget(self):
        agent = self._agent(context_limit_tokens=10)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old " + "x" * 200},
            {"role": "user", "content": "current " + "y" * 200},
        ]
        trimmed = agent._trim_context(messages)
        self.assertEqual(trimmed[-1]["content"], "current " + "y" * 200)

    def test_user_task_secrets_are_redacted_before_sending(self):
        config = Config(
            api_key="k", model="m", stream=False, subagents=False, skills=False
        )
        agent = Agent(config)
        agent.llm = _StubLLMWithFinishReason("stop")
        agent.run("configure api_key='sk-abc1234567890' for me")
        self.assertNotIn("sk-abc1234567890", agent.messages[1]["content"])
        self.assertIn("REDACTED", agent.messages[1]["content"])

    def test_tool_calls_finish_reason_without_calls_is_error(self):
        config = Config(api_key="k", model="m", stream=False, subagents=False)
        agent = Agent(config)
        agent.llm = _StubLLMWithFinishReason("tool_calls")
        with self.assertRaises(AgentError):
            agent.run("task")

    def test_is_stuck_ignores_different_results(self):
        a = (("grep", "{}", "result-a"),)
        b = (("grep", "{}", "result-b"),)
        self.assertFalse(Agent._is_stuck([a, b, a]))
        self.assertTrue(Agent._is_stuck([a, a, a]))


class _FakeSummarizerLLM:
    def chat(self, messages, tools=None):
        if tools is None:
            return AssistantMessage(content="THE SUMMARY")
        raise AssertionError("main chat should not run during compaction")


class CompactionTestCase(unittest.TestCase):
    def test_compact_folds_oldest_into_stable_summary(self):
        config = Config(api_key="k", model="m", compact=True, context_limit_tokens=500)
        agent = Agent(config)
        agent.llm = _FakeSummarizerLLM()
        messages = [{"role": "system", "content": "sys"}]
        for i in range(50):
            messages.append({"role": "user", "content": f"old turn {i} " + "x" * 100})
        messages.append({"role": "user", "content": "recent " + "y" * 100})

        out = agent._compact(messages)

        self.assertEqual(out[0]["role"], "system")
        self.assertTrue(out[1]["content"].startswith(SUMMARY_MARKER))
        self.assertIn("THE SUMMARY", out[1]["content"])
        # the newest turn is retained verbatim
        self.assertIn("recent", out[-1]["content"])
        # the summary sits immediately after system (stable prefix)
        self.assertEqual(out[1]["role"], "user")
        self.assertLessEqual(agent._estimate_tokens(out), 500)



class _StubLLMWithFinishReason:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason

    def chat(self, messages, tools=None):
        return AssistantMessage(content=None, finish_reason=self.finish_reason)


if __name__ == "__main__":
    unittest.main()
