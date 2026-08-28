"""Tests for subagents and the parallel_search tool."""

import tempfile
import unittest
from pathlib import Path

from coding_agent.config import Config
from coding_agent.llm import AssistantMessage
from coding_agent.subagent import run_subagents
from coding_agent.tools import ToolRunner


class _StubLLM:
    def __init__(self, prefix="summary:"):
        self.prefix = prefix

    def chat(self, messages, tools=None):
        last_user = [m for m in messages if m["role"] == "user"][-1]["content"]
        return AssistantMessage(content=f"{self.prefix}{last_user}")


class SubagentTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _config(self, **overrides):
        defaults = dict(
            api_key="k", model="m", workspace=str(self.workdir),
            skills=False, subagents=False, stream=False,
        )
        defaults.update(overrides)
        return Config(**defaults)

    def test_run_subagents_parallel(self):
        llm = _StubLLM()
        results = run_subagents(
            llm, self._config(), str(self.workdir), ["one", "two", "three"], max_parallel=2
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], {"query": "one", "summary": "summary:one"})
        self.assertEqual(results[2]["summary"], "summary:three")

    def test_read_only_toolset(self):
        runner = ToolRunner(workdir=str(self.workdir), read_only=True)
        names = {s["function"]["name"] for s in runner.tool_schemas}
        self.assertEqual(names, {"list_files", "read_file", "grep"})
        self.assertFalse(runner.execute("write_file", {"path": "x", "content": "y"})["ok"])
        self.assertFalse(runner.execute("run_command", {"command": "ls"})["ok"])
        self.assertFalse(runner.execute("parallel_search", {"queries": ["x"]})["ok"])

    def test_parallel_search_dispatch(self):
        def executor(queries):
            return [{"query": q, "summary": f"summary:{q}"} for q in queries]

        runner = ToolRunner(workdir=str(self.workdir), subagent_executor=executor)
        res = runner.execute("parallel_search", {"queries": ["a", "b"]})
        self.assertTrue(res["ok"])
        self.assertEqual(res["count"], 2)
        self.assertEqual(res["results"][1]["summary"], "summary:b")

    def test_parallel_search_requires_executor(self):
        runner = ToolRunner(workdir=str(self.workdir))  # no executor
        self.assertFalse(runner.execute("parallel_search", {"queries": ["a"]})["ok"])

    def test_parallel_search_rejects_bad_queries(self):
        runner = ToolRunner(workdir=str(self.workdir), subagent_executor=lambda q: q)
        self.assertFalse(runner.execute("parallel_search", {"queries": "not-a-list"})["ok"])
        self.assertFalse(runner.execute("parallel_search", {"queries": [1, 2]})["ok"])


if __name__ == "__main__":
    unittest.main()
