"""End-to-end tests for the CLI: run flow + git-like session commands."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from coding_agent.cli import main
from coding_agent.store import SessionStore
from tests.fake_server import FakeOpenAIServer, final_response, tool_call_response


def _scenario():
    state = {"n": 0}

    def scenario(handler, body):
        n = state["n"]
        state["n"] += 1
        if n == 0:
            return tool_call_response("write_file",
                                      {"path": "greeting.txt", "content": "hello\n"}, "c0")
        if n == 1:
            return tool_call_response("run_command", {"command": "cat greeting.txt"}, "c1")
        return final_response("done")

    return scenario


class CLIRunTestCase(unittest.TestCase):
    def test_run_records_session_and_artifacts(self):
        server = FakeOpenAIServer(_scenario())
        with tempfile.TemporaryDirectory() as ws:
            try:
                rc = main([
                    "--workspace", ws,
                    "--base-url", server.base_url,
                    "--api-key", "k",
                    "--model", "m",
                    "run", "create a greeting file",
                ])
            finally:
                server.shutdown()
            self.assertEqual(rc, 0)
            store = SessionStore(ws)
            head = store.resolve_head()
            self.assertIsNotNone(head)
            # the agent's working directory holds the produced file
            self.assertTrue((store.work_dir / "greeting.txt").exists())
            # and it was snapshotted into the session's artifacts
            self.assertTrue((store.sessions_dir / head / "artifacts" / "greeting.txt").exists())
            # conversation log exists and is non-empty
            self.assertGreater(len(store.load_conversation(head)), 0)

    def test_session_commands_need_no_api_key(self):
        with tempfile.TemporaryDirectory() as ws:
            self.assertEqual(main(["--workspace", ws, "init"]), 0)
            self.assertEqual(main(["--workspace", ws, "status"]), 0)
            self.assertEqual(main(["--workspace", ws, "log"]), 0)
            # create a branch requires at least one session
            self.assertEqual(main(["--workspace", ws, "branch", "feature"]), 1)

    def test_branch_and_log_after_run(self):
        server = FakeOpenAIServer(_scenario())
        with tempfile.TemporaryDirectory() as ws:
            try:
                main(["--workspace", ws, "--base-url", server.base_url,
                      "--api-key", "k", "--model", "m", "run", "first"])
            finally:
                server.shutdown()
            store = SessionStore(ws)
            head = store.resolve_head()
            self.assertEqual(main(["--workspace", ws, "branch", "feature"]), 0)
            self.assertEqual(store.list_branches()["feature"], head)
            self.assertEqual(main(["--workspace", ws, "log", "--oneline"]), 0)

    def test_repl_merges_turns_into_one_session(self):
        server = FakeOpenAIServer(lambda handler, body: final_response("ok"))
        with tempfile.TemporaryDirectory() as ws:
            with mock.patch("builtins.input", side_effect=["first task", "second task", "/exit"]):
                try:
                    rc = main(["--workspace", ws, "--base-url", server.base_url,
                               "--api-key", "k", "--model", "m", "repl"])
                finally:
                    server.shutdown()
            self.assertEqual(rc, 0)
            store = SessionStore(ws)
            sessions = store.list_sessions()
            self.assertEqual(len(sessions), 1)  # one session, not one per turn
            conv = store.load_conversation(store.resolve_head())
            user_msgs = [m["content"] for m in conv if m["role"] == "user"]
            self.assertEqual(user_msgs, ["first task", "second task"])

    def test_log_graph_flag(self):
        server = FakeOpenAIServer(_scenario())
        with tempfile.TemporaryDirectory() as ws:
            try:
                main(["--workspace", ws, "--base-url", server.base_url,
                      "--api-key", "k", "--model", "m", "run", "first"])
            finally:
                server.shutdown()
            self.assertEqual(main(["--workspace", ws, "log", "--graph"]), 0)


if __name__ == "__main__":
    unittest.main()
