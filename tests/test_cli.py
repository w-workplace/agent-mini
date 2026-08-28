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


    def test_show_bad_ref_is_friendly(self):
        with tempfile.TemporaryDirectory() as ws:
            rc = main(["--workspace", ws, "show", "does-not-exist"])
        self.assertEqual(rc, 1)

    def test_run_reads_task_from_stdin(self):
        server = FakeOpenAIServer(lambda handler, body: final_response("ok"))
        with tempfile.TemporaryDirectory() as ws:
            with mock.patch("sys.stdin", create=True) as fake_stdin:
                fake_stdin.isatty.return_value = False
                fake_stdin.read.return_value = "task from stdin"
                try:
                    rc = main([
                        "--workspace", ws, "--base-url", server.base_url,
                        "--api-key", "k", "--model", "m", "run",
                    ])
                finally:
                    server.shutdown()
            self.assertEqual(rc, 0)
            store = SessionStore(ws)
            conv = store.load_conversation(store.resolve_head())
            self.assertEqual(conv[0]["content"], "task from stdin")

    def test_failed_run_is_recorded_as_failed_session(self):
        state = {"n": 0}

        def never_finishes(handler, body):
            state["n"] += 1
            return tool_call_response(
                "list_files", {"pattern": f"p{state['n']}"}, f"c{state['n']}"
            )

        server = FakeOpenAIServer(never_finishes)
        with tempfile.TemporaryDirectory() as ws:
            try:
                rc = main([
                    "--workspace", ws, "--base-url", server.base_url,
                    "--api-key", "k", "--model", "m",
                    "--max-iterations", "2", "run", "never finish",
                ])
            finally:
                server.shutdown()
            self.assertEqual(rc, 1)
            store = SessionStore(ws)
            meta = store.load_meta(store.resolve_head())
            self.assertEqual(meta["status"], "failed")
            self.assertIn("max_iterations", meta["error"])

    def test_repl_save_checkpoints_and_continues(self):
        server = FakeOpenAIServer(lambda handler, body: final_response("ok"))
        with tempfile.TemporaryDirectory() as ws:
            with mock.patch("builtins.input", side_effect=[
                "first task", "/save", "second task", "/exit"
            ]):
                try:
                    rc = main([
                        "--workspace", ws, "--base-url", server.base_url,
                        "--api-key", "k", "--model", "m", "repl",
                    ])
                finally:
                    server.shutdown()
            self.assertEqual(rc, 0)
            store = SessionStore(ws)
            sessions = store.list_sessions()
            self.assertEqual(len(sessions), 2)
            parent = sessions[0]["id"]
            child = sessions[1]["id"]
            self.assertEqual(sessions[1]["parent"], parent)
            self.assertEqual(store.resolve_head(), child)


if __name__ == "__main__":
    unittest.main()
