"""Tests for the local tool executors (ToolRunner)."""

import tempfile
import unittest
from pathlib import Path

from coding_agent.tools import ToolRunner


class ToolRunnerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.runner = ToolRunner(workdir=self._tmp.name)
        self.workdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_and_read_roundtrip(self):
        res = self.runner.write_file("hello.txt", "line one\nline two\nline three\n")
        self.assertTrue(res["ok"], res)
        read = self.runner.read_file("hello.txt")
        self.assertTrue(read["ok"], read)
        self.assertIn("line one", read["content"])
        self.assertIn("     1\tline one", read["content"])
        self.assertEqual(read["total_lines"], 3)

    def test_read_file_offset_limit(self):
        self.runner.write_file("f.txt", "".join(f"row{i}\n" for i in range(10)))
        read = self.runner.read_file("f.txt", offset=3, limit=2)
        self.assertTrue(read["ok"])
        self.assertEqual(read["lines_returned"], 2)
        self.assertIn("row3", read["content"])
        self.assertNotIn("row5", read["content"])

    def test_read_file_missing(self):
        res = self.runner.read_file("nope.txt")
        self.assertFalse(res["ok"])
        self.assertIn("cannot read", res["error"])

    def test_edit_file_single_replacement(self):
        self.runner.write_file("a.py", "x = 1\ny = 2\n")
        res = self.runner.edit_file("a.py", "x = 1", "x = 42")
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["replacements"], 1)
        content = self.runner.read_file("a.py")["content"]
        self.assertIn("x = 42", content)
        self.assertNotIn("x = 1", content)

    def test_edit_file_not_found(self):
        self.runner.write_file("a.py", "hello\n")
        res = self.runner.edit_file("a.py", "goodbye", "bye")
        self.assertFalse(res["ok"])
        self.assertIn("not found", res["error"])

    def test_edit_file_multiple_without_replace_all(self):
        self.runner.write_file("a.py", "dup\ndup\n")
        res = self.runner.edit_file("a.py", "dup", "x")
        self.assertFalse(res["ok"])
        self.assertIn("appears 2 times", res["error"])

    def test_edit_file_replace_all(self):
        self.runner.write_file("a.py", "dup\ndup\n")
        res = self.runner.edit_file("a.py", "dup", "x", replace_all=True)
        self.assertTrue(res["ok"])
        self.assertEqual(res["replacements"], 2)

    def test_list_files(self):
        self.runner.write_file("sub/a.py", "")
        self.runner.write_file("sub/b.py", "")
        self.runner.write_file("c.txt", "")
        res = self.runner.list_files(pattern="**/*.py")
        self.assertTrue(res["ok"])
        names = [Path(f).name for f in res["files"]]
        self.assertIn("a.py", names)
        self.assertIn("b.py", names)
        self.assertNotIn("c.txt", names)

    def test_grep(self):
        self.runner.write_file("s/f.py", "def foo():\n    return 1\n")
        self.runner.write_file("s/g.py", "print('hi')\n")
        res = self.runner.grep("foo", path="s")
        self.assertTrue(res["ok"])
        self.assertEqual(res["count"], 1)
        self.assertIn("foo", res["matches"][0])

    def test_grep_invalid_regex(self):
        res = self.runner.grep("([", path=".")
        self.assertFalse(res["ok"])
        self.assertIn("invalid regex", res["error"])

    def test_run_command_echo(self):
        res = self.runner.run_command("echo hi && echo err >&2")
        self.assertTrue(res["ok"])
        self.assertEqual(res["exit_code"], 0)
        self.assertIn("hi", res["stdout"])
        self.assertIn("err", res["stderr"])

    def test_run_command_exit_code(self):
        res = self.runner.run_command("exit 3")
        self.assertTrue(res["ok"])
        self.assertEqual(res["exit_code"], 3)

    def test_dangerous_command_blocked(self):
        res = self.runner.run_command("rm -rf /")
        self.assertFalse(res["ok"])
        self.assertIn("blocked", res["error"])

    def test_dangerous_command_allowed_when_flagged(self):
        runner = ToolRunner(workdir=self._tmp.name, allow_dangerous_commands=True)
        res = runner.run_command("echo safe")
        self.assertTrue(res["ok"])

    def test_path_escape_blocked(self):
        outside = Path(tempfile.gettempdir()) / "secret_outside.txt"
        outside.write_text("secret")
        try:
            res = self.runner.read_file(str(outside))
            self.assertFalse(res["ok"])
            self.assertIn("escapes", res["error"])
        finally:
            outside.unlink(missing_ok=True)

    def test_unknown_tool(self):
        res = self.runner.execute("nope_tool", {})
        self.assertFalse(res["ok"])
        self.assertIn("unknown tool", res["error"])

    def test_dunder_tool_blocked(self):
        # Model-provided names can never reach arbitrary methods.
        res = self.runner.execute("__init__", {})
        self.assertFalse(res["ok"])
        self.assertIn("unknown tool", res["error"])

    def test_unknown_argument_rejected(self):
        res = self.runner.execute("read_file", {"path": "a.txt", "bogus": 1})
        self.assertFalse(res["ok"])
        self.assertIn("unknown argument", res["error"])

    def test_missing_required_argument(self):
        res = self.runner.execute("read_file", {})
        self.assertFalse(res["ok"])
        self.assertIn("missing required argument", res["error"])

    def test_wrong_type_argument(self):
        res = self.runner.execute("read_file", {"path": 123})
        self.assertFalse(res["ok"])
        self.assertIn("must be a string", res["error"])

    def test_tool_result_is_redacted(self):
        from coding_agent.tools import format_tool_result
        out = format_tool_result({"ok": True, "stdout": "key=sk-abc1234567890"})
        self.assertNotIn("sk-abc1234567890", out)
        self.assertIn("REDACTED", out)


    def test_read_large_file_offsets_past_first_chunk(self):
        # Regression: offset/limit must work beyond the first MAX_READ_BYTES.
        self.runner.write_file(
            "big.txt", "".join(f"line {i}\n" for i in range(40000))
        )
        res = self.runner.read_file("big.txt", offset=4000, limit=5)
        self.assertTrue(res["ok"])
        self.assertEqual(res["lines_returned"], 5)
        self.assertIn("4000\tline 3999", res["content"])
        self.assertTrue(res["truncated"])

    def test_large_file_path_is_relative(self):
        self.runner.write_file("sub/f.txt", "hello\n")
        res = self.runner.read_file("sub/f.txt")
        self.assertTrue(res["ok"])
        self.assertEqual(res["path"], "sub/f.txt")

    def test_run_command_captures_bounded_tail(self):
        runner = ToolRunner(workdir=self._tmp.name, stream=False)
        res = runner.run_command(
            "python -c \"import sys; sys.stdout.write('a'*200000)\""
        )
        self.assertTrue(res["ok"])
        self.assertLessEqual(len(res["stdout"]), 50 * 1024)
        self.assertTrue(res.get("truncated"))


    def test_write_file_returns_diff(self):
        self.runner.write_file("f.txt", "one\n")
        res = self.runner.write_file("f.txt", "one\ntwo\n")
        self.assertTrue(res["ok"])
        self.assertIn("+two", res["diff"])
        self.assertEqual(res["added_lines"], 1)

    def test_edit_file_returns_diff(self):
        self.runner.write_file("f.txt", "a\nb\n")
        res = self.runner.edit_file("f.txt", "a", "A")
        self.assertTrue(res["ok"])
        self.assertIn("-a", res["diff"])
        self.assertIn("+A", res["diff"])


if __name__ == "__main__":
    unittest.main()
