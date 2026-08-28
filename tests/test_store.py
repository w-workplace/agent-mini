"""Tests for the git-like session/workspace store."""

import tempfile
import unittest
from pathlib import Path

from coding_agent.store import SessionStore, StoreError


class SessionStoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SessionStore(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, task, message=None):
        sid = self.store.create_session(task=task, message=message, workdir=str(self.store.work_dir))
        self.store.save_conversation(sid, [{"role": "user", "content": task}])
        self.store.advance_head(sid)
        return sid

    def test_initial_layout_and_head(self):
        self.assertTrue(self.store.work_dir.is_dir())
        self.assertEqual(self.store.current_branch(), "main")
        self.assertIsNone(self.store.resolve_head())

    def test_create_and_advance(self):
        sid = self._run("do thing")
        self.assertEqual(self.store.resolve_head(), sid)
        meta = self.store.load_meta(sid)
        self.assertEqual(meta["parent"], None)
        self.assertEqual(meta["message"], "do thing")
        self.assertEqual(self.store.load_conversation(sid)[0]["content"], "do thing")

    def test_parent_chain(self):
        a = self._run("first")
        b = self._run("second")
        c = self._run("third")
        self.assertEqual(self.store.load_meta(b)["parent"], a)
        self.assertEqual(self.store.load_meta(c)["parent"], b)
        self.assertEqual(self.store.resolve_head(), c)
        self.assertEqual(self.store.ancestors(c), {c, b, a})

    def test_log_newest_first(self):
        self._run("one")
        self._run("two")
        self._run("three")
        entries = self.store.log()
        self.assertEqual([m["id"] for m in entries], [
            self.store.resolve_head(),
            self.store.load_meta(self.store.resolve_head())["parent"],
            self.store.load_meta(self.store.load_meta(self.store.resolve_head())["parent"])["parent"],
        ])
        self.assertEqual(entries[0]["message"], "three")

    def test_graph_renders_fork(self):
        a = self._run("base")
        self.store.create_branch("feature")   # feature -> a
        b = self._run("main work")            # main -> b (parent a)
        self.store.checkout("feature")
        c = self._run("feature work")         # feature -> c (parent a)
        lines = self.store.graph(all_branches=True)
        # exactly one "*" line per session
        self.assertEqual(sum(1 for ln in lines if "*" in ln), 3)
        text = "\n".join(lines)
        self.assertIn("HEAD", text)
        self.assertIn("main", text)
        self.assertIn("feature", text)
        # a fork is drawn with a connector
        self.assertTrue(any("/" in ln or "\\" in ln for ln in lines))

    def test_branch_create_and_list(self):
        sid = self._run("base")
        self.store.create_branch("feature")
        branches = self.store.list_branches()
        self.assertIn("feature", branches)
        self.assertEqual(branches["feature"], sid)
        self.assertEqual(branches["main"], sid)

    def test_branch_delete(self):
        self._run("base")
        self.store.create_branch("feature")
        self.store.delete_branch("feature")
        self.assertNotIn("feature", self.store.list_branches())

    def test_branch_delete_current_refused(self):
        self._run("base")
        with self.assertRaises(StoreError):
            self.store.delete_branch("main")

    def test_switch_branch(self):
        self._run("base")
        self.store.create_branch("feature")
        self.store.checkout("feature")
        self.assertEqual(self.store.current_branch(), "feature")

    def test_checkout_detaches_and_forks(self):
        a = self._run("base")
        b = self._run("next")
        # roll back to a (detached), then a new run forks from a
        self.store.checkout(a)
        self.assertIsNone(self.store.current_branch())
        self.assertEqual(self.store.resolve_head(), a)
        c = self._run("fork")
        self.assertEqual(self.store.load_meta(c)["parent"], a)
        self.assertNotEqual(c, b)

    def test_resolve_ref_by_prefix(self):
        sid = self._run("base")
        self.assertEqual(self.store.resolve_ref(sid[:4]), sid)

    def test_resolve_ref_unknown(self):
        self._run("base")
        with self.assertRaises(StoreError):
            self.store.resolve_ref("zzzz")

    def test_resolve_ref_ambiguous_prefix(self):
        for sid in ("aaaa0001", "aaaa0002"):
            d = self.store.sessions_dir / sid
            d.mkdir()
            (d / "meta.json").write_text(
                '{"id": "%s", "parent": null, "created_at": 0}' % sid
            )
        with self.assertRaises(StoreError):
            self.store.resolve_ref("aaaa")

    def test_delete_session_guards(self):
        a = self._run("a")
        b = self._run("b")
        # b descends from a -> cannot delete a
        with self.assertRaises(StoreError):
            self.store.delete_session(a)
        # cannot delete HEAD
        with self.assertRaises(StoreError):
            self.store.delete_session(b)

    def test_delete_leaf_session(self):
        a = self._run("a")
        b = self._run("b")  # main -> b (parent a)
        self.store.advance_head(a)  # move main back to a; HEAD -> a
        self.store.delete_session(b)  # b is now an unreferenced leaf
        self.assertNotIn(b, {m["id"] for m in self.store.list_sessions()})

    def test_artifacts_snapshot_and_restore(self):
        sid = self._run("make files")
        work = self.store.work_dir
        (work / "out.txt").write_text("hello")
        (work / "sub").mkdir()
        (work / "sub" / "nested.py").write_text("x=1")
        files, _ = self.store.snapshot_artifacts(sid, str(work))
        self.assertEqual(files, 2)

        # clobber work, then restore
        (work / "out.txt").write_text("changed")
        self.assertTrue(self.store.restore_artifacts(sid))
        self.assertEqual((work / "out.txt").read_text(), "hello")
        self.assertEqual((work / "sub" / "nested.py").read_text(), "x=1")


    def test_session_id_is_16_hex_chars(self):
        sid = self._run("collision resistance")
        self.assertEqual(len(sid), 16)

    def test_resolve_ref_rejects_empty_and_traversal(self):
        for bad in ("", ".", "..", "../x", "a/b"):
            with self.assertRaises(StoreError):
                self.store.resolve_ref(bad)

    def test_load_conversation_tolerates_corrupt_lines(self):
        sid = self.store.create_session(task="x")
        log = self.store.sessions_dir / sid / "conversation.jsonl"
        log.write_text(
            '{"role": "user", "content": "good"}\n'
            'not json\n'
            '{"role": "assistant", "content": "also good"}\n',
            encoding="utf-8",
        )
        msgs = self.store.load_conversation(sid)
        self.assertEqual(
            [m["content"] for m in msgs], ["good", "also good"]
        )

    def test_restore_skips_symlinks(self):
        sid = self._run("make files")
        work = self.store.work_dir
        (work / "out.txt").write_text("hello")
        outside = Path(self._tmp.name) / "outside.txt"
        outside.write_text("secret")
        (work / "link").symlink_to(outside)
        files, _ = self.store.snapshot_artifacts(sid, str(work))
        self.assertEqual(files, 1)
        self.assertFalse(
            (self.store.sessions_dir / sid / "artifacts" / "link").exists()
        )


    def test_diff_session_and_workdir(self):
        a = self._run("base")
        (self.store.work_dir / "a.txt").write_text("one")
        self.store.snapshot_artifacts(a, str(self.store.work_dir))
        b = self.store.create_session(task="change", parent=a)
        (self.store.work_dir / "a.txt").write_text("two")
        (self.store.work_dir / "b.txt").write_text("new")
        self.store.snapshot_artifacts(b, str(self.store.work_dir))
        changes = self.store.diff_session(b)
        self.assertEqual({c["path"]: c["status"] for c in changes},
                         {"a.txt": "M", "b.txt": "A"})
        self.store.advance_head(b)
        (self.store.work_dir / "a.txt").write_text("three")
        wd_changes = {c["path"]: c["status"] for c in self.store.diff_workdir()}
        self.assertEqual(wd_changes, {"a.txt": "M"})


if __name__ == "__main__":
    unittest.main()
