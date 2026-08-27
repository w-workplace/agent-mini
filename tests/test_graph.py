"""Tests for git-log-style DAG rendering."""

import unittest

from coding_agent.graph import render_graph


class RenderGraphTestCase(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(render_graph([], {}, {}), [])

    def test_linear(self):
        commits = ["c", "b", "a"]
        parents = {"c": "b", "b": "a", "a": None}
        labels = {"c": "c", "b": "b", "a": "a"}
        self.assertEqual(
            render_graph(commits, parents, labels),
            ["* c", "* b", "* a"],
        )

    def test_fork(self):
        # aaaaa is the root; ccccc (main) and bbbbb (feature) both descend from it.
        commits = ["ccccc", "bbbbb", "aaaaa"]
        parents = {"ccccc": "aaaaa", "bbbbb": "aaaaa", "aaaaa": None}
        labels = {"ccccc": "c", "bbbbb": "b", "aaaaa": "a"}
        self.assertEqual(
            render_graph(commits, parents, labels),
            ["* c", "| * b", "|/", "* a"],
        )

    def test_two_independent_chains(self):
        commits = ["c", "y", "b", "a", "x"]
        parents = {"c": "b", "b": "a", "a": None, "y": "x", "x": None}
        labels = {k: k for k in commits}
        self.assertEqual(
            render_graph(commits, parents, labels),
            ["* c", "| * y", "* | b", "* | a", "  * x"],
        )

    def test_merge_into_left_lane(self):
        # bbbbb (right lane) is newer than ccccc (left lane); both merge into aaaaa.
        commits = ["bbbbb", "ccccc", "aaaaa"]
        parents = {"bbbbb": "aaaaa", "ccccc": "aaaaa", "aaaaa": None}
        labels = {"bbbbb": "b", "ccccc": "c", "aaaaa": "a"}
        self.assertEqual(
            render_graph(commits, parents, labels),
            ["* b", "| * c", "|/", "* a"],
        )


if __name__ == "__main__":
    unittest.main()
