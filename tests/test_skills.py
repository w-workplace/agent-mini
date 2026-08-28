"""Tests for agent skills: parsing, discovery, built-in presets, injection."""

import tempfile
import unittest
from pathlib import Path

from coding_agent.agent import Agent
from coding_agent.config import Config
from coding_agent.llm import AssistantMessage
from coding_agent.skills import (
    Skill,
    discover_skills,
    load_builtin_skills,
    parse_skill_md,
    skill_prompt,
)


class _StubLLM:
    def chat(self, messages, tools=None):
        return AssistantMessage(content="done")


class SkillTestCase(unittest.TestCase):
    def test_parse_frontmatter(self):
        md = "---\nname: foo\ndescription: does things\nkeywords: a, b, c\n---\nBody here."
        s = parse_skill_md(md)
        self.assertEqual(s.name, "foo")
        self.assertEqual(s.description, "does things")
        self.assertEqual(s.keywords, ["a", "b", "c"])
        self.assertEqual(s.body, "Body here.")

    def test_parse_no_frontmatter(self):
        s = parse_skill_md("just a body", name_hint="hint")
        self.assertEqual(s.name, "hint")
        self.assertEqual(s.body, "just a body")

    def test_builtin_skills_present(self):
        skills = load_builtin_skills()
        names = {s.name for s in skills}
        for expected in ("git-commit", "testing", "code-review", "security-review", "python-style", "documentation"):
            self.assertIn(expected, names)

    def test_discover_skills(self):
        skills = [
            Skill(name="testing", keywords=["test", "pytest"]),
            Skill(name="git-commit", keywords=["commit", "git"]),
        ]
        self.assertEqual([s.name for s in discover_skills("add a unit test", skills)], ["testing"])
        self.assertEqual([s.name for s in discover_skills("commit my changes", skills)], ["git-commit"])
        self.assertEqual(discover_skills("nothing relevant here", skills), [])

    def test_discover_respects_cap(self):
        skills = [
            Skill(name=f"s{i}", keywords=["widget"]) for i in range(5)
        ]
        self.assertEqual(len(discover_skills("make a widget", skills, max_skills=3)), 3)

    def test_skill_prompt_format(self):
        p = skill_prompt(Skill(name="x", description="d", body="b"))
        self.assertIn("[Loaded skill: x]", p)
        self.assertIn("d", p)
        self.assertIn("b", p)

    def test_skill_injected_on_first_run(self):
        with tempfile.TemporaryDirectory() as ws:
            (Path(ws) / "skills").mkdir()
            (Path(ws) / "skills" / "my-skill.md").write_text(
                "---\nname: my-skill\ndescription: d\nkeywords: widget\n---\nAlways use widgets.\n"
            )
            config = Config(
                api_key="k", model="m", workspace=ws, skills=True,
                stream=False, subagents=False,
            )
            agent = Agent(config, llm=_StubLLM())
            agent.run("please build a widget")
            # [system, skill, task]
            self.assertIn("[Loaded skill: my-skill]", agent.messages[1]["content"])
            self.assertEqual(agent.messages[2]["content"], "please build a widget")

    def test_skill_not_reinjected_on_continue(self):
        with tempfile.TemporaryDirectory() as ws:
            (Path(ws) / "skills").mkdir()
            (Path(ws) / "skills" / "my-skill.md").write_text(
                "---\nname: my-skill\ndescription: d\nkeywords: widget\n---\nUse widgets.\n"
            )
            config = Config(
                api_key="k", model="m", workspace=ws, skills=True,
                stream=False, subagents=False,
            )
            # Continuing a session: history already carries any injected skill.
            history = [{"role": "user", "content": "[Loaded skill: my-skill]\n\nUse widgets."}]
            agent = Agent(config, llm=_StubLLM(), history=history)
            agent.run("another widget task")
            # No second skill injection.
            skill_msgs = [m for m in agent.messages if "Loaded skill" in (m.get("content") or "")]
            self.assertEqual(len(skill_msgs), 1)


if __name__ == "__main__":
    unittest.main()
