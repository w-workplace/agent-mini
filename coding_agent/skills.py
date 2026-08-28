"""Agent Skills: on-demand, keyword-discovered instruction bundles.

A *skill* is a markdown file with a small frontmatter header (``name``,
``description``, ``keywords``) and a body of instructions. At the start of a
fresh session, the task is matched against each skill's keywords; matching
skills' instructions are injected into the conversation (as a user message
right after the system prompt) so the model follows them. This implements
"progressive disclosure": only relevant skills are loaded, and the system
prompt stays constant (cache-friendly).

Skills are discovered from two places, with user skills overriding built-ins:
  1. built-in skills shipped in ``coding_agent/default_skills/<name>/SKILL.md``
  2. user skills in ``<workspace>/skills/<name>.md`` or ``<name>/SKILL.md``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Skill:
    name: str
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    body: str = ""

    def terms(self) -> set[str]:
        terms: set[str] = set()
        terms.add(self.name.lower())
        for part in re.split(r"[^a-z0-9]+", self.name.lower()):
            if len(part) >= 3:
                terms.add(part)
        for k in self.keywords:
            k = k.strip().lower()
            if len(k) >= 3:
                terms.add(k)
        return terms


def parse_skill_md(text: str, name_hint: str = "") -> Skill:
    """Parse a SKILL.md file: optional ``---`` frontmatter + markdown body."""
    name = name_hint
    description = ""
    keywords: list[str] = []
    body = text.strip()
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        end = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end = i
                break
        if end is not None:
            body = "\n".join(lines[end + 1:]).strip()
            for ln in lines[1:end]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    k = k.strip().lower()
                    v = v.strip()
                    if k == "name":
                        name = v
                    elif k == "description":
                        description = v
                    elif k == "keywords":
                        keywords = [x.strip() for x in v.split(",") if x.strip()]
    return Skill(name=name, description=description, keywords=keywords, body=body)


def _read_skill_file(path: Path, name_hint: str) -> Skill | None:
    try:
        return parse_skill_md(path.read_text(encoding="utf-8"), name_hint)
    except OSError:
        return None


def load_builtin_skills() -> list[Skill]:
    base = Path(__file__).parent / "default_skills"
    skills: list[Skill] = []
    if base.is_dir():
        for d in sorted(base.iterdir()):
            f = d / "SKILL.md" if d.is_dir() else d
            if f.is_file() and f.suffix == ".md":
                skill = _read_skill_file(f, d.name)
                if skill:
                    skills.append(skill)
    return skills


def load_user_skills(workspace: str) -> list[Skill]:
    base = Path(workspace).expanduser() / "skills"
    skills: list[Skill] = []
    if not base.is_dir():
        return skills
    for entry in sorted(base.iterdir()):
        if entry.is_dir():
            f = entry / "SKILL.md"
            if f.is_file():
                skill = _read_skill_file(f, entry.name)
                if skill:
                    skills.append(skill)
        elif entry.is_file() and entry.suffix == ".md":
            skill = _read_skill_file(entry, entry.stem)
            if skill:
                skills.append(skill)
    return skills


def load_skills(workspace: str) -> list[Skill]:
    """Built-in skills plus user skills (user overrides by name)."""
    by_name: dict[str, Skill] = {}
    for s in load_builtin_skills():
        by_name[s.name] = s
    for s in load_user_skills(workspace):
        by_name[s.name] = s
    return list(by_name.values())


def discover_skills(task: str, skills: list[Skill], max_skills: int = 3) -> list[Skill]:
    """Return skills whose keywords match the task, best matches first."""
    t = task.lower()
    scored: list[tuple[int, str, Skill]] = []
    for s in skills:
        score = sum(1 for term in s.terms() if term in t)
        if score:
            scored.append((score, s.name, s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [s for _, _, s in scored[:max_skills]]


def skill_prompt(skill: Skill) -> str:
    parts = [f"[Loaded skill: {skill.name}]"]
    if skill.description:
        parts.append(skill.description)
    if skill.body:
        parts.append(skill.body)
    return "\n\n".join(parts)
