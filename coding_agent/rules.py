"""Project rule loading for coding-agent.

Mature coding agents auto-load repository instructions (``AGENTS.md`` and
friends). This module keeps that behaviour dependency-free and constant for
the lifetime of an Agent, so prompt-cache prefixes stay stable.
"""

from __future__ import annotations

from pathlib import Path

AUTO_RULE_FILES = ("AGENTS.md", "CODING_AGENT.md", "CLAUDE.md")
MAX_RULE_BYTES_PER_FILE = 32 * 1024
MAX_RULE_BYTES_TOTAL = 64 * 1024


def _read_rule_file(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) > MAX_RULE_BYTES_PER_FILE:
        data = data[:MAX_RULE_BYTES_PER_FILE]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", "replace")


def load_rules(workdir: str, explicit_rules: str = "") -> str:
    """Load project rules from a workdir and explicit rule files.

    ``explicit_rules`` is a comma-separated list of file paths. Explicit files
    take precedence over auto-discovered files but are deduplicated.
    """
    seen: set[str] = set()
    chunks: list[str] = []
    total = 0

    def add(path: Path) -> None:
        nonlocal total
        if total >= MAX_RULE_BYTES_TOTAL:
            return
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key in seen or not resolved.is_file():
            return
        seen.add(key)
        text = _read_rule_file(resolved)
        if not text.strip():
            return
        text = text[: max(0, MAX_RULE_BYTES_TOTAL - total)]
        total += len(text)
        chunks.append(text)

    base = Path(workdir)
    if (base / ".coding-agent" / "rules.md").is_file():
        add(base / ".coding-agent" / "rules.md")
    for name in AUTO_RULE_FILES:
        add(base / name)
    for raw in (explicit_rules or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute() and (base / path).is_file():
            path = base / path
        add(path)

    if not chunks:
        return ""
    return "[Project rules]\n" + "\n\n".join(chunks).strip() + "\n"
