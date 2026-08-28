"""Read-only subagents for context-isolated, parallel exploration.

Subagents let the main agent offload context-heavy but low-output work (search
the codebase, read many files) to isolated child agents that run in parallel
and return only a concise summary — keeping the main context clean, saving
tokens, and speeding up broad investigation. Subagents are read-only (no file
writes and no command execution) and cannot spawn further subagents.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .agent import Agent, AgentError
from .tools import ToolRunner

SUBAGENT_SYSTEM_PROMPT = """\
You are a research subagent of a coding agent. You work read-only: you may list
files, read files, and search with grep, but you must NOT write or edit files
and must NOT run commands. Investigate the question you are given and reply
with a concise, factual summary of your findings: the relevant file paths, the
key code or facts, and a direct answer. Do not propose code changes — just
report what you found.
"""

MAX_SUBAGENT_QUERIES = 16
MAX_SUBAGENT_SUMMARY_CHARS = 4000


def run_subagents(
    llm: Any,
    config: Any,
    workdir: str,
    queries: list[str],
    max_parallel: int = 4,
) -> list[dict[str, Any]]:
    """Run read-only subagents over ``queries`` in parallel, returning summaries."""
    sub_config = config.with_overrides(
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
        subagents=False,  # subagents cannot spawn subagents
        skills=False,  # skill instructions are for the main coding agent
        stream=False,  # summaries are returned, not streamed to stdout
        max_iterations=min(config.max_iterations, 12),
        context_limit_tokens=min(config.context_limit_tokens, 16000),
        quiet=True,
    )

    def run_one(query: str) -> dict[str, Any]:
        tools = ToolRunner(
            workdir=workdir,
            allow_outside_workdir=config.allow_outside_workdir,
            read_only=True,
        )
        agent = Agent(sub_config, llm=llm, tools=tools)
        try:
            summary = agent.run(query)
            if len(summary) > MAX_SUBAGENT_SUMMARY_CHARS:
                summary = (
                    summary[:MAX_SUBAGENT_SUMMARY_CHARS]
                    + "\n[summary truncated]"
                )
            return {"query": query, "summary": summary}
        except AgentError as exc:
            return {"query": query, "error": str(exc)}

    queries = list(queries)[:MAX_SUBAGENT_QUERIES]
    max_parallel = max(1, int(max_parallel))
    if max_parallel == 1:
        return [run_one(q) for q in queries]
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        return list(ex.map(run_one, queries))
