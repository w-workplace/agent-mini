"""Command-line interface for the coding agent."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from . import __version__
from .agent import Agent, AgentError
from .config import load_config
from .llm import LLMClient
from .tools import TOOL_SCHEMAS, ToolRunner

_REPL_BANNER = """\
coding-agent REPL — type a task and press Enter. Multi-turn context is kept.
  /help   show this message
  /tools  list available tools
  /exit   quit (or send an empty line)
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coding-agent",
        description=(
            "A minimal, dependency-free coding agent: give it a task and it "
            "autonomously reads/writes files and runs commands until done."
        ),
    )
    p.add_argument("task", nargs="?", help="The task to perform. Use '-' to read from stdin.")
    p.add_argument("-m", "--model", help="Model name.")
    p.add_argument("--base-url", help="OpenAI-compatible API base URL (e.g. https://api.openai.com/v1).")
    p.add_argument("--api-key", help="API key (prefer the LLM_API_KEY / OPENAI_API_KEY env var).")
    p.add_argument("--max-iterations", type=int, help="Max agent loop iterations (default 40).")
    p.add_argument("--max-tokens", type=int, help="Max response tokens (0 = model default).")
    p.add_argument("--temperature", type=float, help="Sampling temperature (default 0.2).")
    p.add_argument("--context-limit-tokens", type=int, help="Trim history beyond this token budget.")
    p.add_argument("--workdir", help="Directory the agent operates in (default: current dir).")
    p.add_argument("--system-prompt", help="Override the default system prompt.")
    p.add_argument("--command-timeout", type=float, help="Max seconds per command (default 120).")
    p.add_argument("--allow-outside-workdir", action="store_true",
                   help="Allow file tools to touch paths outside the working directory.")
    p.add_argument("--allow-dangerous-commands", action="store_true",
                   help="Allow commands that match the blocked dangerous-command patterns.")
    p.add_argument("-i", "--interactive", action="store_true", help="Run an interactive REPL.")
    p.add_argument("-v", "--verbose", action="store_true", help="Log each agent/tool step to stderr.")
    p.add_argument("--list-tools", action="store_true", help="Print available tools and exit.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in (
        "model", "base_url", "api_key", "max_iterations", "max_tokens",
        "temperature", "context_limit_tokens", "workdir", "system_prompt",
        "command_timeout",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if args.allow_outside_workdir:
        overrides["allow_outside_workdir"] = True
    if args.allow_dangerous_commands:
        overrides["allow_dangerous_commands"] = True
    if args.verbose:
        overrides["verbose"] = True
    return overrides


def _build_agent(config: Any) -> Agent:
    llm = LLMClient(
        config.base_url,
        config.api_key,
        config.model,
        timeout=config.timeout,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        retries=config.request_retries,
        verbose=config.verbose,
    )
    tools = ToolRunner(
        workdir=config.workdir,
        allow_outside_workdir=config.allow_outside_workdir,
        allow_dangerous_commands=config.allow_dangerous_commands,
        command_timeout=config.command_timeout,
    )
    return Agent(config, llm=llm, tools=tools)


def _print_tools() -> None:
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        print(f"{fn['name']}: {fn['description']}")


def _run_once(agent: Agent, task: str) -> int:
    try:
        answer = agent.run(task)
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(answer)
    return 0


def _run_repl(agent: Agent) -> int:
    print(_REPL_BANNER, file=sys.stderr)
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        stripped = line.strip()
        if not stripped or stripped in ("/exit", "/quit", "exit", "quit"):
            break
        if stripped in ("/help", "help"):
            print(_REPL_BANNER, file=sys.stderr)
            continue
        if stripped == "/tools":
            _print_tools()
            continue
        try:
            answer = agent.run(line)
        except AgentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            continue
        print(answer)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_tools:
        _print_tools()
        return 0

    config = load_config(_cli_overrides(args))

    if not config.api_key:
        print(
            "error: no API key configured. Set LLM_API_KEY (or OPENAI_API_KEY) "
            "in the environment, or put it in an untracked config file — see "
            "README.md.",
            file=sys.stderr,
        )
        return 2

    agent = _build_agent(config)

    if args.interactive:
        return _run_repl(agent)

    task = args.task
    if task is None or task == "-":
        task = sys.stdin.read()
    if not task or not task.strip():
        print("error: no task provided (pass it as an argument or via stdin)", file=sys.stderr)
        return 2

    return _run_once(agent, task)


if __name__ == "__main__":
    sys.exit(main())
