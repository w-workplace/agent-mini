"""Command-line interface for the coding agent.

Sessions are managed like git commits: every ``run`` records a new immutable
session, branches are named pointers to sessions, and ``switch``/``checkout``
roll HEAD back. Commands are short and memorable:

    coding-agent "task"            # run a task (new session)   [= run "task"]
    coding-agent run "task"        # explicit
    coding-agent status            # workspace / branch / HEAD
    coding-agent log [--all]       # session history (current branch)
    coding-agent show <ref>        # view a session's conversation
    coding-agent switch <ref>      # move HEAD (no file changes)
    coding-agent checkout <ref>    # move HEAD + restore work/ artifacts
    coding-agent branch [<name>]   # list / create a branch
    coding-agent branch -d <name>  # delete a branch
    coding-agent rm <ref>          # delete a session
    coding-agent repl              # interactive (each turn = a session)
    coding-agent init              # initialize the workspace
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .agent import Agent, AgentError
from .config import load_config
from .llm import LLMClient
from .store import SessionStore, StoreError
from .tools import TOOL_SCHEMAS, ToolRunner

KNOWN_COMMANDS = {
    "run", "status", "log", "show", "switch", "sw", "checkout", "co",
    "branch", "br", "rm", "init", "repl",
}

_REPL_BANNER = """\
coding-agent REPL — each turn is recorded as a new session (git-like commit).
  /status   workspace & branch status
  /log      recent sessions
  /branch   list branches
  /exit     quit (or send an empty line)
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coding-agent",
        description=(
            "A minimal, dependency-free coding agent with git-like session "
            "management: every run is a session (commit); switch/checkout to "
            "roll back; branches to organize."
        ),
        add_help=True,
        allow_abbrev=False,
    )
    p.add_argument("--workspace", help="Session workspace dir (default ~/.coding-agent).")
    p.add_argument("-m", "--model", help="Model name.")
    p.add_argument("--base-url", help="OpenAI-compatible API base URL (e.g. https://api.openai.com/v1).")
    p.add_argument("--api-key", help="API key (prefer the LLM_API_KEY / OPENAI_API_KEY env var).")
    p.add_argument("-M", "--message", help="Short label for the session (default: first line of the task).")
    p.add_argument("--max-iterations", type=int, help="Max agent loop iterations (default 40).")
    p.add_argument("--max-tokens", type=int, help="Max response tokens (0 = model default).")
    p.add_argument("--temperature", type=float, help="Sampling temperature (default 0.2).")
    p.add_argument("--context-limit-tokens", type=int, help="Trim/compact history beyond this token budget.")
    p.add_argument("--workdir", help="Working directory for the agent (default: <workspace>/work).")
    p.add_argument("--system-prompt", help="Override the default system prompt.")
    p.add_argument("--command-timeout", type=float, help="Max seconds per command (default 120).")
    p.add_argument("--compact", action="store_true",
                   help="Summarize oldest turns when context overflows (cache-friendly).")
    p.add_argument("--allow-outside-workdir", action="store_true",
                   help="Allow file tools to touch paths outside the working directory.")
    p.add_argument("--allow-dangerous-commands", action="store_true",
                   help="Allow commands that match the blocked dangerous-command patterns.")
    p.add_argument("-v", "--verbose", action="store_true", help="Log each agent/tool step to stderr.")
    p.add_argument("-i", "--interactive", action="store_true", help="Alias for `repl`.")
    p.add_argument("--list-tools", action="store_true", help="Print available tools and exit.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _cli_overrides(opts: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in (
        "model", "base_url", "api_key", "max_iterations", "max_tokens",
        "temperature", "context_limit_tokens", "workdir", "system_prompt",
        "command_timeout", "workspace",
    ):
        value = getattr(opts, key, None)
        if value is not None:
            overrides[key] = value
    for flag in ("allow_outside_workdir", "allow_dangerous_commands", "verbose", "compact"):
        if getattr(opts, flag, False):
            overrides[flag] = True
    return overrides


def _build_agent(config: Any, workdir: str, history: list[dict[str, Any]] | None = None) -> Agent:
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
        workdir=workdir,
        allow_outside_workdir=config.allow_outside_workdir,
        allow_dangerous_commands=config.allow_dangerous_commands,
        command_timeout=config.command_timeout,
    )
    return Agent(config, llm=llm, tools=tools, history=history)


def _resolve_workdir(config: Any, opts: argparse.Namespace, store: SessionStore) -> str:
    raw = opts.workdir or config.workdir or str(store.work_dir)
    return str(Path(raw).expanduser().resolve())


def _run_task(config: Any, store: SessionStore, workdir: str, task: str, message: str | None = None) -> int:
    head = store.resolve_head()
    history = store.load_conversation(head) if head else None
    agent = _build_agent(config, workdir, history=history)
    try:
        answer = agent.run(task)
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    sid = store.create_session(
        task=task, parent=head, message=message, workdir=workdir, model=config.model
    )
    store.save_conversation(sid, agent.messages[1:])  # drop the system prompt
    store.snapshot_artifacts(sid, workdir)
    store.advance_head(sid)

    print(answer)
    print(f"[session {sid}]", file=sys.stderr)
    return 0


# -- commands ---------------------------------------------------------------
def _run(opts: argparse.Namespace, config: Any, task: str) -> int:
    if not task or not task.strip():
        print("error: no task provided (pass it as an argument or via stdin)", file=sys.stderr)
        return 2
    if not config.api_key:
        _no_api_key()
        return 2
    store = SessionStore(config.workspace)
    workdir = _resolve_workdir(config, opts, store)
    return _run_task(config, store, workdir, task, message=opts.message)


def _no_api_key() -> None:
    print(
        "error: no API key configured. Set LLM_API_KEY (or OPENAI_API_KEY) in "
        "the environment, or put it in the workspace config.json — see README.md.",
        file=sys.stderr,
    )


def _status(store: SessionStore) -> int:
    branch = store.current_branch()
    head = store.resolve_head()
    branches = store.list_branches()
    print(f"workspace : {store.root}")
    print(f"branch    : {branch or '(detached HEAD)'}")
    print(f"HEAD      : {head or '(none)'}")
    print(f"sessions  : {len(store.list_sessions())}")
    print(f"branches  : {', '.join(sorted(branches)) or '(none)'}")
    return 0


def _log(store: SessionStore, all_branches: bool, oneline: bool) -> int:
    entries = store.log(all_branches)
    branches = store.list_branches()
    head = store.resolve_head()
    for m in entries:
        sid = m["id"]
        markers = []
        if sid == head:
            markers.append("HEAD")
        for name, tip in branches.items():
            if tip == sid:
                markers.append(name)
        tag = f" ({', '.join(markers)})" if markers else ""
        if oneline:
            print(f"{sid}  {m.get('message', '')}{tag}")
        else:
            created = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.get("created_at", 0)))
            print(f"{sid}  {m.get('message', '')}{tag}")
            print(f"    parent: {m.get('parent') or '-'}   at {created}   model: {m.get('model', '')}")
            for line in (m.get("task") or "").splitlines()[:2]:
                if line.strip():
                    print(f"      {line.strip()}")
    if not entries:
        print("no sessions yet")
    return 0


def _show(store: SessionStore, ref: str, tail: int) -> int:
    sid = store.resolve_ref(ref)
    msgs = store.load_conversation(sid)
    if not msgs:
        print("(empty session)")
        return 0
    for m in msgs[-tail:]:
        role = m["role"]
        content = m.get("content") or ""
        if role == "assistant" and m.get("tool_calls"):
            names = ", ".join(t["function"]["name"] for t in m["tool_calls"])
            print(f"[{role}] -> {names}")
        else:
            content = content.replace("\n", "\n        ")
            if len(content) > 600:
                content = content[:600] + "..."
            print(f"[{role}] {content}")
    return 0


def _switch(store: SessionStore, ref: str, restore: bool) -> int:
    try:
        sid = store.checkout(ref, restore=restore)
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    note = " (work/ restored)" if restore else ""
    branch = store.current_branch()
    print(f"HEAD -> {sid} on {branch or 'detached HEAD'}{note}")
    return 0


def _branch(store: SessionStore, args: list[str]) -> int:
    if not args:
        branches = store.list_branches()
        cur = store.current_branch()
        if not branches:
            print("no branches")
            return 0
        for name in sorted(branches):
            mark = "*" if name == cur else " "
            tip = branches[name] or "(empty)"
            print(f"{mark} {name}  ->  {tip}")
        return 0
    if args[0] in ("-d", "-D", "--delete"):
        if len(args) < 2:
            print("usage: branch -d <name>", file=sys.stderr)
            return 1
        try:
            store.delete_branch(args[1])
        except StoreError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"deleted branch {args[1]}")
        return 0
    name = args[0]
    if name.startswith("-"):
        print("usage: branch [<name>] | branch -d <name>", file=sys.stderr)
        return 1
    try:
        store.create_branch(name)
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"branch {name} -> {store.resolve_head()}")
    return 0


def _rm(store: SessionStore, args: list[str]) -> int:
    if not args:
        print("usage: rm <session-id-or-prefix>", file=sys.stderr)
        return 1
    try:
        store.delete_session(args[0])
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"deleted session {args[0]}")
    return 0


def _init(store: SessionStore) -> int:
    print(f"workspace ready at {store.root}")
    return 0


def _repl(opts: argparse.Namespace, config: Any, store: SessionStore, workdir: str) -> int:
    print(_REPL_BANNER, file=sys.stderr)
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        s = line.strip()
        if not s:
            continue
        if s in ("/exit", "/quit", "exit", "quit"):
            break
        if s == "/status":
            _status(store)
            continue
        if s == "/log":
            _log(store, False, True)
            continue
        if s == "/branch":
            _branch(store, [])
            continue
        if s in ("/help", "help"):
            print(_REPL_BANNER, file=sys.stderr)
            continue
        if s.startswith("/"):
            print(f"unknown command {s!r}", file=sys.stderr)
            continue
        _run_task(config, store, workdir, line)
    return 0


def _print_tools() -> None:
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        print(f"{fn['name']}: {fn['description']}")


# -- dispatch ---------------------------------------------------------------
def _dispatch(opts: argparse.Namespace, cmd: str, cmd_args: list[str]) -> int:
    config = load_config(_cli_overrides(opts))

    if cmd == "run":
        task = " ".join(cmd_args)
        return _run(opts, config, task)

    if cmd == "init":
        return _init(SessionStore(config.workspace))
    if cmd == "status":
        return _status(SessionStore(config.workspace))
    if cmd == "log":
        return _log(SessionStore(config.workspace), "--all" in cmd_args, "--oneline" in cmd_args)
    if cmd == "show":
        ref = next((a for a in cmd_args if not a.startswith("-")), None)
        if not ref:
            print("usage: show <ref> [--all]", file=sys.stderr)
            return 1
        tail = 100000 if "--all" in cmd_args else 20
        return _show(SessionStore(config.workspace), ref, tail)
    if cmd in ("switch", "sw"):
        if not cmd_args:
            print("usage: switch <ref>", file=sys.stderr)
            return 1
        return _switch(SessionStore(config.workspace), cmd_args[0], restore=False)
    if cmd in ("checkout", "co"):
        ref = next((a for a in cmd_args if not a.startswith("-")), None)
        if not ref:
            print("usage: checkout <ref> [--no-restore]", file=sys.stderr)
            return 1
        restore = "--no-restore" not in cmd_args
        return _switch(SessionStore(config.workspace), ref, restore=restore)
    if cmd in ("branch", "br"):
        return _branch(SessionStore(config.workspace), cmd_args)
    if cmd == "rm":
        return _rm(SessionStore(config.workspace), cmd_args)
    if cmd == "repl":
        if not config.api_key:
            _no_api_key()
            return 2
        store = SessionStore(config.workspace)
        return _repl(opts, config, store, _resolve_workdir(config, opts, store))

    print(f"unknown command {cmd!r}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    opts, rest = build_parser().parse_known_args(argv)

    if opts.list_tools:
        _print_tools()
        return 0
    if opts.interactive:
        return _dispatch(opts, "repl", [])

    if not rest:
        cmd = "run"
        cmd_args: list[str] = []
    else:
        cmd = rest[0]
        cmd_args = rest[1:]

    if cmd not in KNOWN_COMMANDS:
        # A bare task (e.g. `coding-agent "fix the bug"`).
        task = " ".join([cmd] + cmd_args)
        return _dispatch(opts, "run", [task])

    return _dispatch(opts, cmd, cmd_args)


if __name__ == "__main__":
    sys.exit(main())
