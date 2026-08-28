"""Command-line interface for the coding agent.

Sessions are managed like git commits: every ``run`` records a new immutable
session, branches are named pointers to sessions, and ``switch``/``checkout``
roll HEAD back. Commands are short and memorable:

    coding-agent "task"            # run a task (new session)   [= run "task"]
    coding-agent run "task"        # explicit
    coding-agent status            # workspace / branch / HEAD
    coding-agent log [--all] [--graph] [--oneline]   # session history (DAG)
    coding-agent show <ref>        # view a session's conversation
    coding-agent switch <ref>      # move HEAD (no file changes)
    coding-agent checkout <ref>    # move HEAD + restore work/ artifacts
    coding-agent branch [<name>]   # list / create a branch
    coding-agent branch -d <name>  # delete a branch
    coding-agent rm <ref>          # delete a session
    coding-agent repl              # interactive (turns merge into ONE session)
    coding-agent init              # initialize the workspace
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .agent import DEFAULT_SYSTEM_PROMPT, Agent, AgentError
from .config import Config, load_config, parse_headers, validate_config
from .llm import LLMClient
from .security import detect_sandbox_backend, redact
from .store import SessionStore, StoreError
from .tools import TOOL_SCHEMAS, ToolRunner

KNOWN_COMMANDS = {
    "run", "status", "log", "show", "switch", "sw", "checkout", "co",
    "branch", "br", "rm", "init", "repl", "doctor", "config", "completion",
}

_SUBCOMMAND_HELP: dict[str, str] = {
    "run": "usage: coding-agent run [TASK]\n\nRun one task and seal it as a new session.\nUse --continue, --resume REF, --plan, --ask, --json, --file FILE.",
    "status": "usage: coding-agent status\n\nShow workspace, HEAD, branch, session stats and work/ changes.",
    "log": "usage: coding-agent log [--all] [--oneline] [--graph]\n\nList session history as a DAG.",
    "show": "usage: coding-agent show REF [--all] [--diff]\n\nShow a session conversation and artifact diff.",
    "switch": "usage: coding-agent switch REF\n\nMove HEAD only (no file restore).",
    "checkout": "usage: coding-agent checkout REF [--no-restore]\n\nMove HEAD and restore workspace work/ snapshot.",
    "branch": "usage: coding-agent branch [NAME] | branch -d NAME\n\nList, create or delete branches.",
    "rm": "usage: coding-agent rm REF\n\nDelete a session (safety guards apply).",
    "init": "usage: coding-agent init\n\nInitialize the workspace layout.",
    "repl": "usage: coding-agent repl\n\nInteractive multi-turn session. Commands: /status /log /branch /save /tokens /diff /undo /checkout REF /compact /skills /exit.",
    "doctor": "usage: coding-agent doctor\n\nCheck workspace, configuration, sandbox and ripgrep.",
    "config": "usage: coding-agent config [show|path|get KEY|set KEY VALUE|unset KEY]\n\nInspect or update workspace config.json.",
    "completion": "usage: coding-agent completion bash\n\nEmit shell completion (bash).",
}

_REPL_BANNER = """\
coding-agent REPL — turns are merged into ONE session (sealed on exit).
  /status       workspace & branch status
  /log          recent sessions
  /branch       list branches
  /save         save (checkpoint) the current session to disk now
  /tokens       token usage for the current REPL session
  /diff         work/ changes vs HEAD snapshot
  /undo         restore work/ from the HEAD snapshot
  /checkout REF save + switch to another session (restores work/)
  /compact      compact conversation now
  /skills       list loaded/available skills
  /exit         quit
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
    p.add_argument("--sandbox", action=argparse.BooleanOptionalAction, default=None,
                   help="Run commands in a network-less, read-only-root sandbox (bwrap/firejail).")
    p.add_argument("--env-allow", help="Extra env vars (comma-separated) to pass through to commands.")
    p.add_argument("--subagents", action=argparse.BooleanOptionalAction, default=None,
                   help="Enable read-only subagents (parallel_search).")
    p.add_argument("--subagent-parallel", type=int, help="Max concurrent subagents (default 4).")
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None,
                   help="Stream assistant text and command output live.")
    p.add_argument("--skills", action=argparse.BooleanOptionalAction, default=None,
                   help="Auto-load keyword-matched agent skills.")
    p.add_argument("--max-skills", type=int, help="Max skills to inject per session (default 3).")
    p.add_argument("--minimal", action=argparse.BooleanOptionalAction, default=None,
                   help="Minimal mode: print only the final answer (silences progress, streaming, skills, subagents).")
    p.add_argument("--compact", action=argparse.BooleanOptionalAction, default=None,
                   help="Summarize oldest turns when context overflows (cache-friendly).")
    p.add_argument("--allow-outside-workdir", action=argparse.BooleanOptionalAction, default=None,
                   help="Allow file tools to touch paths outside the working directory.")
    p.add_argument("--allow-dangerous-commands", action=argparse.BooleanOptionalAction, default=None,
                   help="Allow commands that match the blocked dangerous-command patterns.")
    p.add_argument("--snapshot", action=argparse.BooleanOptionalAction, default=None,
                   help="Snapshot the workspace work/ directory into each session.")
    p.add_argument("--plan", action=argparse.BooleanOptionalAction, default=None,
                   help="Read-only planning mode: no writes or commands.")
    p.add_argument("--ask", action=argparse.BooleanOptionalAction, default=None,
                   help="Ask before write_file/edit_file/run_command.")
    p.add_argument("--rules", action="append", metavar="FILE",
                   help="Load extra rule file(s) into the system prompt.")
    p.add_argument("--file", action="append", metavar="FILE",
                   help="Attach a file to the task as context.")
    p.add_argument("--continue", action="store_true", dest="continue_run",
                   help="Continue the current HEAD session with a new task.")
    p.add_argument("--resume", metavar="REF",
                   help="Switch to REF (restoring work/ if available), then run.")
    p.add_argument("--json", action="store_true", dest="json_output",
                   help="Print the run result as JSON on stdout.")
    p.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction, default=None,
                   help="Log each agent/tool step to stderr.")
    p.add_argument("-q", "--quiet", action=argparse.BooleanOptionalAction, default=None,
                   help="Suppress progress output (stderr).")
    p.add_argument("--header", action="append", metavar="NAME:VALUE",
                   help="Extra HTTP header for the model API; may be repeated.")
    p.add_argument("-i", "--interactive", action="store_true", help="Alias for `repl`.")
    p.add_argument("--list-tools", action="store_true", help="Print available tools and exit.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _cli_overrides(opts: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in (
        "model", "base_url", "api_key", "max_iterations", "max_tokens",
        "temperature", "context_limit_tokens", "workdir", "system_prompt",
        "command_timeout", "workspace", "env_allow",
    ):
        value = getattr(opts, key, None)
        if value is not None:
            overrides[key] = value
    if getattr(opts, "subagent_parallel", None) is not None:
        overrides["subagent_parallel"] = opts.subagent_parallel
    if getattr(opts, "max_skills", None) is not None:
        overrides["max_skills"] = opts.max_skills
    for flag in (
        "allow_outside_workdir", "allow_dangerous_commands", "verbose",
        "compact", "quiet", "sandbox", "minimal", "subagents", "stream",
        "skills", "snapshot", "plan", "ask",
    ):
        value = getattr(opts, flag, None)
        if value is not None:
            overrides[flag] = value
    if getattr(opts, "rules", None):
        overrides["rules"] = ",".join(opts.rules)
    if getattr(opts, "json_output", False):
        # JSON stdout must not be polluted by streamed assistant text.
        overrides["stream"] = False
    headers: dict[str, str] = {}
    for raw in getattr(opts, "header", None) or []:
        try:
            headers.update(parse_headers(raw))
        except ValueError as exc:
            # argparse cannot validate this at parse time; surface it during
            # dispatch through a synthetic override key handled by main().
            overrides["_header_error"] = str(exc)
    if headers:
        overrides["extra_headers"] = headers
    return overrides


PLAN_MODE_SUFFIX = """

You are in PLAN MODE (read-only). Do not call write_file, edit_file or
run_command. Explore the codebase, produce a concrete step-by-step
implementation plan, and stop with that plan as your final answer."""


def _build_agent(config: Any, workdir: str, history: list[dict[str, Any]] | None = None) -> Agent:
    agent_config = config
    approval_state = {"all": False}

    def ask_approval(action: str, detail: str) -> bool:
        if approval_state["all"]:
            return True
        while True:
            print(
                f"Allow {action} {detail}? [y/N/a]: ",
                file=sys.stderr, end="", flush=True,
            )
            answer = input().strip().lower()
            if answer in ("y", "yes"):
                return True
            if answer in ("a", "all"):
                approval_state["all"] = True
                return True
            if answer in ("", "n", "no"):
                return False

    if config.plan:
        base_prompt = config.system_prompt or DEFAULT_SYSTEM_PROMPT
        agent_config = config.with_overrides(
            system_prompt=base_prompt.rstrip() + PLAN_MODE_SUFFIX,
            subagents=False,
            skills=False,
        )

    llm = LLMClient(
        config.base_url,
        config.api_key,
        config.model,
        timeout=config.timeout,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        retries=config.request_retries,
        verbose=config.verbose,
        extra_headers=getattr(config, "extra_headers", None),
    )
    tools = ToolRunner(
        workdir=workdir,
        allow_outside_workdir=config.allow_outside_workdir,
        allow_dangerous_commands=config.allow_dangerous_commands,
        command_timeout=config.command_timeout,
        sandbox=config.sandbox,
        env_allow=config.env_allow,
        stream=config.stream,
        quiet=config.quiet,
        read_only=config.plan,
        approval_callback=ask_approval if config.ask else None,
    )
    return Agent(agent_config, llm=llm, tools=tools, history=history)


def _resolve_workdir(config: Any, opts: argparse.Namespace, store: SessionStore) -> str:
    raw = opts.workdir or config.workdir or str(store.work_dir)
    return str(Path(raw).expanduser().resolve())


def _should_snapshot(config: Any, store: SessionStore, workdir: str) -> tuple[bool, str]:
    if getattr(config, "plan", False):
        return False, "plan mode is read-only"
    if not getattr(config, "snapshot", True):
        return False, "snapshot disabled"
    if Path(workdir).resolve() == store.work_dir.resolve():
        return True, ""
    return False, "custom workdir; only the workspace work/ directory is snapshotted"


def _seal_session(
    config: Any,
    store: SessionStore,
    workdir: str,
    agent: Agent,
    task: str,
    parent: str | None,
    message: str | None = None,
    error: str = "",
) -> str:
    """Create, persist and advance a session (successful or failed)."""
    mode = "plan" if getattr(config, "plan", False) else (
        "ask" if getattr(config, "ask", False) else ""
    )
    sid = store.create_session(
        task=redact(task),
        parent=parent,
        message=redact(message) if message else None,
        workdir=workdir,
        model=config.model,
        status="failed" if error else "ok",
        error=error,
        duration_ms=agent.duration_ms,
        usage=agent.usage,
        changed_files=agent.stats.get("changed_files", []),
        mode=mode,
    )
    store.save_conversation(sid, agent.messages[1:])  # drop the system prompt
    do_snapshot, reason = _should_snapshot(config, store, workdir)
    if do_snapshot:
        try:
            store.snapshot_artifacts(sid, workdir)
        except (StoreError, OSError) as exc:
            print(f"warning: could not snapshot artifacts: {exc}", file=sys.stderr)
    elif reason and not getattr(config, "quiet", False):
        print(f"[run] not snapshotted: {reason}", file=sys.stderr)
    store.advance_head(sid)
    return sid


def _run_task(
    config: Any,
    store: SessionStore,
    workdir: str,
    task: str,
    message: str | None = None,
    json_output: bool = False,
) -> int:
    head = store.resolve_head()
    history = store.load_conversation(head) if head else None
    branch = store.current_branch() or "(detached)"
    if not getattr(config, "quiet", False):
        print(f"[run] branch {branch} · model {config.model} · {workdir}", file=sys.stderr)
    agent = _build_agent(config, workdir, history=history)
    error = ""
    try:
        answer = agent.run(task)
    except AgentError as exc:
        error = str(exc)
        answer = ""

    result: dict[str, Any] = {
        "session_id": None,
        "status": "error" if error else "ok",
        "answer": answer,
        "error": error or None,
        "changed_files": agent.stats.get("changed_files", []),
        "tokens": {
            "prompt": agent.usage.get("prompt_tokens", 0),
            "completion": agent.usage.get("completion_tokens", 0),
            "total": agent.total_tokens,
        },
        "duration_ms": agent.stats.get("duration_ms", 0),
    }

    # Emit the result before the (potentially slow) artifact snapshot.
    if json_output:
        pass  # JSON is printed after sealing so session_id is available.
    elif error:
        print(f"error: {error}", file=sys.stderr)
    elif getattr(config, "stream", False):
        if answer and not answer.endswith("\n"):
            print()
    else:
        print(answer)

    try:
        with store.locked():
            sid = _seal_session(
                config, store, workdir, agent, task, head,
                message=message, error=error,
            )
            result["session_id"] = sid
            result["changed_files"] = agent.stats.get("changed_files", [])
    except (StoreError, OSError) as exc:
        result["status"] = "error"
        result["error"] = f"could not save session: {exc}"
        if json_output:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(result["error"], file=sys.stderr)
        return 1

    if json_output:
        print(json.dumps(result, ensure_ascii=False))
    if not getattr(config, "quiet", False):
        status = "failed" if error else ""
        footer = f"[session {result['session_id']}]{(' ' + status) if status else ''} on branch {store.current_branch() or '(detached)'}"
        if agent.changed_files:
            footer += " · changed " + ", ".join(sorted(agent.changed_files))
        print(footer, file=sys.stderr)
    return 1 if error else 0


# -- commands ---------------------------------------------------------------
def _read_context_file(path: str, workdir: str) -> str | None:
    p = Path(path)
    if not p.is_absolute():
        candidate = Path(workdir) / p
        if candidate.is_file():
            p = candidate
        else:
            candidate = Path.cwd() / p
            if candidate.is_file():
                p = candidate
    if not p.is_file():
        print(f"error: context file not found: {path}", file=sys.stderr)
        return None
    try:
        data = p.read_bytes()
    except OSError as exc:
        print(f"error: cannot read context file {path}: {exc}", file=sys.stderr)
        return None
    data = data[: 128 * 1024]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", "replace")
    return text


def _expand_task_references(task: str, workdir: str) -> str:
    """Expand ``@relative/path`` references in a task into file contents."""
    def replace(match: re.Match[str]) -> str:
        raw = match.group(1).rstrip(".,;:!?")
        p = Path(raw)
        if not p.is_absolute():
            candidate = Path(workdir) / p
            if not candidate.is_file():
                candidate = Path.cwd() / p
        else:
            candidate = p
        if not candidate.is_file():
            return match.group(0)
        try:
            data = candidate.read_bytes()[: 128 * 1024]
        except OSError:
            return match.group(0)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", "replace")
        return f"\n\n[Attached file: {raw}]\n{text}\n[/Attached file]"

    return re.sub(r"@([^\s]+)", replace, task)


def _build_task_text(task: str, workdir: str, context_files: list[str] | None) -> str | None:
    """Attach explicit --file contents and expand @path references."""
    text = _expand_task_references(task, workdir)
    if not context_files:
        return text
    for path in context_files:
        content = _read_context_file(path, workdir)
        if content is None:
            return None
        text += f"\n\n[Attached file: {path}]\n{content}\n[/Attached file]"
    return text


def _run(opts: argparse.Namespace, config: Any, task: str) -> int:
    if not task or not task.strip():
        print("error: no task provided (pass it as an argument or via stdin)", file=sys.stderr)
        return 2
    if not config.api_key:
        _no_api_key()
        return 2
    store = SessionStore(config.workspace)
    if getattr(opts, "resume", None):
        try:
            with store.locked():
                store.checkout(opts.resume, restore=True)
        except (StoreError, OSError) as exc:
            print(f"error: cannot resume {opts.resume!r}: {exc}", file=sys.stderr)
            return 1
    if getattr(opts, "continue_run", False) and store.resolve_head() is None:
        print("error: --continue requested but there is no current session", file=sys.stderr)
        return 1
    workdir = _resolve_workdir(config, opts, store)
    task = _build_task_text(task, workdir, getattr(opts, "file", None))
    if task is None:
        return 1
    return _run_task(
        config, store, workdir, task,
        message=opts.message,
        json_output=getattr(opts, "json_output", False),
    )


def _no_api_key() -> None:
    print(
        "error: no API key configured. Set LLM_API_KEY (or OPENAI_API_KEY) in "
        "the environment, or put it in the workspace config.json — see README.md.",
        file=sys.stderr,
    )


def _check_sandbox(config: Any) -> bool:
    """Fail closed if --sandbox was requested but no backend is available."""
    if config.sandbox and detect_sandbox_backend() is None:
        print(
            "error: --sandbox requires bwrap (bubblewrap) or firejail, "
            "neither was found; install one or drop --sandbox",
            file=sys.stderr,
        )
        return False
    return True


def _fmt_duration(ms: Any) -> str:
    try:
        seconds = float(ms or 0) / 1000
    except (TypeError, ValueError):
        return ""
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


def _fmt_tokens(meta: dict[str, Any]) -> str:
    usage = meta.get("usage") or {}
    total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    return f"{total} tok" if total else ""


def _fmt_changes(meta: dict[str, Any]) -> str:
    files = meta.get("changed_files") or []
    return f"{len(files)} file{'s' if len(files) != 1 else ''}" if files else ""


def _status_mark(meta: dict[str, Any]) -> str:
    status = meta.get("status", "ok")
    if status == "failed":
        return "✗ failed"
    if status == "ok" and meta.get("mode") == "plan":
        return "plan"
    return "ok"


def _print_changes(changes: list[dict[str, str]]) -> None:
    for change in changes:
        print(f"  {change['status']} {change['path']}")


def _read_small_text(path: Path) -> str:
    try:
        data = path.read_bytes()[: 256 * 1024]
    except OSError:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", "replace")


def _print_file_diff(store: SessionStore, sid: str, change: dict[str, str]) -> None:
    rel = change["path"]
    meta = store.load_meta(sid)
    parent = meta.get("parent")
    old_path = store.sessions_dir / parent / "artifacts" / rel if parent else None
    new_path = store.sessions_dir / sid / "artifacts" / rel
    old_text = _read_small_text(old_path) if old_path and old_path.is_file() else ""
    new_text = _read_small_text(new_path) if new_path.is_file() else ""
    diff = difflib.unified_diff(
        old_text.splitlines(), new_text.splitlines(),
        fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="",
    )
    for line in diff:
        print(f"    {line}")


def _status(store: SessionStore) -> int:
    branch = store.current_branch()
    head = store.resolve_head()
    branches = store.list_branches()
    print(f"workspace : {store.root}")
    print(f"branch    : {branch or '(detached HEAD)'}")
    print(f"HEAD      : {head or '(none)'}")
    print(f"sessions  : {len(store.list_sessions())}")
    print(f"branches  : {', '.join(sorted(branches)) or '(none)'}")
    if head:
        meta = store.load_meta(head)
        print(f"message   : {meta.get('message') or meta.get('task', '')}")
        print(f"status    : {_status_mark(meta)}")
        print(f"duration  : {_fmt_duration(meta.get('duration_ms'))}")
        print(f"tokens    : {_fmt_tokens(meta) or '-'}")
        print(f"changed   : {_fmt_changes(meta) or '-'}")
        changes = store.diff_workdir(head)
        if changes:
            print("work/ vs HEAD snapshot:")
            _print_changes(changes)
        else:
            print("work/     : clean vs HEAD snapshot")
    return 0


def _log(store: SessionStore, all_branches: bool, oneline: bool, graph: bool = False) -> int:
    if graph:
        for line in store.graph(all_branches):
            print(line)
        if not store.list_sessions():
            print("no sessions yet")
        return 0
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
        meta_extra = " ".join(
            x for x in (_fmt_duration(m.get("duration_ms")), _fmt_tokens(m), _fmt_changes(m)) if x
        )
        if oneline:
            print(f"{_status_mark(m):9} {sid}  {m.get('message', '')}{tag}  {meta_extra}".rstrip())
        else:
            created = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.get("created_at", 0)))
            print(f"{_status_mark(m)} {sid}  {m.get('message', '')}{tag}")
            print(f"    parent: {m.get('parent') or '-'}   at {created}   model: {m.get('model', '')}")
            if meta_extra:
                print(f"    {meta_extra}")
            if m.get("error"):
                print(f"    error: {str(m['error'])[:200]}")
            for line in (m.get("task") or "").splitlines()[:2]:
                if line.strip():
                    print(f"      {line.strip()}")
    if not entries:
        print("no sessions yet")
    return 0


def _show(store: SessionStore, ref: str, tail: int, show_diff: bool = False) -> int:
    try:
        sid = store.resolve_ref(ref)
        meta = store.load_meta(sid)
        msgs = store.load_conversation(sid)
    except (StoreError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"session {sid}  {meta.get('message', '')}  [{_status_mark(meta)}]")
    print(f"model {meta.get('model') or '-'} · duration {_fmt_duration(meta.get('duration_ms')) or '-'} · tokens {_fmt_tokens(meta) or '-'}")
    changes = store.diff_session(sid)
    if changes:
        print(f"changed {len(changes)} file(s):")
        _print_changes(changes)
    else:
        print("no artifact changes vs parent")
    if show_diff and changes:
        print("\n--- artifact diff ---")
        for change in changes:
            print(f"{change['status']} {change['path']}")
            if change["status"] == "M":
                _print_file_diff(store, sid, change)
    if not msgs:
        print("\n(empty session)")
        return 0
    print("\n--- conversation ---")
    for m in msgs[-tail:]:
        role = m["role"]
        content = m.get("content") or ""
        if role == "assistant" and m.get("tool_calls"):
            names = ", ".join(t["function"]["name"] for t in m["tool_calls"])
            print(f"[{role}] -> {names}")
        elif role == "tool":
            text = content.replace("\n", " ")
            if len(text) > 300:
                text = text[:300] + "..."
            print(f"[tool] {text}")
        else:
            content = content.replace("\n", "\n        ")
            if len(content) > 600:
                content = content[:600] + "..."
            print(f"[{role}] {content}")
    return 0


def _switch(store: SessionStore, ref: str, restore: bool) -> int:
    try:
        with store.locked():
            sid = store.checkout(ref, restore=restore)
    except (StoreError, OSError) as exc:
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
            with store.locked():
                store.delete_branch(args[1])
        except (StoreError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"deleted branch {args[1]}")
        return 0
    name = args[0]
    if name.startswith("-"):
        print("usage: branch [<name>] | branch -d <name>", file=sys.stderr)
        return 1
    try:
        with store.locked():
            store.create_branch(name)
    except (StoreError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"branch {name} -> {store.resolve_head()}")
    return 0


def _rm(store: SessionStore, args: list[str]) -> int:
    if not args:
        print("usage: rm <session-id-or-prefix>", file=sys.stderr)
        return 1
    try:
        with store.locked():
            store.delete_session(args[0])
    except (StoreError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"deleted session {args[0]}")
    return 0


def _init(store: SessionStore) -> int:
    print(f"workspace ready at {store.root}")
    return 0


def _repl(opts: argparse.Namespace, config: Any, store: SessionStore, workdir: str) -> int:
    branch = store.current_branch() or "(detached)"
    print(_REPL_BANNER, file=sys.stderr)
    if not getattr(config, "quiet", False):
        print(f"[repl] branch {branch} · {workdir}", file=sys.stderr)
    prompt = f"{branch}> "
    head = store.resolve_head()
    history = store.load_conversation(head) if head else None
    agent = _build_agent(config, workdir, history=history)
    initial_len = len(agent.messages)
    first_task = ""
    last_error = ""

    def checkpoint() -> bool:
        """Save accumulated turns as one session and continue from it."""
        nonlocal head, agent, initial_len, first_task, last_error
        if len(agent.messages) <= initial_len:
            print("(nothing to save)", file=sys.stderr)
            return True
        try:
            with store.locked():
                sid = _seal_session(
                    config, store, workdir, agent,
                    task=first_task or "repl checkpoint",
                    parent=head, message=opts.message, error=last_error,
                )
        except (StoreError, OSError) as exc:
            print(f"error: could not save session: {exc}", file=sys.stderr)
            return False
        print(f"[session {sid}]", file=sys.stderr)
        head = sid
        history = store.load_conversation(sid)
        agent = _build_agent(config, workdir, history=history)
        initial_len = len(agent.messages)
        first_task = ""
        last_error = ""
        return True

    while True:
        try:
            line = input(prompt)
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
        if s == "/save":
            checkpoint()
            continue
        if s == "/tokens":
            print(
                f"tokens: prompt={agent.usage.get('prompt_tokens', 0)} "
                f"completion={agent.usage.get('completion_tokens', 0)} "
                f"total={agent.total_tokens} · "
                f"estimated context={agent._estimate_tokens(agent.messages)}",
                file=sys.stderr,
            )
            continue
        if s == "/diff":
            changes = store.diff_workdir(head)
            if changes:
                print("work/ vs HEAD snapshot:", file=sys.stderr)
                _print_changes(changes)
            else:
                print("work/ clean vs HEAD snapshot", file=sys.stderr)
            continue
        if s == "/undo":
            if head and store.restore_artifacts(head):
                print("work/ restored from HEAD snapshot", file=sys.stderr)
            else:
                print("no HEAD snapshot available", file=sys.stderr)
            continue
        if s == "/compact":
            before, after = agent.compact_now()
            print(
                f"context compacted: {before} -> {after} messages", file=sys.stderr
            )
            continue
        if s == "/skills":
            if agent.skills:
                for skill in agent.skills:
                    print(f"  {skill.name}: {skill.description}", file=sys.stderr)
            else:
                print("no skills loaded", file=sys.stderr)
            continue
        if s.startswith("/checkout"):
            parts = s.split(maxsplit=1)
            if len(parts) < 2:
                print("usage: /checkout REF", file=sys.stderr)
                continue
            if len(agent.messages) > initial_len:
                checkpoint()
            try:
                with store.locked():
                    store.checkout(parts[1], restore=True)
            except (StoreError, OSError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                continue
            head = store.resolve_head()
            history = store.load_conversation(head) if head else None
            agent = _build_agent(config, workdir, history=history)
            initial_len = len(agent.messages)
            first_task = ""
            last_error = ""
            prompt = f"{store.current_branch() or '(detached)'}> "
            print(f"HEAD -> {head}", file=sys.stderr)
            continue
        if s in ("/help", "help"):
            print(_REPL_BANNER, file=sys.stderr)
            continue
        if s.startswith("/"):
            print(f"unknown command {s!r}", file=sys.stderr)
            continue
        if not first_task:
            first_task = s
        try:
            answer = agent.run(s)
        except AgentError as exc:
            last_error = str(exc)
            print(f"error: {exc}", file=sys.stderr)
            continue
        if getattr(config, "stream", False):
            if answer and not answer.endswith("\n"):
                print()
        else:
            print(answer)

    # Seal the whole REPL interaction as ONE session (multi-turn, single commit).
    if len(agent.messages) > initial_len:
        try:
            with store.locked():
                sid = _seal_session(
                    config, store, workdir, agent,
                    task=first_task or "repl",
                    parent=head, message=opts.message, error=last_error,
                )
        except (StoreError, OSError) as exc:
            print(f"error: could not save session: {exc}", file=sys.stderr)
            return 1
        print(f"[session {sid}]", file=sys.stderr)
    return 0


def _mask_config(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    if out.get("api_key"):
        out["api_key"] = "<set>" if len(str(out["api_key"])) > 4 else "<set>"
    if "extra_headers" in out and isinstance(out.get("extra_headers"), dict):
        masked = dict(out["extra_headers"])
        for name in list(masked):
            if name.lower() in ("authorization", "api-key", "x-api-key"):
                masked[name] = "<set>"
        out["extra_headers"] = masked
    return out


def _doctor(opts: argparse.Namespace, config: Any) -> int:
    store = SessionStore(config.workspace)
    print(f"python      : {sys.version.split()[0]}")
    print(f"workspace   : {store.root}")
    print(f"workdir     : {store.work_dir}")
    print(f"config      : {store.root / 'config.json'}")
    print(f"api key     : {'configured' if config.api_key else 'missing (set LLM_API_KEY or config.json)'}")
    print(f"endpoint    : {config.base_url}")
    print(f"model       : {config.model}")
    print(f"sandbox     : {detect_sandbox_backend() or 'none (install bwrap/firejail for --sandbox)'}")
    print(f"ripgrep     : {shutil.which('rg') or 'not found (recommended for large-repo grep)'}")
    print(f"sessions    : {len(store.list_sessions())}")
    print(f"branches    : {', '.join(sorted(store.list_branches())) or '(none)'}")
    for path in (store.root, store.work_dir):
        print(f"writable    : {path} -> {'yes' if os.access(path, os.W_OK) else 'no'}")
    errors = validate_config(config)
    if config.sandbox and detect_sandbox_backend() is None:
        errors.append("--sandbox requested but bwrap/firejail is unavailable")
    if errors:
        print("config      : invalid")
        for exc in errors:
            print(f"  - {exc}")
        return 1
    print("config      : valid")
    return 0


def _config_command(store: SessionStore, args: list[str]) -> int:
    path = store.root / "config.json"
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"error: invalid workspace config {path}: {exc}", file=sys.stderr)
            return 1
    if not isinstance(data, dict):
        data = {}

    def save() -> int:
        try:
            store._write_json_atomic(path, data)  # atomic config write
        except OSError as exc:
            print(f"error: cannot write config {path}: {exc}", file=sys.stderr)
            return 1
        return 0

    if not args or args[0] == "show":
        print(f"# {path}")
        print(json.dumps(_mask_config(data), indent=2, ensure_ascii=False))
        return 0
    if args[0] == "path":
        print(path)
        return 0
    if args[0] == "get":
        if len(args) < 2:
            print("usage: config get <key>", file=sys.stderr)
            return 1
        key = args[1]
        if key not in data:
            print(f"error: {key} is not set in {path}", file=sys.stderr)
            return 1
        value = "<set>" if key == "api_key" and data.get(key) else data[key]
        print(json.dumps(value, ensure_ascii=False))
        return 0
    if args[0] == "set":
        if len(args) < 3:
            print("usage: config set <key> <json-or-string>", file=sys.stderr)
            return 1
        key, raw = args[1], args[2]
        if key not in {f.name for f in Config.__dataclass_fields__.values()} and key != "workspace":
            print(f"error: unknown config key {key!r}", file=sys.stderr)
            return 1
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        data[key] = value
        return save()
    if args[0] == "unset":
        if len(args) < 2:
            print("usage: config unset <key>", file=sys.stderr)
            return 1
        data.pop(args[1], None)
        return save()
    print("usage: config [show|path|get KEY|set KEY VALUE|unset KEY]", file=sys.stderr)
    return 1


def _completion(shell: str) -> int:
    commands = " ".join(sorted(KNOWN_COMMANDS))
    if shell == "bash":
        print(f"complete -W '{commands}' coding-agent")
        return 0
    if shell == "zsh":
        choices = " ".join(commands.split())
        print(f"#compdef coding-agent\n_arguments '1:command:(({choices}))'")
        return 0
    if shell == "fish":
        print(f"complete -c coding-agent -f -a '{commands}'")
        return 0
    print("error: supported shells: bash, zsh, fish", file=sys.stderr)
    return 1


def _print_tools() -> None:
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        print(f"{fn['name']}: {fn['description']}")


# -- dispatch ---------------------------------------------------------------
def _dispatch(opts: argparse.Namespace, cmd: str, cmd_args: list[str]) -> int:
    overrides = _cli_overrides(opts)
    header_error = overrides.pop("_header_error", None)
    if header_error:
        print(f"error: {header_error}", file=sys.stderr)
        return 2
    config = load_config(overrides)
    config_errors = validate_config(config)
    if config_errors:
        for exc in config_errors:
            print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2

    if cmd == "run":
        if not _check_sandbox(config):
            return 2
        task = " ".join(cmd_args).strip()
        if not task and not sys.stdin.isatty():
            task = sys.stdin.read().strip()
        return _run(opts, config, task)

    if cmd == "init":
        return _init(SessionStore(config.workspace))
    if cmd == "status":
        return _status(SessionStore(config.workspace))
    if cmd == "log":
        return _log(SessionStore(config.workspace), "--all" in cmd_args, "--oneline" in cmd_args, "--graph" in cmd_args)
    if cmd == "show":
        ref = next((a for a in cmd_args if not a.startswith("-")), None)
        if not ref:
            print("usage: show <ref> [--all] [--diff]", file=sys.stderr)
            return 1
        tail = 100000 if "--all" in cmd_args else 20
        return _show(
            SessionStore(config.workspace), ref, tail,
            show_diff="--diff" in cmd_args,
        )
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
    if cmd == "doctor":
        return _doctor(opts, config)
    if cmd == "config":
        return _config_command(SessionStore(config.workspace), cmd_args)
    if cmd == "completion":
        return _completion(cmd_args[0] if cmd_args else "bash")
    if cmd == "repl":
        if not _check_sandbox(config):
            return 2
        if not config.api_key:
            _no_api_key()
            return 2
        store = SessionStore(config.workspace)
        return _repl(opts, config, store, _resolve_workdir(config, opts, store))

    print(f"unknown command {cmd!r}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Subcommand help: argparse would otherwise print only the global help.
    if "-h" in argv or "--help" in argv:
        help_pos = min(
            [i for i, a in enumerate(argv) if a in ("-h", "--help")] or [len(argv)]
        )
        for cmd in KNOWN_COMMANDS:
            if cmd in argv[:help_pos]:
                print(_SUBCOMMAND_HELP.get(cmd, "no help available"), end="")
                return 0

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
