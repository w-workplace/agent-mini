"""Tool definitions and local executors.

Tools are declared as OpenAI tool-calling JSON schemas and executed locally in
this process — never delegated to a hosted code-execution or file service.
Every executor returns a plain dict that the agent serializes to a JSON string
and feeds back to the model, so tool failures are visible to the model and it
can self-correct.
"""

from __future__ import annotations

import fnmatch
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Any

from .security import build_sandbox_argv, command_env, detect_sandbox_backend, redact_obj

# ---------------------------------------------------------------------------
# Output limits (keep tool results bounded so they don't blow up the context).
# ---------------------------------------------------------------------------
MAX_LIST_FILES = 300
MAX_GREP_MATCHES = 300
MAX_GREP_FILE_BYTES = 1024 * 1024  # skip files larger than this in grep
MAX_READ_BYTES = 256 * 1024        # full-file fast path threshold
MAX_READ_LINES = 2000             # lines returned by read_file when paging a large file
MAX_OUTPUT_BYTES = 50 * 1024       # stdout/stderr kept from a command
MAX_LIVE_OUTPUT_BYTES = 256 * 1024 # live command output echoed to stderr
MAX_GREP_LINE_LEN = 500


class ToolError(Exception):
    """A tool-level failure to report back to the model."""


# Commands we refuse to run unless ``allow_dangerous_commands`` is enabled.
# This is a conservative, best-effort guardrail — not a security boundary.
DANGEROUS_PATTERNS: list[str] = [
    r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/(\s|$)",   # rm -rf /
    r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/\*(\s|$)",  # rm -rf /*
    r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+~(\s|$)",    # rm -rf ~
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\b[^\n]*\bof=/dev/",
    r"\b(sudo\s+)?shutdown\b",
    r"\b(sudo\s+)?reboot\b",
    r"\b(sudo\s+)?poweroff\b",
    r":\(\)\s*\{\s*:\|:&\s*\}\s*;",  # fork bomb
]

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
    "build", ".tox", ".mypy_cache", ".pytest_cache", ".eggs", ".idea",
}


class _BoundedTail:
    """A small in-memory buffer that keeps only the most recent characters."""

    def __init__(self, max_chars: int):
        self.max_chars = max_chars
        self.parts: deque[str] = deque()
        self.size = 0
        self.truncated = False

    def write(self, text: str) -> None:
        if not text:
            return
        self.parts.append(text)
        self.size += len(text)
        if self.size > self.max_chars:
            self.truncated = True
            while self.parts and self.size > self.max_chars:
                part = self.parts.popleft()
                self.size -= len(part)

    def value(self) -> str:
        return "".join(self.parts)


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI tool-calling format).
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and directories matching a glob pattern under a "
                "directory. Directories are suffixed with '/'. Use this to "
                "explore the project layout before reading or editing files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '**/*.py' or 'src/*.ts'. Defaults to '*'.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search. Defaults to the working directory.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file and return its contents with line "
                "numbers. Large files return at most 2000 lines per call; "
                "use offset/limit to page through them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."},
                    "offset": {"type": "integer", "description": "1-based first line to return."},
                    "limit": {"type": "integer", "description": "Maximum number of lines to return."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a file or overwrite its full contents. Parent "
                "directories are created as needed. Use this to create new "
                "files or rewrite a file entirely."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to write."},
                    "content": {"type": "string", "description": "Full UTF-8 contents to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace an exact string in a file with a new string. By "
                "default the old string must occur exactly once; set "
                "replace_all=true to replace every occurrence. Prefer this over "
                "write_file for small, targeted edits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to edit."},
                    "old_string": {"type": "string", "description": "Exact text to replace."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences instead of exactly one.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search files for a regular expression and return matching "
                "lines as 'path:line:content'. Searches a single file, or "
                "recursively under a directory (skipping common build/VCS dirs)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "path": {"type": "string", "description": "File or directory to search. Defaults to the working directory."},
                    "include": {"type": "string", "description": "Optional glob filter for filenames, e.g. '*.py'."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the working directory and return its "
                "exit code, stdout and stderr. Use this to run tests, builds, "
                "git, or any verification step. Commands are killed after a "
                "timeout; a few obviously destructive commands are blocked by "
                "default."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."},
                    "timeout": {
                        "type": "number",
                        "description": "Optional timeout in seconds (capped by the configured limit).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parallel_search",
            "description": (
                "Delegate several independent research/exploration subtasks to "
                "isolated read-only subagents that run in parallel and return "
                "concise summaries. Use this to explore a large codebase (find "
                "where something is defined, how modules are structured) without "
                "polluting the main conversation with large file contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of independent, self-contained subtasks/questions.",
                    },
                },
                "required": ["queries"],
            },
        },
    },
]


# The set of tool names the model may invoke. ``execute`` only ever dispatches
# to these, via an explicit name -> method map (no `getattr` on model input).
_EXECUTOR_NAMES = frozenset({
    "list_files", "read_file", "write_file", "edit_file", "grep", "run_command",
})
_READ_ONLY_NAMES = frozenset({"list_files", "read_file", "grep"})


class ToolRunner:
    """Executes tool calls locally, scoped to a working directory."""

    def __init__(
        self,
        workdir: str = ".",
        allow_outside_workdir: bool = False,
        allow_dangerous_commands: bool = False,
        command_timeout: float = 120.0,
        sandbox: bool = False,
        env_allow: str = "",
        read_only: bool = False,
        stream: bool = True,
        quiet: bool = False,
        subagent_executor: Any = None,
    ):
        self.workdir = Path(workdir).resolve()
        self.allow_outside_workdir = allow_outside_workdir
        self.allow_dangerous_commands = allow_dangerous_commands
        self.command_timeout = command_timeout
        self.sandbox = sandbox
        self.env_allow = env_allow
        self.read_only = read_only
        self.stream = stream
        self.quiet = quiet
        self.subagent_executor = subagent_executor
        self.sandbox_backend = detect_sandbox_backend() if sandbox else None
        self._schemas = {s["function"]["name"]: s["function"] for s in TOOL_SCHEMAS}
        self._executors: dict[str, Any] = {}
        self._rebuild_executors()

    def _rebuild_executors(self) -> None:
        """Explicit allowlist: only these callables are reachable from the model."""
        names: set[str] = set(_READ_ONLY_NAMES) if self.read_only else set(_EXECUTOR_NAMES)
        if not self.read_only and self.subagent_executor is not None:
            names.add("parallel_search")
        self._executors = {name: getattr(self, name) for name in names}

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [s for s in TOOL_SCHEMAS if s["function"]["name"] in self._executors]

    def set_subagent_executor(self, executor: Any) -> None:
        """Enable the ``parallel_search`` tool by wiring a subagent executor."""
        self.subagent_executor = executor
        if not self.read_only:
            self._rebuild_executors()

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _ok(**kwargs: Any) -> dict[str, Any]:
        d: dict[str, Any] = {"ok": True}
        d.update(kwargs)
        return d

    @staticmethod
    def _err(message: str) -> dict[str, Any]:
        return {"ok": False, "error": message}

    def resolve_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.workdir / p
        p = p.resolve()
        if not self.allow_outside_workdir and not p.is_relative_to(self.workdir):
            raise ToolError(
                f"path {path!r} escapes the working directory {str(self.workdir)!r}"
            )
        return p

    def _display_path(self, p: Path) -> str:
        """Return a tool-result path relative to the workdir when possible.

        Absolute paths leak the local workspace layout into the model context
        and the session log; relative paths keep both portable.
        """
        try:
            rel = p.relative_to(self.workdir)
        except ValueError:
            return str(p)
        return str(rel) if str(rel) not in ("", ".") else str(p.name)

    def _atomic_write_text(self, p: Path, text: str) -> None:
        """Write ``text`` atomically (temp file + rename) and preserve mode."""
        p.parent.mkdir(parents=True, exist_ok=True)
        old_mode: int | None = None
        if p.exists():
            old_mode = p.stat().st_mode & 0o7777
        fd, tmp = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            if old_mode is not None:
                os.chmod(tmp, old_mode)
            else:
                current_umask = os.umask(0)
                os.umask(current_umask)
                os.chmod(tmp, 0o666 & ~current_umask)
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- individual tools ----------------------------------------------------
    def list_files(self, pattern: str = "*", path: str = ".") -> dict[str, Any]:
        try:
            root = self.resolve_path(path)
        except ToolError as exc:
            return self._err(str(exc))
        if not root.is_dir():
            return self._err(f"not a directory: {path}")
        try:
            matches = sorted(glob.glob(str(root / pattern), recursive=True))
        except re.error as exc:
            return self._err(f"bad glob pattern: {exc}")
        files: list[str] = []
        for m in matches:
            p = Path(m)
            files.append(self._display_path(p) + ("/" if p.is_dir() else ""))
            if len(files) >= MAX_LIST_FILES:
                break
        return self._ok(
            files=files,
            count=len(files),
            truncated=len(matches) > MAX_LIST_FILES,
        )

    @staticmethod
    def _render_lines(lines: list[str], start: int) -> str:
        return "".join(
            f"{i:6}\t{line}\n" for i, line in enumerate(lines, start=start)
        )

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> dict[str, Any]:
        try:
            p = self.resolve_path(path)
        except ToolError as exc:
            return self._err(str(exc))
        if p.is_dir():
            return self._err(f"is a directory, not a file: {path}")
        try:
            size = p.stat().st_size
        except OSError as exc:
            return self._err(f"cannot read {path}: {exc}")

        # Fast path: small files keep exact ``total_lines`` semantics.
        if size <= MAX_READ_BYTES:
            try:
                data = p.read_bytes()
            except OSError as exc:
                return self._err(f"cannot read {path}: {exc}")
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("utf-8", "replace")
            lines = text.splitlines()
            total = len(lines)
            start = max(1, int(offset or 1))
            if limit is not None:
                end = start + max(0, int(limit)) - 1
            else:
                end = total
            shown = lines[start - 1 : end]
            return self._ok(
                path=self._display_path(p),
                content=self._render_lines(shown, start),
                total_lines=total,
                lines_returned=len(shown),
                truncated=False,
                file_bytes=size,
            )
        return self._read_large_file(p, offset, limit, size)

    def _read_large_file(
        self,
        p: Path,
        offset: int | None,
        limit: int | None,
        size: int,
    ) -> dict[str, Any]:
        """Page through files larger than MAX_READ_BYTES without loading them.

        Lines are streamed and counted from the start of the file, so an
        arbitrary 1-based ``offset`` works even past the first chunk.
        ``limit`` is capped to keep one tool result bounded.
        """
        start = max(1, int(offset or 1))
        if limit is None:
            limit = MAX_READ_LINES
        else:
            limit = max(0, min(int(limit), MAX_READ_LINES))

        shown: list[str] = []
        total = 0
        scanned_to_end = False
        try:
            with p.open("r", encoding="utf-8", errors="replace", newline=None) as f:
                for line in f:
                    total += 1
                    if total >= start and len(shown) < limit:
                        shown.append(line.rstrip("\r\n"))
                    if len(shown) >= limit and total >= start + limit - 1:
                        break
                else:
                    scanned_to_end = True
        except OSError as exc:
            return self._err(f"cannot read {p}: {exc}")

        result: dict[str, Any] = {
            "path": self._display_path(p),
            "content": self._render_lines(shown, start),
            "lines_returned": len(shown),
            "truncated": True,
            "file_bytes": size,
        }
        if scanned_to_end:
            result["total_lines"] = total
        return self._ok(**result)

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        try:
            p = self.resolve_path(path)
        except ToolError as exc:
            return self._err(str(exc))
        if p.is_dir():
            return self._err(f"is a directory, not a file: {path}")
        data = content.encode("utf-8")
        try:
            self._atomic_write_text(p, content)
        except OSError as exc:
            return self._err(f"cannot write {path}: {exc}")
        return self._ok(path=self._display_path(p), bytes_written=len(data))

    def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        try:
            p = self.resolve_path(path)
        except ToolError as exc:
            return self._err(str(exc))
        if old_string == "":
            return self._err("old_string must be non-empty")
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            return self._err(f"cannot read {path}: {exc}")
        count = text.count(old_string)
        if count == 0:
            return self._err(f"old_string not found in {path}")
        if count > 1 and not replace_all:
            return self._err(
                f"old_string appears {count} times; use a more specific string "
                "or pass replace_all=true to replace every occurrence"
            )
        replacements = count if replace_all else 1
        new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        try:
            self._atomic_write_text(p, new_text)
        except OSError as exc:
            return self._err(f"cannot write {path}: {exc}")
        return self._ok(path=self._display_path(p), replacements=replacements)

    def grep(self, pattern: str, path: str = ".", include: str | None = None) -> dict[str, Any]:
        try:
            root = self.resolve_path(path)
        except ToolError as exc:
            return self._err(str(exc))
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return self._err(f"invalid regex: {exc}")

        if root.is_file():
            targets = [root]
        elif root.is_dir():
            targets = self._walk(root, include)
        else:
            return self._err(f"not found: {path}")

        matches: list[str] = []
        skipped_files = 0
        for fpath in targets:
            try:
                if fpath.stat().st_size > MAX_GREP_FILE_BYTES:
                    skipped_files += 1
                    continue
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                skipped_files += 1
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    shown = line if len(line) <= MAX_GREP_LINE_LEN else line[:MAX_GREP_LINE_LEN] + "..."
                    matches.append(f"{self._display_path(fpath)}:{lineno}:{shown}")
                    if len(matches) >= MAX_GREP_MATCHES:
                        break
            if len(matches) >= MAX_GREP_MATCHES:
                break
        return self._ok(
            matches=matches,
            count=len(matches),
            truncated=len(matches) >= MAX_GREP_MATCHES,
            skipped_files=skipped_files,
        )

    def _walk(self, root: Path, include: str | None) -> list[Path]:
        results: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fn in filenames:
                if include and not fnmatch.fnmatch(fn, include):
                    continue
                results.append(Path(dirpath) / fn)
                if len(results) >= 20000:
                    return results
        return results

    def run_command(self, command: str, timeout: float | None = None) -> dict[str, Any]:
        if self._is_dangerous(command):
            return self._err(
                "command matches a blocked dangerous-command pattern; "
                "if this is intended, run the agent with --allow-dangerous-commands"
            )
        if self.sandbox and not self.sandbox_backend:
            return self._err(
                "sandbox requested but neither bwrap nor firejail is available; "
                "install one or drop --sandbox"
            )
        if timeout is None:
            limit = self.command_timeout
        else:
            try:
                limit = max(0.0, min(float(timeout), self.command_timeout))
            except (TypeError, ValueError):
                return self._err(f"invalid timeout: {timeout!r}")
        if limit <= 0:
            return self._err("command timeout must be greater than zero")

        # Scrub the environment: never pass API keys/tokens to child commands.
        env = command_env(str(self.workdir), sandbox=self.sandbox, extra_allow=self.env_allow)

        if self.sandbox:
            argv = build_sandbox_argv(self.sandbox_backend, str(self.workdir), command)
            run_kwargs = dict(args=argv, shell=False)
        else:
            run_kwargs = dict(args=command, shell=True)

        try:
            exit_code, out, err, truncated, live_truncated = self._run_piped(
                run_kwargs, env, limit, live=self.stream and not self.quiet
            )
        except subprocess.TimeoutExpired:
            return self._err(f"command timed out after {limit}s")
        except OSError as exc:
            return self._err(f"failed to run command: {exc}")

        if live_truncated and not self.quiet:
            print(
                f"[command output truncated after {MAX_LIVE_OUTPUT_BYTES} bytes]",
                file=sys.stderr,
                flush=True,
            )
        res = self._ok(exit_code=exit_code, stdout=out, stderr=err)
        if truncated:
            res["truncated"] = True
        return res

    def _run_piped(
        self,
        run_kwargs: dict[str, Any],
        env: dict[str, str],
        limit: float,
        live: bool,
    ) -> tuple[int, str, str, bool, bool]:
        """Run a command with bounded, optionally streamed, pipe capture.

        Both stdout and stderr are drained from dedicated threads (no
        deadlock), captured tails are capped at MAX_OUTPUT_BYTES, live echo is
        capped at MAX_LIVE_OUTPUT_BYTES, and pipe handles are always closed.
        """
        proc = subprocess.Popen(
            cwd=str(self.workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            **run_kwargs,
        )
        out_tail = _BoundedTail(MAX_OUTPUT_BYTES)
        err_tail = _BoundedTail(MAX_OUTPUT_BYTES)
        live_counts = {"stdout": 0, "stderr": 0}
        live_truncated = False

        def pump(pipe: Any, tail: _BoundedTail, name: str) -> None:
            nonlocal live_truncated
            try:
                while True:
                    chunk = pipe.read(4096)
                    if chunk == "":
                        break
                    tail.write(chunk)
                    if live:
                        allowed = MAX_LIVE_OUTPUT_BYTES - live_counts[name]
                        if allowed > 0:
                            piece = chunk if len(chunk) <= allowed else chunk[:allowed]
                            sys.stderr.write(piece)
                            sys.stderr.flush()
                            live_counts[name] += len(piece)
                        if live_counts[name] >= MAX_LIVE_OUTPUT_BYTES:
                            live_truncated = True
            finally:
                try:
                    pipe.close()
                except OSError:
                    pass

        assert proc.stdout is not None and proc.stderr is not None
        t1 = threading.Thread(target=pump, args=(proc.stdout, out_tail, "stdout"), daemon=True)
        t2 = threading.Thread(target=pump, args=(proc.stderr, err_tail, "stderr"), daemon=True)
        t1.start()
        t2.start()
        try:
            proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            t1.join(timeout=5)
            t2.join(timeout=5)
            raise
        t1.join(timeout=5)
        t2.join(timeout=5)
        return (
            proc.returncode,
            out_tail.value(),
            err_tail.value(),
            out_tail.truncated or err_tail.truncated,
            live_truncated,
        )

    def _is_dangerous(self, command: str) -> bool:
        if self.allow_dangerous_commands:
            return False
        cmd = command or ""
        return any(re.search(pat, cmd) for pat in DANGEROUS_PATTERNS)

    # -- dispatch ------------------------------------------------------------
    def _validate_arguments(self, name: str, arguments: Any) -> str | None:
        """Validate a tool call's arguments against its declared JSON schema.

        Returns an error message, or ``None`` if valid. This treats model
        output as untrusted input (defense against tool confusion / injection).
        """
        schema = self._schemas.get(name)
        if schema is None:
            return f"unknown tool: {name}"
        if not isinstance(arguments, dict):
            return f"arguments for {name} must be a JSON object"
        props = schema.get("parameters", {}).get("properties", {})
        required = schema.get("parameters", {}).get("required", [])

        for key in arguments:
            if key not in props:
                return f"{name}: unknown argument {key!r}"
        for key in required:
            if arguments.get(key) is None:
                return f"{name}: missing required argument {key!r}"

        for key, value in arguments.items():
            if value is None:
                continue
            expected = props[key].get("type")
            if expected == "string" and not isinstance(value, str):
                return f"{name}: argument {key!r} must be a string"
            if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
                return f"{name}: argument {key!r} must be an integer"
            if expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                return f"{name}: argument {key!r} must be a number"
            if expected == "boolean" and not isinstance(value, bool):
                return f"{name}: argument {key!r} must be a boolean"
            if expected == "array" and not isinstance(value, list):
                return f"{name}: argument {key!r} must be an array"
        return None

    def parallel_search(self, queries: Any) -> dict[str, Any]:
        """Delegate independent subtasks to read-only subagents (in parallel)."""
        if self.subagent_executor is None:
            return self._err("subagents are not available")
        if not isinstance(queries, list) or not all(isinstance(q, str) and q.strip() for q in queries):
            return self._err("queries must be a list of non-empty strings")
        if len(queries) > 16:
            return self._err("too many subagent queries (max 16)")
        results = self.subagent_executor(queries)
        return self._ok(results=results, count=len(results))

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        fn = self._executors.get(name)
        if fn is None:
            return self._err(f"unknown tool: {name}")
        error = self._validate_arguments(name, arguments)
        if error is not None:
            return self._err(error)
        try:
            result = fn(**arguments)
        except ToolError as exc:
            return self._err(str(exc))
        except Exception as exc:  # noqa: BLE001 — last line of defense
            return self._err(f"{name} failed: {exc}")
        if not isinstance(result, dict):
            result = {"ok": True, "output": str(result)}
        return result


def format_tool_result(result: dict[str, Any]) -> str:
    """Serialize a tool result into the string fed back to the model.

    Secret-looking content is redacted before serialization, so it neither
    reaches the model nor lands (unredacted) in the session log.
    """
    return json.dumps(redact_obj(result), ensure_ascii=False)
