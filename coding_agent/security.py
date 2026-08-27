"""Security helpers: command sandboxing, environment scrubbing, redaction.

These are best-effort, defense-in-depth measures. The only real security
boundary is a dedicated OS-level sandbox (container / VM); the sandbox wrapper
here gives unprivileged isolation (no network, read-only root) when bwrap or
firejail is available, and fails closed otherwise.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Command sandboxing (bubblewrap / firejail).
# ---------------------------------------------------------------------------

def detect_sandbox_backend() -> str | None:
    """Return ``"bwrap"`` or ``"firejail"`` if available, else ``None``."""
    if shutil.which("bwrap"):
        return "bwrap"
    if shutil.which("firejail"):
        return "firejail"
    return None


def _under(path: str, parent: str) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def build_sandbox_argv(backend: str, workdir: str, command: str) -> list[str]:
    """Wrap a shell command in a network-less, read-only-root sandbox."""
    if backend == "bwrap":
        argv = [
            "bwrap",
            "--unshare-net", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--new-session", "--die-with-parent",
            "--ro-bind", "/", "/",
            "--bind", workdir, workdir,
        ]
        # A fresh /tmp would shadow the workdir if the workdir lives under /tmp.
        if not _under(workdir, "/tmp"):
            argv += ["--tmpfs", "/tmp"]
        argv += [
            "--proc", "/proc",
            "--dev-bind", "/dev/null", "/dev/null",
            "--dev-bind", "/dev/zero", "/dev/zero",
            "--dev-bind", "/dev/random", "/dev/random",
            "--dev-bind", "/dev/urandom", "/dev/urandom",
            "--",
            "/bin/sh", "-c", command,
        ]
        return argv

    if backend == "firejail":
        return [
            "firejail", "--quiet",
            "--net=none",
            "--noprofile",
            "--private-tmp",
            "--read-only=/",
            f"--read-write={workdir}",
            "--",
            "/bin/sh", "-c", command,
        ]

    raise ValueError(f"unknown sandbox backend: {backend!r}")


# ---------------------------------------------------------------------------
# Environment scrubbing: only a curated allowlist is passed to child commands,
# so API keys/tokens never leak via `env`.
# ---------------------------------------------------------------------------

_ENV_ALLOWLIST = {
    "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "LC_NUMERIC",
    "LC_TIME", "TERM", "USER", "LOGNAME", "SHELL", "TMPDIR", "TZ",
    "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    "PYTHONPATH", "NODE_PATH", "CC", "CXX", "CPATH", "LD_LIBRARY_PATH",
    "PWD", "SHLVL", "CI", "NODE_ENV", "MAKEFLAGS", "COLUMNS", "LINES",
}


def command_env(workdir: str, sandbox: bool = False, extra_allow: str = "") -> dict[str, str]:
    """Build the (scrubbed) environment passed to a child command."""
    allow = set(_ENV_ALLOWLIST)
    for name in (extra_allow or "").split(","):
        name = name.strip()
        if name:
            allow.add(name)

    env = {k: v for k, v in os.environ.items() if k in allow}
    env["PWD"] = workdir
    if sandbox:
        # Inside a read-only root, redirect $HOME into the writable workdir.
        env["HOME"] = workdir
        env["TMPDIR"] = "/tmp"
    return env


# ---------------------------------------------------------------------------
# Secret redaction (conservative: avoids mangling ordinary source code).
# ---------------------------------------------------------------------------

_PEM_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
# Assignment-style secret keys whose *quoted literal* value is redacted.
_SECRET_KEY_RE = re.compile(
    r"(?i)\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"aws_secret_access_key|private[_-]?key)\s*[:=]\s*(['\"])[^'\"]{4,}\2"
)

_REDACTORS: list[tuple[re.Pattern[str], Any]] = [
    (_PEM_RE, "[REDACTED PRIVATE KEY]"),
    (re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{12,}\b"), "[REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*"), "Bearer [REDACTED]"),
    (_SECRET_KEY_RE, lambda m: m.group(1) + "=[REDACTED]"),
]


def redact(text: str) -> str:
    """Redact obvious secrets from a string."""
    for pattern, replacement in _REDACTORS:
        text = pattern.sub(replacement, text)
    return text


def redact_obj(obj: Any) -> Any:
    """Recursively redact every string in a (JSON-serializable) value."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    return obj
