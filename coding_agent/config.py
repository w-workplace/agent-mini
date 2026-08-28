"""Configuration loading for the coding agent.

Precedence (later wins):

1. built-in defaults
2. project config file ``.coding-agent.json`` (up the directory tree)
3. workspace config file ``<workspace>/config.json``
4. environment variables (``LLM_*`` with ``OPENAI_*`` fallbacks)
5. explicit CLI arguments

Credentials are never stored in the repository: provide the API key through an
environment variable or the untracked workspace ``config.json`` (``.gitignore``
already covers ``config.json`` / ``.coding-agent.json``).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable

DEFAULTS: dict[str, Any] = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "",
    "model": "gpt-4o-mini",
    "max_iterations": 40,
    "max_tokens": 0,  # 0 => omit from request, use the model's default
    "temperature": 0.2,
    "timeout": 120.0,
    "request_retries": 3,
    "context_limit_tokens": 64000,
    "workdir": "",  # "" => use <workspace>/work (the store's working directory)
    "workspace": "~/.coding-agent",
    "allow_outside_workdir": False,
    "allow_dangerous_commands": False,
    "command_timeout": 120.0,
    "compact": False,  # summarize oldest turns instead of dropping them
    "sandbox": False,  # run commands in a network-less, read-only-root sandbox
    "env_allow": "",   # extra (comma-separated) env vars to pass to commands
    "subagents": True,  # allow delegating to read-only subagents (parallel_search)
    "subagent_parallel": 4,  # max concurrent subagents
    "stream": True,  # stream assistant text and command output live
    "skills": True,  # auto-load keyword-matched agent skills
    "max_skills": 3,  # max skills to inject per session
    "verbose": False,
    "quiet": False,  # suppress progress output
    "system_prompt": "",
}

# Environment variable names, checked in order; the first one set wins.
_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "base_url": ("LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"),
    "api_key": ("LLM_API_KEY", "OPENAI_API_KEY"),
    "model": ("LLM_MODEL", "OPENAI_MODEL"),
    "env_allow": ("LLM_ENV_ALLOW",),
}

# Numeric environment variables (``LLM_<UPPER_NAME>``).
_ENV_CASTERS: dict[str, Callable[[str], Any]] = {
    "max_iterations": int,
    "max_tokens": int,
    "temperature": float,
    "timeout": float,
    "request_retries": int,
    "context_limit_tokens": int,
    "command_timeout": float,
    "subagent_parallel": int,
    "max_skills": int,
}


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


_BOOLEAN_CASTERS: dict[str, Callable[[str], bool]] = {
    "allow_outside_workdir": _parse_bool,
    "allow_dangerous_commands": _parse_bool,
    "compact": _parse_bool,
    "sandbox": _parse_bool,
    "subagents": _parse_bool,
    "stream": _parse_bool,
    "skills": _parse_bool,
    "verbose": _parse_bool,
    "quiet": _parse_bool,
}


@dataclass
class Config:
    base_url: str = DEFAULTS["base_url"]
    api_key: str = ""
    model: str = DEFAULTS["model"]
    max_iterations: int = DEFAULTS["max_iterations"]
    max_tokens: int = DEFAULTS["max_tokens"]
    temperature: float = DEFAULTS["temperature"]
    timeout: float = DEFAULTS["timeout"]
    request_retries: int = DEFAULTS["request_retries"]
    context_limit_tokens: int = DEFAULTS["context_limit_tokens"]
    workdir: str = DEFAULTS["workdir"]
    workspace: str = DEFAULTS["workspace"]
    allow_outside_workdir: bool = False
    allow_dangerous_commands: bool = False
    command_timeout: float = DEFAULTS["command_timeout"]
    compact: bool = False
    sandbox: bool = False
    env_allow: str = ""
    subagents: bool = True
    subagent_parallel: int = 4
    stream: bool = True
    skills: bool = True
    max_skills: int = 3
    verbose: bool = False
    quiet: bool = False
    system_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **kwargs: Any) -> "Config":
        """Return a copy with the given non-``None`` values applied."""
        data = self.to_dict()
        valid = {f.name for f in fields(self)}
        for key, value in kwargs.items():
            if value is not None and key in valid:
                data[key] = value
        return Config(**data)


def _project_config_path(start: Path | None = None) -> Path | None:
    d = Path(start or os.getcwd()).resolve()
    if d.is_file():
        d = d.parent
    for p in (d, *d.parents):
        candidate = p / ".coding-agent.json"
        if candidate.is_file():
            return candidate
    return None


def _read_json_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return data


def _apply_env(cfg: dict[str, Any]) -> None:
    for key, names in _ENV_KEYS.items():
        for name in names:
            if os.environ.get(name):
                cfg[key] = os.environ[name]
                break
    for key, caster in _ENV_CASTERS.items():
        name = "LLM_" + key.upper()
        if os.environ.get(name):
            try:
                cfg[key] = caster(os.environ[name])
            except ValueError:
                print(
                    f"warning: ignoring invalid {name}={os.environ[name]!r}",
                    file=sys.stderr,
                )
    for key, caster in _BOOLEAN_CASTERS.items():
        name = "LLM_" + key.upper()
        if os.environ.get(name):
            cfg[key] = caster(os.environ[name])


def load_config(cli_overrides: dict[str, Any] | None = None) -> Config:
    """Load configuration from defaults, files, environment, then CLI args."""
    cli: dict[str, Any] = dict(cli_overrides or {})
    cfg: dict[str, Any] = dict(DEFAULTS)

    # 1. project config
    proj_data: dict[str, Any] = {}
    proj_path = _project_config_path()
    if proj_path is not None:
        try:
            proj_data = _read_json_file(proj_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"warning: could not read config {proj_path}: {exc}", file=sys.stderr)
    cfg.update({k: v for k, v in proj_data.items() if k in DEFAULTS})

    # 2. resolve the workspace (cli > env > project config > default)
    workspace = (
        cli.get("workspace")
        or os.environ.get("LLM_WORKSPACE")
        or proj_data.get("workspace")
        or DEFAULTS["workspace"]
    )
    workspace = str(Path(str(workspace)).expanduser())
    cfg["workspace"] = workspace

    # 3. workspace config file
    ws_path = Path(workspace) / "config.json"
    if ws_path.is_file():
        try:
            ws_data = _read_json_file(ws_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"warning: could not read config {ws_path}: {exc}", file=sys.stderr)
        else:
            cfg.update({k: v for k, v in ws_data.items() if k in DEFAULTS})

    # 4. environment
    _apply_env(cfg)

    # 5. CLI overrides (last)
    cfg.update({k: v for k, v in cli.items() if v is not None})

    valid = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in cfg.items() if k in valid})
