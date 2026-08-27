"""Configuration loading for the coding agent.

Precedence (later wins):

1. built-in defaults
2. config file — project ``.coding-agent.json`` first, then the user-level file
   ``~/.config/coding-agent/config.json`` (or ``$XDG_CONFIG_HOME/...``)
3. environment variables (``LLM_*`` with ``OPENAI_*`` fallbacks)
4. explicit CLI arguments, applied last by the caller

Credentials are never stored in the repository: provide the API key through an
environment variable or an untracked config file (``.gitignore`` already covers
``config.json`` / ``.coding-agent.json``).
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
    "workdir": ".",
    "allow_outside_workdir": False,
    "allow_dangerous_commands": False,
    "command_timeout": 120.0,
    "verbose": False,
    "system_prompt": "",
}

# Environment variable names, checked in order; the first one set wins.
_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "base_url": ("LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"),
    "api_key": ("LLM_API_KEY", "OPENAI_API_KEY"),
    "model": ("LLM_MODEL", "OPENAI_MODEL"),
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
}

def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


_BOOLEAN_CASTERS: dict[str, Callable[[str], bool]] = {
    "allow_outside_workdir": _parse_bool,
    "allow_dangerous_commands": _parse_bool,
    "verbose": _parse_bool,
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
    allow_outside_workdir: bool = False
    allow_dangerous_commands: bool = False
    command_timeout: float = DEFAULTS["command_timeout"]
    verbose: bool = False
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


def _user_config_path() -> Path | None:
    base = os.environ.get("XDG_CONFIG_HOME")
    p = Path(base) if base else Path.home() / ".config"
    candidate = p / "coding-agent" / "config.json"
    return candidate if candidate.is_file() else None


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
    cfg: dict[str, Any] = dict(DEFAULTS)

    for path in (_user_config_path(), _project_config_path()):
        if path is None:
            continue
        try:
            data = _read_json_file(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"warning: could not read config {path}: {exc}", file=sys.stderr)
            continue
        cfg.update({k: v for k, v in data.items() if k in DEFAULTS})

    _apply_env(cfg)

    if cli_overrides:
        cfg.update({k: v for k, v in cli_overrides.items() if v is not None})

    valid = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in cfg.items() if k in valid})
