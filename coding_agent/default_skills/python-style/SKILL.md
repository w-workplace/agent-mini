---
name: python-style
description: Follow idiomatic, typed, well-formatted Python conventions.
keywords: python, pep8, type hints, lint, format, black, typing, pythonic
---
Python conventions:
- Follow PEP 8 and the project's formatter (black/ruff) if configured.
- Add type hints to public functions and methods (use `from __future__ import
  annotations` where helpful).
- Prefer `pathlib.Path`, `dataclasses`, and standard-library tools over ad-hoc
  helpers; keep functions small and single-purpose.
- Use `if __name__ == "__main__":` guards, docstrings for public APIs, and
  explicit exceptions (avoid bare `except`).
- After editing, run the linter/type checker (e.g. `ruff check`, `mypy`) if the
  project has one configured.
