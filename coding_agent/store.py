"""Git-like workspace/session store.

Every agent run is recorded as an immutable *session* (analogous to a git
commit). Sessions form a tree via a ``parent`` pointer; *branches* are named
pointers to sessions, and ``HEAD`` points at the current session or branch.
This gives the user add / delete / rollback sessions and create / delete
branches, with git-like semantics.

Layout (the "专属文件夹"):

    <workspace>/
        config.json              optional workspace config (untracked, may hold api_key)
        work/                    the agent's working directory (stable path)
        sessions/<id>/
            meta.json            {id, parent, task, message, workdir, model, created_at, ...}
            conversation.jsonl   the session log (one JSON message per line)
            artifacts/           snapshot of the working directory when sealed
        refs/
            heads/<branch>       -> session id (or empty when unborn)
            HEAD                 -> "refs/heads/<branch>" or a bare session id
"""

from __future__ import annotations

import json
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

from .graph import render_graph

ID_BYTES = 4  # -> 8 hex chars per session id

# Directories/files skipped when snapshotting artifacts.
_SKIP_SNAPSHOT = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".tox", ".mypy_cache", ".pytest_cache", ".eggs", ".idea", "__pycache__",
}
_MAX_SNAPSHOT_FILES = 5000
_MAX_SNAPSHOT_BYTES = 100 * 1024 * 1024  # 100 MiB


class StoreError(Exception):
    """A session-store operation failure (bad ref, unsafe delete, ...)."""


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()[:120]
    return ""


def _copytree_limited(src: Path, dest: Path) -> tuple[int, int]:
    """Copy a directory tree, skipping noise dirs and enforcing limits.

    Returns (files_copied, bytes_copied).
    """
    files = 0
    nbytes = 0

    def copy_dir(s: Path, d: Path) -> None:
        nonlocal files, nbytes
        if not s.is_dir():
            return
        for entry in sorted(s.iterdir()):
            if files >= _MAX_SNAPSHOT_FILES or nbytes >= _MAX_SNAPSHOT_BYTES:
                return
            if entry.is_dir():
                if entry.name in _SKIP_SNAPSHOT:
                    continue
                copy_dir(entry, d / entry.name)
            elif entry.is_file():
                try:
                    data = entry.read_bytes()
                except OSError:
                    continue
                nbytes += len(data)
                if nbytes > _MAX_SNAPSHOT_BYTES:
                    return
                d.mkdir(parents=True, exist_ok=True)
                (d / entry.name).write_bytes(data)
                files += 1

    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    copy_dir(src, dest)
    return files, nbytes


class SessionStore:
    def __init__(self, workspace: str):
        self.root = Path(workspace).expanduser().resolve()
        self.sessions_dir = self.root / "sessions"
        self.refs_dir = self.root / "refs"
        self.heads_dir = self.refs_dir / "heads"
        self.work_dir = self.root / "work"
        self._ensure_layout()

    # -- layout --------------------------------------------------------------
    def _ensure_layout(self) -> None:
        for d in (self.sessions_dir, self.heads_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)
        head = self.refs_dir / "HEAD"
        if not head.exists():
            self._write_ref(head, "refs/heads/main")
        main = self.heads_dir / "main"
        if not main.exists():
            self._write_ref(main, "")

    # -- refs ----------------------------------------------------------------
    @staticmethod
    def _write_ref(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n", encoding="utf-8")

    @staticmethod
    def _read_ref(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def current_branch(self) -> str | None:
        head = self._read_ref(self.refs_dir / "HEAD")
        if head.startswith("refs/heads/"):
            return head[len("refs/heads/"):]
        return None

    def resolve_head(self) -> str | None:
        head = self._read_ref(self.refs_dir / "HEAD")
        if head.startswith("refs/heads/"):
            branch = head[len("refs/heads/"):]
            sid = self._read_ref(self.heads_dir / branch)
            return sid or None
        return head or None

    def list_branches(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.heads_dir.is_dir():
            for p in sorted(self.heads_dir.iterdir()):
                if p.is_file():
                    out[p.name] = self._read_ref(p)
        return out

    def create_branch(self, name: str, session_id: str | None = None) -> None:
        if not name or "/" in name or name.startswith(".") or name in {"HEAD"}:
            raise StoreError(f"invalid branch name: {name!r}")
        if (self.heads_dir / name).exists():
            raise StoreError(f"branch {name!r} already exists")
        sid = session_id or self.resolve_head()
        if not sid:
            raise StoreError("cannot create a branch: no session yet")
        self._write_ref(self.heads_dir / name, sid)

    def delete_branch(self, name: str) -> None:
        if name == self.current_branch():
            raise StoreError(f"cannot delete the current branch {name!r}")
        p = self.heads_dir / name
        if not p.exists():
            raise StoreError(f"branch {name!r} does not exist")
        p.unlink()

    def set_head_branch(self, branch: str) -> None:
        if not (self.heads_dir / branch).exists():
            raise StoreError(f"branch {branch!r} does not exist")
        self._write_ref(self.refs_dir / "HEAD", f"refs/heads/{branch}")

    def set_head_detached(self, session_id: str) -> None:
        self._write_ref(self.refs_dir / "HEAD", session_id)

    def advance_head(self, session_id: str) -> None:
        """Point HEAD (or the current branch) at a newly created session."""
        branch = self.current_branch()
        if branch:
            self._write_ref(self.heads_dir / branch, session_id)
        else:
            self.set_head_detached(session_id)

    # -- ref resolution ------------------------------------------------------
    def _session_ids(self) -> list[str]:
        if not self.sessions_dir.is_dir():
            return []
        return [p.name for p in self.sessions_dir.iterdir() if p.is_dir()]

    def resolve_ref(self, ref: str) -> str:
        """Resolve a branch name or session id (full or unique prefix)."""
        branches = self.list_branches()
        if ref in branches:
            sid = branches[ref]
            if not sid:
                raise StoreError(f"branch {ref!r} is empty (no session yet)")
            return sid
        if (self.sessions_dir / ref).is_dir():
            return ref
        matches = [sid for sid in self._session_ids() if sid.startswith(ref)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise StoreError(f"ambiguous ref {ref!r}: matches {', '.join(sorted(matches))}")
        raise StoreError(f"no session or branch matches {ref!r}")

    def checkout(self, ref: str, restore: bool = False) -> str:
        """Move HEAD to a branch or session; optionally restore artifacts."""
        sid = self.resolve_ref(ref)
        if ref in self.list_branches():
            self.set_head_branch(ref)
        else:
            self.set_head_detached(sid)
        if restore:
            self.restore_artifacts(sid)
        return sid

    # -- sessions ------------------------------------------------------------
    def _new_id(self) -> str:
        while True:
            sid = secrets.token_hex(ID_BYTES)
            if not (self.sessions_dir / sid).exists():
                return sid

    def create_session(
        self,
        task: str,
        parent: str | None = None,
        message: str | None = None,
        workdir: str | None = None,
        model: str = "",
    ) -> str:
        if parent is None:
            # Like git: a new commit's parent is the current HEAD.
            parent = self.resolve_head()
        sid = self._new_id()
        d = self.sessions_dir / sid
        d.mkdir(parents=True, exist_ok=False)
        meta = {
            "id": sid,
            "parent": parent,
            "task": task,
            "message": (message or _first_line(task)),
            "workdir": workdir,
            "model": model,
            "branch": self.current_branch(),
            "created_at": time.time(),
        }
        (d / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return sid

    def save_conversation(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        d = self.sessions_dir / session_id
        if not d.is_dir():
            raise StoreError(f"session {session_id!r} not found")
        with (d / "conversation.jsonl").open("w", encoding="utf-8") as f:
            for m in messages:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def load_conversation(self, session_id: str) -> list[dict[str, Any]]:
        p = self.sessions_dir / session_id / "conversation.jsonl"
        if not p.exists():
            return []
        msgs: list[dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    msgs.append(json.loads(line))
        return msgs

    def load_meta(self, session_id: str) -> dict[str, Any]:
        p = self.sessions_dir / session_id / "meta.json"
        if not p.exists():
            raise StoreError(f"session {session_id!r} has no metadata")
        return json.loads(p.read_text(encoding="utf-8"))

    def list_sessions(self) -> list[dict[str, Any]]:
        metas: list[dict[str, Any]] = []
        for sid in self._session_ids():
            try:
                metas.append(self.load_meta(sid))
            except (StoreError, OSError, json.JSONDecodeError):
                continue
        metas.sort(key=lambda m: m.get("created_at", 0))
        return metas

    def ancestors(self, session_id: str) -> set[str]:
        seen: set[str] = set()
        cur = session_id
        while cur and cur not in seen:
            seen.add(cur)
            try:
                cur = self.load_meta(cur).get("parent") or ""
            except StoreError:
                break
        return seen

    def _reachable(self, all_branches: bool = False) -> list[str]:
        """Ids reachable from HEAD (or all tips), newest first."""
        metas = {m["id"]: m for m in self.list_sessions()}
        tips: set[str] = set()
        if all_branches:
            tips = {sid for sid in self.list_branches().values() if sid}
        head = self.resolve_head()
        if head:
            tips.add(head)
        reachable: set[str] = set()
        for tip in tips:
            reachable |= self.ancestors(tip)
        ordered = [sid for sid in reachable if sid in metas]
        ordered.sort(key=lambda sid: metas[sid].get("created_at", 0), reverse=True)
        return ordered

    def log(self, all_branches: bool = False) -> list[dict[str, Any]]:
        """Sessions reachable from HEAD (or all tips), newest first."""
        metas = {m["id"]: m for m in self.list_sessions()}
        return [metas[sid] for sid in self._reachable(all_branches)]

    def graph(self, all_branches: bool = False) -> list[str]:
        """Return ``git log --graph``-style lines for the session DAG."""
        metas = {m["id"]: m for m in self.list_sessions()}
        ordered = self._reachable(all_branches)

        parents = {sid: (metas[sid].get("parent") or None) for sid in ordered}

        head = self.resolve_head()
        refs: dict[str, list[str]] = {}
        for name, tip in self.list_branches().items():
            if tip:
                refs.setdefault(tip, []).append(name)
        if head:
            refs.setdefault(head, []).append("HEAD")

        labels: dict[str, str] = {}
        for sid in ordered:
            m = metas[sid]
            parts = [m.get("message", "") or sid]
            names = refs.get(sid)
            if names:
                parts.append("(" + ", ".join(names) + ")")
            labels[sid] = " ".join(parts)

        return render_graph(ordered, parents, labels)

    def delete_session(self, ref: str) -> None:
        sid = self.resolve_ref(ref)
        if self.resolve_head() == sid:
            raise StoreError("cannot delete the current session; switch away first")
        for name, tip in self.list_branches().items():
            if tip == sid:
                raise StoreError(f"cannot delete: branch {name!r} points at it")
        for m in self.list_sessions():
            if m.get("parent") == sid:
                raise StoreError(
                    f"cannot delete: session {m['id']} descends from it; delete it first"
                )
        shutil.rmtree(self.sessions_dir / sid, ignore_errors=True)

    # -- artifacts -----------------------------------------------------------
    def snapshot_artifacts(self, session_id: str, workdir: str) -> tuple[int, int]:
        src = Path(workdir)
        if not src.is_dir():
            return (0, 0)
        return _copytree_limited(src, self.sessions_dir / session_id / "artifacts")

    def restore_artifacts(self, session_id: str) -> bool:
        """Restore the workspace work/ directory from a session snapshot.

        Only the workspace ``work/`` directory is touched; a custom external
        workdir is never overwritten.
        """
        src = self.sessions_dir / session_id / "artifacts"
        if not src.is_dir():
            return False
        _copytree_limited(src, self.work_dir)
        return True
