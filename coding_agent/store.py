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
import os
import secrets
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .graph import render_graph

try:
    import fcntl
except ImportError:  # non-POSIX fallback: process-local calls only
    fcntl = None  # type: ignore[assignment]

ID_BYTES = 8  # -> 16 hex chars per session id

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

    Iterative (no deep recursion), skips symlinks entirely so a symlink loop
    cannot escape the snapshot root or recurse forever.

    Returns (files_copied, bytes_copied).
    """
    files = 0
    nbytes = 0
    if not src.is_dir():
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        return 0, 0

    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)

    stack: list[tuple[Path, Path]] = [(src, dest)]
    while stack and files < _MAX_SNAPSHOT_FILES and nbytes < _MAX_SNAPSHOT_BYTES:
        current_src, current_dest = stack.pop()
        try:
            with os.scandir(current_src) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            if files >= _MAX_SNAPSHOT_FILES or nbytes >= _MAX_SNAPSHOT_BYTES:
                break
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in _SKIP_SNAPSHOT:
                        continue
                    stack.append((Path(entry.path), current_dest / entry.name))
                elif entry.is_file(follow_symlinks=False):
                    try:
                        data = Path(entry.path).read_bytes()
                    except OSError:
                        continue
                    nbytes += len(data)
                    if nbytes > _MAX_SNAPSHOT_BYTES:
                        break
                    current_dest.mkdir(parents=True, exist_ok=True)
                    (current_dest / entry.name).write_bytes(data)
                    files += 1
            except OSError:
                continue
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
        """Atomically write a ref file (temp file + os.replace)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(value + "\n")
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

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
        self._require_session(sid)
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
        self._require_session(session_id)
        self._write_ref(self.refs_dir / "HEAD", session_id)

    def advance_head(self, session_id: str) -> None:
        """Point HEAD (or the current branch) at a newly created session."""
        self._require_session(session_id)
        branch = self.current_branch()
        if branch:
            self._write_ref(self.heads_dir / branch, session_id)
        else:
            self.set_head_detached(session_id)

    # -- ref resolution ------------------------------------------------------
    def _is_session_dir(self, sid: str) -> bool:
        d = self.sessions_dir / sid
        return d.is_dir() and (d / "meta.json").is_file()

    def _session_ids(self) -> list[str]:
        if not self.sessions_dir.is_dir():
            return []
        return [p.name for p in self.sessions_dir.iterdir() if self._is_session_dir(p.name)]

    def _require_session(self, sid: str) -> None:
        if not sid or not self._is_session_dir(sid):
            raise StoreError(f"session {sid!r} does not exist")

    @staticmethod
    def _validate_ref_token(ref: str) -> None:
        if not ref or ref in {".", ".."} or "/" in ref or "\\" in ref or "\x00" in ref:
            raise StoreError(f"invalid ref {ref!r}")

    def resolve_ref(self, ref: str) -> str:
        """Resolve a branch name or session id (full or unique prefix)."""
        if not isinstance(ref, str):
            raise StoreError(f"invalid ref {ref!r}")
        self._validate_ref_token(ref)
        branches = self.list_branches()
        if ref in branches:
            sid = branches[ref]
            if not sid:
                raise StoreError(f"branch {ref!r} is empty (no session yet)")
            self._require_session(sid)
            return sid
        if self._is_session_dir(ref):
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
        status: str = "ok",
        error: str = "",
    ) -> str:
        if parent is None:
            # Like git: a new commit's parent is the current HEAD.
            parent = self.resolve_head()
        elif parent:
            self._require_session(parent)
        while True:
            sid = self._new_id()
            d = self.sessions_dir / sid
            try:
                d.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue  # another process created this id first
            break
        meta = {
            "id": sid,
            "parent": parent,
            "task": task,
            "message": (message or _first_line(task)),
            "workdir": workdir,
            "model": model,
            "branch": self.current_branch(),
            "created_at": time.time(),
            "status": status,
            "error": error,
        }
        self._write_json_atomic(d / "meta.json", meta)
        return sid

    @staticmethod
    def _write_json_atomic(path: Path, obj: Any) -> None:
        """Write a JSON file atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(obj, indent=2, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def save_conversation(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        d = self.sessions_dir / session_id
        if not d.is_dir():
            raise StoreError(f"session {session_id!r} not found")
        payload = "".join(
            json.dumps(m, ensure_ascii=False) + "\n" for m in messages
        )
        fd, tmp = tempfile.mkstemp(
            prefix="conversation.jsonl.", dir=str(d)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, d / "conversation.jsonl")
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load_conversation(self, session_id: str) -> list[dict[str, Any]]:
        """Load a session log, tolerating/ignoring corrupt trailing lines.

        A partially-written final line after a crash should not make the whole
        history unusable; valid earlier messages are preserved.
        """
        p = self.sessions_dir / session_id / "conversation.jsonl"
        if not p.exists():
            return []
        msgs: list[dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # Skip corrupt lines but keep the rest of the log usable.
                    continue
                if isinstance(msg, dict):
                    msgs.append(msg)
        return msgs

    def load_meta(self, session_id: str) -> dict[str, Any]:
        p = self.sessions_dir / session_id / "meta.json"
        if not p.exists():
            raise StoreError(f"session {session_id!r} has no metadata")
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StoreError(f"session {session_id!r} has invalid metadata") from exc
        if not isinstance(meta, dict):
            raise StoreError(f"session {session_id!r} has invalid metadata")
        return meta

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
    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold an exclusive workspace lock for multi-step operations.

        On POSIX systems this also protects against concurrent processes using
        ``flock`` on ``<workspace>/.lock``.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        lock_file = (self.root / ".lock").open("a", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    def snapshot_artifacts(self, session_id: str, workdir: str) -> tuple[int, int]:
        self._require_session(session_id)
        src = Path(workdir)
        if not src.is_dir():
            return (0, 0)
        return _copytree_limited(src, self.sessions_dir / session_id / "artifacts")

    def restore_artifacts(self, session_id: str) -> bool:
        """Restore the workspace work/ directory from a session snapshot.

        Only the workspace ``work/`` directory is touched; a custom external
        workdir is never overwritten. The snapshot is first materialised next
        to ``work/`` and then swapped in, so a failed copy never leaves a
        half-populated working directory.
        """
        src = self.sessions_dir / session_id / "artifacts"
        if not src.is_dir():
            return False
        token = secrets.token_hex(4)
        staging = self.root / f".work-restore-{token}"
        previous = self.root / f".work-previous-{token}"
        try:
            _copytree_limited(src, staging)
            if self.work_dir.exists():
                os.replace(self.work_dir, previous)
            try:
                os.replace(staging, self.work_dir)
            except BaseException:
                # Roll the previous workdir back into place if the swap failed.
                if previous.exists() and not self.work_dir.exists():
                    os.replace(previous, self.work_dir)
                raise
            shutil.rmtree(previous, ignore_errors=True)
            return True
        finally:
            shutil.rmtree(staging, ignore_errors=True)
