"""`git log --graph`-style DAG rendering.

Pure function that takes a list of commit ids (newest first), a parent map and
per-commit labels, and returns formatted lines showing ``*`` for commits, ``|``
for vertical edges, and ``/`` / ``\\`` for forks and merges.
"""

from __future__ import annotations

from typing import Any


def render_graph(
    commits: list[str],
    parents: dict[str, str | None],
    labels: dict[str, str],
) -> list[str]:
    """Render a commit DAG as ``git log --graph``-style lines.

    ``commits`` must be in reverse-chronological order (newest first). Each
    commit has at most one parent (``parents[id] -> parent_id`` or ``None``).
    ``labels`` maps a commit id to the text printed to the right of the graph.
    """
    commits = list(commits)
    if not commits:
        return []

    commit_set = set(commits)

    # Tips = commits that no other commit in this set has as a parent.
    has_child: set[str] = set()
    for c in commits:
        p = parents.get(c)
        if p in commit_set:
            has_child.add(p)
    tips = [c for c in commits if c not in has_child]

    # Assign each commit a fixed lane (column) by walking each tip's ancestor
    # chain; the newest tip takes the leftmost lane.
    lane: dict[str, int] = {}
    col = 0
    for tip in tips:
        cur: str | None = tip
        while cur in commit_set and cur not in lane:
            lane[cur] = col
            cur = parents.get(cur)
        col += 1
    ncols = max(lane.values()) + 1 if lane else 0

    cols: list[str | None] = [None] * ncols
    lines: list[str] = []

    for c in commits:
        i = lane[c]
        cols[i] = c

        chars: list[str] = []
        for j in range(ncols):
            if j == i:
                chars.append("*")
            elif cols[j] is not None:
                chars.append("|")
            else:
                chars.append(" ")
        # Commit rows separate lanes with a space (git-style "| * "); the
        # trailing space is trimmed before the message.
        lines.append(" ".join(chars).rstrip() + " " + labels.get(c, c))

        p = parents.get(c)
        if p in commit_set:
            pi = lane[p]
            if pi == i:
                cols[i] = p
            else:
                # The line moves from lane i to lane pi (fork/merge): draw a
                # connector row (git-style "|/", "|\"), then continue in lane pi.
                conn: list[str] = []
                for j in range(ncols):
                    if j == i:
                        conn.append("/" if pi < i else "\\")
                    elif j == pi or cols[j] is not None:
                        conn.append("|")
                    else:
                        conn.append(" ")
                lines.append("".join(conn).rstrip())
                cols[i] = None
                cols[pi] = p
        else:
            cols[i] = None

    return lines
