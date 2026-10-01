"""Shared workspace confinement for tools (fail-closed)."""

from __future__ import annotations

from pathlib import Path


class WorkspaceError(Exception):
    """Base error for workspace confinement failures."""


class PathOutsideWorkspaceError(WorkspaceError):
    """Raised when a path escapes the workspace root."""


def resolve_in_workspace(workspace_root: Path, path: str | Path) -> Path:
    """Resolve *path* inside *workspace_root* or raise.

    Rejects ``..`` traversal, absolute paths escaping the root, and
    symlinks (including symlink components) pointing outside the root.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        raise PathOutsideWorkspaceError(
            f"path escapes workspace root {workspace_root}: {path}"
        )
    # Walk the resolved path's parents: any symlink component that lands
    # outside the workspace is rejected (defense in depth; resolve() already
    # collapses most of these, but an inner symlink to /etc must not pass).
    current = resolved
    while current != current.parent:
        if current.is_symlink():
            target = current.resolve()
            try:
                target.relative_to(workspace_root)
            except ValueError:
                raise PathOutsideWorkspaceError(
                    f"symlink points outside workspace root: {path}"
                )
        if current == workspace_root:
            break
        current = current.parent
    return resolved
