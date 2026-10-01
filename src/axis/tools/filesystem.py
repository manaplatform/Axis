"""Filesystem tools: directory search and file creation.

These are the local executors behind the two OpenAI function-calling tools
that Axis deploys:

- ``search_directory`` — read-only search over a workspace directory
  (glob patterns and/or content grep). ``RiskLevel.READ``: free, no approval.
- ``create_file`` — create a new file with the given content.
  Mutating: goes through :class:`PermissionGate` (default ``MEDIUM`` risk).

Safety rules (fail-closed):

- Every path is resolved against ``workspace_root`` and must stay inside it.
  Path traversal (``..``), absolute paths escaping the root, and symlinks
  pointing outside are rejected.
- ``create_file`` never overwrites an existing file unless ``overwrite=True``
  is passed explicitly *and* the permission gate approves it.
- Search never follows symlinks outside the root and skips hidden
  directories (``.git`` etc.) unless asked.

The :data:`OPENAI_FUNCTION_SCHEMAS` list contains the Responses API
``tools`` entries (``type: "function"``) so these tools can be deployed
as-is::

    from axis.tools.filesystem import OPENAI_FUNCTION_SCHEMAS
    response = client.responses.create(
        model="gpt-5.x",
        tools=OPENAI_FUNCTION_SCHEMAS,
        input="...",
    )
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from axis.safety.permissions import PermissionGate, RiskLevel

# Directories that are never descended into during a search unless the
# caller explicitly opts in with ``include_hidden=True``.
_SKIPPED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }
)

_DEFAULT_MAX_RESULTS = 100
_DEFAULT_MAX_SNIPPET_CHARS = 300


class FilesystemError(Exception):
    """Base error for filesystem tool failures."""


class PathOutsideWorkspaceError(FilesystemError):
    """Raised when a path escapes the workspace root (fail-closed)."""


class FileExistsError(FilesystemError):
    """Raised when create_file would overwrite without explicit consent."""


@dataclass
class FilesystemTool:
    """Directory search + file creation confined to a workspace root."""

    workspace_root: Path | str = field(default_factory=lambda: Path.cwd())
    gate: PermissionGate = field(default_factory=PermissionGate)

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()
        if not self.workspace_root.is_dir():
            raise FilesystemError(
                f"workspace root does not exist or is not a directory: {self.workspace_root}"
            )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _resolve(self, path: str | Path) -> Path:
        """Resolve *path* inside the workspace root or raise."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        # resolve() with strict=False: normalises `..` without requiring existence
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            raise PathOutsideWorkspaceError(
                f"path escapes workspace root {self.workspace_root}: {path}"
            )
        # A symlink inside the root pointing outside must also be rejected.
        if resolved.is_symlink():
            target = resolved.resolve()
            try:
                target.relative_to(self.workspace_root)
            except ValueError:
                raise PathOutsideWorkspaceError(
                    f"symlink points outside workspace root: {path}"
                )
        return resolved

    def _iter_files(self, root: Path, include_hidden: bool):
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            if not include_hidden:
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in _SKIPPED_DIRS and not d.startswith(".")
                ]
            for name in filenames:
                if not include_hidden and name.startswith("."):
                    continue
                full = Path(dirpath) / name
                # Skip symlinks that escape the workspace.
                if full.is_symlink():
                    try:
                        full.resolve().relative_to(self.workspace_root)
                    except ValueError:
                        continue
                yield full

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------
    def search_directory(
        self,
        pattern: str = "**/*",
        content_query: Optional[str] = None,
        root: str = ".",
        *,
        max_results: int = _DEFAULT_MAX_RESULTS,
        include_hidden: bool = False,
        case_sensitive: bool = False,
    ) -> Dict[str, Any]:
        """Search files under *root* by glob *pattern* and/or content.

        Returns ``{"matches": [...], "truncated": bool}``. Each match has
        ``path`` (workspace-relative), and for content searches also
        ``line`` and ``snippet``.
        """
        search_root = self._resolve(root)
        if not search_root.is_dir():
            raise FilesystemError(f"search root is not a directory: {root}")

        if max_results < 1 or max_results > 1000:
            raise FilesystemError("max_results must be between 1 and 1000")

        flags = 0 if case_sensitive else re.IGNORECASE
        content_re = re.compile(content_query, flags) if content_query else None

        matches: List[Dict[str, Any]] = []
        truncated = False
        for full in self._iter_files(search_root, include_hidden):
            rel = full.relative_to(self.workspace_root).as_posix()
            # Match the glob against the path relative to the search root
            # as well as the workspace-relative path, so patterns like
            # ``*.yaml`` work at any depth via ``**``.
            rel_to_root = full.relative_to(search_root).as_posix()
            if not (
                fnmatch.fnmatchcase(rel_to_root, pattern)
                or fnmatch.fnmatchcase(rel, pattern)
            ):
                continue
            if content_re is None:
                matches.append({"path": rel})
            else:
                try:
                    text = full.read_text(encoding="utf-8", errors="strict")
                except (OSError, UnicodeDecodeError, ValueError):
                    continue  # binary / unreadable: skip
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if content_re.search(line):
                        snippet = line.strip()
                        if len(snippet) > _DEFAULT_MAX_SNIPPET_CHARS:
                            snippet = snippet[:_DEFAULT_MAX_SNIPPET_CHARS] + "…"
                        matches.append(
                            {"path": rel, "line": lineno, "snippet": snippet}
                        )
                        if len(matches) >= max_results:
                            break
            if len(matches) >= max_results:
                truncated = True
                break

        return {"matches": matches[:max_results], "truncated": truncated}

    def create_file(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = False,
        risk: RiskLevel = RiskLevel.MEDIUM,
    ) -> Dict[str, Any]:
        """Create *path* with *content* inside the workspace.

        Parent directories are created as needed. Fails closed if the file
        exists and ``overwrite`` is not set, and asks the permission gate
        before writing (mutating operation).
        """
        target = self._resolve(path)
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"file already exists (pass overwrite=True to replace): {path}"
            )

        details = f"create file {target.relative_to(self.workspace_root)} ({len(content)} chars)"
        if not self.gate.check("create_file", risk, details):
            return {"created": False, "path": None, "reason": "denied by approval gate"}

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "created": True,
            "path": target.relative_to(self.workspace_root).as_posix(),
            "bytes": len(content.encode("utf-8")),
        }


# ----------------------------------------------------------------------
# OpenAI function-calling schemas ("which tools to deploy")
# ----------------------------------------------------------------------
# Axis needs exactly one OpenAI built-in tool family for this: **function
# calling**. The two entries below are the deployable units — put them in
# the Responses API ``tools`` parameter (or Agents SDK agent definition).
# The Shell tool is the alternative when the model should run raw shell
# itself; File search only fits when files live in OpenAI vector stores.
OPENAI_FUNCTION_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_directory",
        "description": (
            "Search for files inside the workspace directory by glob pattern "
            "and optionally by text content (grep). Read-only: free to call. "
            "Use it to find manifests, Dockerfiles, configs, or any file "
            "before reading or editing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. '**/*.yaml', 'Dockerfile*'.",
                    "default": "**/*",
                },
                "content_query": {
                    "type": ["string", "null"],
                    "description": "Optional regex searched inside file contents.",
                },
                "root": {
                    "type": "string",
                    "description": "Workspace-relative directory to search under.",
                    "default": ".",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum matches to return (1-1000).",
                    "default": 100,
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files and skipped dirs like .git.",
                    "default": False,
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive content matching.",
                    "default": False,
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "create_file",
        "description": (
            "Create a new file inside the workspace with the given content "
            "(e.g. a Kubernetes manifest, Dockerfile, or config). Parent "
            "directories are created automatically. Fails if the file exists "
            "unless overwrite is true. Mutating: requires approval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path of the file to create.",
                },
                "content": {
                    "type": "string",
                    "description": "Full text content to write into the file.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Replace the file if it already exists.",
                    "default": False,
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
]
