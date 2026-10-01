"""Registry mapping OpenAI function names to local executors.

The agent runner asks the model which tool to call; this registry resolves the
returned function name to the local code that executes it. New tool families
(kubernetes, docker, shell, ...) plug in with a single ``register`` call —
the runner loop itself never changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from axis.safety.permissions import PermissionGate
from axis.tools import OPENAI_FUNCTION_SCHEMAS as FILESYSTEM_SCHEMAS
from axis.tools.filesystem import FilesystemTool


class UnknownToolError(KeyError):
    """The model asked for a tool name the registry does not know."""


@dataclass
class RegisteredTool:
    """One callable tool: its OpenAI schema, local executor, and risk."""

    name: str
    schema: Dict[str, Any]
    execute: Callable[[Dict[str, Any]], Dict[str, Any]]
    mutating: bool = False
    description: str = ""


class ToolRegistry:
    """Name → executor mapping with the deployable OpenAI schemas."""

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(name) from None

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        """OpenAI Responses API ``tools`` entries, in registration order."""
        return [tool.schema for tool in self._tools.values()]


def default_registry(
    workspace_root: Union[str, Path] = ".",
    *,
    gate: PermissionGate | None = None,
) -> ToolRegistry:
    """Registry with the filesystem tools.

    Approvals for mutating tools are performed centrally by the agent runner
    (see :mod:`axis.agents.runner`), so the filesystem tool is built with a
    non-interactive gate here — the runner's own gate is the single policy
    point. Using :class:`FilesystemTool` directly still goes through its own
    interactive gate.
    """
    _ = gate  # reserved: future tool families may need their own gate wiring
    registry = ToolRegistry()
    tool = FilesystemTool(
        workspace_root=workspace_root,
        gate=PermissionGate(require_approval=False),
    )
    by_name = {schema["name"]: schema for schema in FILESYSTEM_SCHEMAS}
    registry.register(
        RegisteredTool(
            name="search_directory",
            schema=by_name["search_directory"],
            execute=lambda args: tool.search_directory(**args),
            mutating=False,
            description="Read-only directory search (glob + content grep).",
        )
    )
    registry.register(
        RegisteredTool(
            name="create_file",
            schema=by_name["create_file"],
            execute=lambda args: tool.create_file(**args),
            mutating=True,
            description="Create a file (requires approval).",
        )
    )
    return registry
