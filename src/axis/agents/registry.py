"""Registry mapping OpenAI function names to local executors.

The agent runner asks the model which tool to call; this registry resolves the
returned function name to the local code that executes it. New tool families
plug in with a single ``register`` call — the runner loop itself never changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from axis.safety.permissions import PermissionGate
from axis.tools import OPENAI_FUNCTION_SCHEMAS as FILESYSTEM_SCHEMAS
from axis.tools.docker import DOCKER_FUNCTION_SCHEMAS, DockerTool
from axis.tools.filesystem import FilesystemTool
from axis.tools.kubernetes import KUBERNETES_FUNCTION_SCHEMAS, KubernetesTool
from axis.tools.shell import OPENAI_FUNCTION_SCHEMAS as SHELL_SCHEMAS
from axis.tools.shell import ShellTool, classify_command


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
    # Optional per-call mutation check, for tools whose risk depends on the
    # arguments (e.g. the shell tool classifies each command). When present,
    # the runner treats the call as mutating if either flag says so.
    classify: Optional[Callable[[Dict[str, Any]], bool]] = None


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


def _executor(method: Callable[..., Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Adapt a tool method to the registry's ``args -> result`` shape."""

    def execute(args: Dict[str, Any]) -> Dict[str, Any]:
        return method(**args)

    return execute


def default_registry(
    workspace_root: Union[str, Path] = ".",
    *,
    gate: PermissionGate | None = None,
    namespace: str = "default",
    kube_context: Optional[str] = None,
) -> ToolRegistry:
    """Registry with the filesystem, shell, Kubernetes, and Docker tools.

    Approvals for mutating tools are performed centrally by the agent runner
    (see :mod:`axis.agents.runner`), so the filesystem and shell tools are
    built with non-interactive gates here — the runner's own gate is the
    single policy point. Using those tools directly still goes through their
    own interactive gates.

    The Kubernetes and Docker tools are fully read-only, so they all register
    as non-mutating. The shell tool classifies each command: read-only
    commands run free, mutating ones need approval, denied ones are blocked.
    ``namespace``/``kube_context`` configure the Kubernetes tool; the Docker
    tool needs no configuration.
    """
    _ = gate  # reserved: future tool families may need their own gate wiring
    registry = ToolRegistry()
    _register_filesystem(registry, workspace_root)
    _register_shell(registry, workspace_root)
    _register_kubernetes(registry, namespace=namespace, context=kube_context)
    _register_docker(registry)
    return registry


def _register_filesystem(registry: ToolRegistry, workspace_root: Union[str, Path]) -> None:
    tool = FilesystemTool(
        workspace_root=workspace_root,
        gate=PermissionGate(require_approval=False),
    )
    methods = {
        "search_directory": (tool.search_directory, False, "Read-only directory search (glob + content grep)."),
        "create_file": (tool.create_file, True, "Create a file (requires approval)."),
    }
    for schema in FILESYSTEM_SCHEMAS:
        name = schema["name"]
        method, mutating, description = methods[name]
        registry.register(
            RegisteredTool(
                name=name,
                schema=schema,
                execute=_executor(method),
                mutating=mutating,
                description=description,
            )
        )


def _register_shell(registry: ToolRegistry, workspace_root: Union[str, Path]) -> None:
    tool = ShellTool(
        workspace_root=workspace_root,
        gate=PermissionGate(require_approval=False),
    )
    for schema in SHELL_SCHEMAS:
        registry.register(
            RegisteredTool(
                name=schema["name"],
                schema=schema,
                execute=_executor(tool.run),
                mutating=False,
                description="Run a shell command locally (read-only free; mutating needs approval).",
                classify=lambda args: classify_command(args.get("command") or "") == "mutating",
            )
        )


def _register_kubernetes(
    registry: ToolRegistry, *, namespace: str, context: Optional[str]
) -> None:
    tool = KubernetesTool(namespace=namespace, context=context)
    methods = {
        "k8s_get_pods": (tool.get_pods, "List pods in the namespace."),
        "k8s_get_deployments": (tool.get_deployments, "List deployments in the namespace."),
        "k8s_get_services": (tool.get_services, "List services in the namespace."),
        "k8s_get_nodes": (tool.get_nodes, "List cluster nodes."),
        "k8s_get_events": (tool.get_events, "List events for a named resource."),
        "k8s_current_context": (tool.current_context, "Show the active kubectl context."),
        "k8s_describe": (tool.describe, "Describe a Kubernetes resource."),
        "k8s_logs": (tool.logs, "Read recent pod log lines."),
    }
    for schema in KUBERNETES_FUNCTION_SCHEMAS:
        name = schema["name"]
        method, description = methods[name]
        registry.register(
            RegisteredTool(
                name=name,
                schema=schema,
                execute=_executor(method),
                mutating=False,
                description=description,
            )
        )


def _register_docker(registry: ToolRegistry) -> None:
    tool = DockerTool()
    methods = {
        "docker_list_containers": (tool.list_containers, "List Docker containers."),
        "docker_logs": (tool.logs, "Read recent container log lines."),
        "docker_inspect": (tool.inspect, "Inspect a container."),
        "docker_version": (tool.version_info, "Show Docker client/server versions."),
        "docker_info": (tool.info, "Show a Docker daemon summary."),
    }
    for schema in DOCKER_FUNCTION_SCHEMAS:
        name = schema["name"]
        method, description = methods[name]
        registry.register(
            RegisteredTool(
                name=name,
                schema=schema,
                execute=_executor(method),
                mutating=False,
                description=description,
            )
        )
