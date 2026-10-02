"""Read-only Kubernetes operations implemented through ``kubectl``."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Optional


class KubernetesToolError(RuntimeError):
    """Base error raised while invoking kubectl."""


class KubernetesUnavailableError(KubernetesToolError):
    """Raised when kubectl cannot be found on PATH."""


class KubernetesCommandError(KubernetesToolError):
    """Raised when kubectl reports an error."""


class KubernetesTool:
    """Small, read-only wrapper around the locally installed ``kubectl`` CLI."""

    def __init__(self, namespace: str = "default", context: Optional[str] = None):
        self.namespace = namespace
        self.context = context

    @staticmethod
    def is_available() -> bool:
        """Return whether kubectl is available on PATH."""
        return shutil.which("kubectl") is not None

    def get_pods(self, label_selector: Optional[str] = None) -> list[dict[str, Any]]:
        """Return pods in the configured namespace as Kubernetes API objects."""
        args = ["get", "pods", "-o", "json"]
        if label_selector:
            args.extend(["--selector", label_selector])
        return self._get_items(args)

    def get_deployments(self) -> list[dict[str, Any]]:
        """Return deployments in the configured namespace as API objects."""
        return self._get_items(["get", "deployments", "-o", "json"])

    def get_services(self) -> list[dict[str, Any]]:
        """Return services in the configured namespace as API objects."""
        return self._get_items(["get", "services", "-o", "json"])

    def get_nodes(self) -> list[dict[str, Any]]:
        """Return cluster nodes as Kubernetes API objects."""
        return self._get_items(["get", "nodes", "-o", "json"], namespaced=False)

    def get_events(self, resource_name: str) -> list[dict[str, Any]]:
        """Return events associated with a named resource in this namespace."""
        return self._get_items(
            ["get", "events", "-o", "json", "--field-selector", f"involvedObject.name={resource_name}"]
        )

    def current_context(self) -> str:
        """Return the active kubectl context."""
        return self._run(["config", "current-context"], namespaced=False).strip()

    def describe(self, kind: str, name: str) -> str:
        """Return the human-readable description for a resource."""
        return self._run(["describe", kind, name])

    def logs(self, pod: str, container: Optional[str] = None, tail: int = 100) -> str:
        """Return the last ``tail`` lines for a pod container."""
        if tail < 0:
            raise ValueError("tail must be zero or greater")
        args = ["logs", pod, f"--tail={tail}"]
        if container:
            args.extend(["--container", container])
        return self._run(args)

    def _get_items(self, args: list[str], *, namespaced: bool = True) -> list[dict[str, Any]]:
        payload = self._run_json(args, namespaced=namespaced)
        items = payload.get("items")
        if not isinstance(items, list):
            raise KubernetesCommandError("kubectl returned JSON without an items list")
        return items

    def _run_json(self, args: list[str], *, namespaced: bool = True) -> dict[str, Any]:
        output = self._run(args, namespaced=namespaced)
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as error:
            raise KubernetesCommandError("kubectl returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise KubernetesCommandError("kubectl returned an unexpected JSON document")
        return payload

    def _run(self, args: list[str], *, namespaced: bool = True) -> str:
        if not self.is_available():
            raise KubernetesUnavailableError("kubectl is not installed or is not on PATH")
        command = ["kubectl"]
        if self.context:
            command.extend(["--context", self.context])
        if namespaced:
            command.extend(["--namespace", self.namespace])
        command.extend(args)
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as error:
            raise KubernetesUnavailableError(f"could not run kubectl: {error}") from error
        if result.returncode:
            detail = _last_error_line(result.stderr) or _last_error_line(result.stdout) or "unknown kubectl error"
            raise KubernetesCommandError(detail)
        return result.stdout


def _last_error_line(output: str) -> str:
    """Return the actionable final line from CLI error output."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


# ----------------------------------------------------------------------
# OpenAI function-calling schemas
# ----------------------------------------------------------------------
# Read-only kubectl operations as deployable function tools for the agent
# runner. Every operation is non-mutating, so the runner executes them
# without approval. Names are prefixed with ``k8s_`` to avoid collisions
# (e.g. both the Kubernetes and Docker tools expose ``logs``).
KUBERNETES_FUNCTION_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "k8s_get_pods",
        "description": (
            "List pods in the configured Kubernetes namespace. Read-only: "
            "free to call. Use it to see what is running before describing "
            "a pod or reading its logs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "label_selector": {
                    "type": ["string", "null"],
                    "description": "Optional label selector, e.g. 'app=web'.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "k8s_get_deployments",
        "description": (
            "List deployments in the configured Kubernetes namespace. "
            "Read-only: free to call."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "k8s_get_services",
        "description": (
            "List services in the configured Kubernetes namespace. "
            "Read-only: free to call."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "k8s_get_nodes",
        "description": (
            "List cluster nodes (not namespaced). Read-only: free to call."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "k8s_get_events",
        "description": (
            "List events for a named resource in the configured namespace. "
            "Read-only: free to call. Useful when a pod or deployment is "
            "unhealthy and you need to know why."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "resource_name": {
                    "type": "string",
                    "description": "Name of the pod, deployment, or service.",
                },
            },
            "required": ["resource_name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "k8s_current_context",
        "description": (
            "Return the active kubectl context. Read-only: free to call. "
            "Use it to confirm which cluster you are operating on."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "k8s_describe",
        "description": (
            "Return the human-readable description of a Kubernetes resource. "
            "Read-only: free to call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "description": "Resource kind, e.g. 'pod', 'deployment'.",
                },
                "name": {
                    "type": "string",
                    "description": "Name of the resource.",
                },
            },
            "required": ["kind", "name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "k8s_logs",
        "description": (
            "Return recent log lines for a pod container. Read-only: free "
            "to call. The first step when a pod is misbehaving."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pod": {
                    "type": "string",
                    "description": "Pod name.",
                },
                "container": {
                    "type": ["string", "null"],
                    "description": "Container name (omit for single-container pods).",
                },
                "tail": {
                    "type": "integer",
                    "description": "Number of recent lines to return.",
                    "default": 100,
                },
            },
            "required": ["pod"],
            "additionalProperties": False,
        },
    },
]
