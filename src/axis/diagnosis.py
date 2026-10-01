"""Read-only target resolution and rule-based infrastructure diagnosis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from axis.tools.docker import DockerTool, DockerToolError, DockerUnavailableError
from axis.tools.kubernetes import KubernetesTool, KubernetesToolError, KubernetesUnavailableError


class TargetType(str, Enum):
    DOCKER_ENGINE = "Docker engine"
    DOCKER_CONTAINER = "Docker container"
    KUBERNETES_CLUSTER = "Kubernetes cluster"
    KUBERNETES_RESOURCE = "Kubernetes resource"


@dataclass
class ResolvedTarget:
    target_type: TargetType
    name: str
    resource: dict[str, Any] | None = None


@dataclass
class Resolution:
    targets: list[ResolvedTarget] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    docker_problem: str | None = None
    kubernetes_problem: str | None = None


@dataclass
class DiagnosisReport:
    target: ResolvedTarget
    status: str
    findings: list[str]
    events: list[str]
    actions: list[str]


def resolve_target(target: str, docker: DockerTool, kubernetes: KubernetesTool) -> Resolution:
    """Resolve a name from observed Docker and Kubernetes state."""
    name = target.strip()
    if name.lower() == "docker":
        return Resolution([ResolvedTarget(TargetType.DOCKER_ENGINE, "Docker")])
    if name.lower() in {"kubernetes", "k8s"}:
        return Resolution([ResolvedTarget(TargetType.KUBERNETES_CLUSTER, "Kubernetes")])
    result = Resolution()
    try:
        containers = docker.list_containers(all=True)
        match = next((item for item in containers if _container_matches(item, name)), None)
        if match:
            result.targets.append(ResolvedTarget(TargetType.DOCKER_CONTAINER, _container_name(match, name)))
        result.suggestions.extend(_container_suggestions(containers, name))
    except DockerToolError as error:
        result.docker_problem = _tool_problem("Docker", error)
    try:
        resources = [*kubernetes.get_pods(), *kubernetes.get_deployments(), *kubernetes.get_services()]
        for resource in resources:
            if _resource_name(resource) == name:
                result.targets.append(ResolvedTarget(TargetType.KUBERNETES_RESOURCE, name, resource))
        result.suggestions.extend(_resource_suggestions(resources, name))
    except KubernetesToolError as error:
        result.kubernetes_problem = _tool_problem("Kubernetes", error)
    return result


def diagnose(target: ResolvedTarget, docker: DockerTool, kubernetes: KubernetesTool) -> DiagnosisReport:
    """Collect facts and apply deterministic health rules."""
    if target.target_type is TargetType.DOCKER_ENGINE:
        return _docker_engine(target, docker)
    if target.target_type is TargetType.DOCKER_CONTAINER:
        return _container(target, docker)
    if target.target_type is TargetType.KUBERNETES_CLUSTER:
        return _cluster(target, kubernetes)
    return _resource(target, kubernetes)


def _docker_engine(target: ResolvedTarget, docker: DockerTool) -> DiagnosisReport:
    if not docker.is_available():
        return _unavailable(target, "Docker CLI is not installed.", "Install Docker Desktop or the Docker CLI, then rerun this command.")
    try:
        version, info, containers = docker.version_info(), docker.info(), docker.list_containers(all=True)
    except DockerToolError:
        return _unavailable(target, "Docker CLI is installed but the daemon is unavailable.", "Start Docker Desktop (or the Docker daemon) and check socket access.")
    running = sum(str(item.get("State", "")).lower() == "running" for item in containers)
    server = version.get("Server", {})
    server_version = server.get("Version") if isinstance(server, dict) else None
    findings = [f"Docker daemon reachable; {running} running container(s), {len(containers)} total."]
    if server_version:
        findings.append(f"Server version: {server_version}.")
    warnings = info.get("Warnings")
    if warnings:
        findings.append(f"Daemon warning: {warnings}")
    return DiagnosisReport(target, "warning" if warnings else "healthy", findings, [], ["Run `docker ps --all` to inspect individual containers."])


def _container(target: ResolvedTarget, docker: DockerTool) -> DiagnosisReport:
    try:
        item = docker.inspect(target.name)
    except DockerToolError:
        return _unavailable(target, "The container could no longer be inspected.", "Check that Docker is running and the container still exists.")
    state = item.get("State") if isinstance(item.get("State"), dict) else {}
    state_name, exit_code = str(state.get("Status", "unknown")), state.get("ExitCode")
    findings = [f"State: {state_name}.", f"Restart count: {item.get('RestartCount', 0)}."]
    if exit_code not in (None, 0):
        findings.append(f"Last exit code: {exit_code}.")
    if state.get("Error"):
        findings.append(f"Runtime error: {state['Error']}")
    try:
        events = _last_lines(docker.logs(target.name, tail=20))
    except DockerToolError:
        events = []
    unhealthy = state_name.lower() in {"exited", "dead", "restarting", "paused"} or exit_code not in (None, 0)
    status = "unhealthy" if unhealthy else "healthy" if state_name.lower() == "running" else "warning"
    actions = ["Review the recent container logs."]
    if unhealthy:
        actions.insert(0, "Inspect the exit code and application configuration before restarting it.")
    return DiagnosisReport(target, status, findings, events, actions)


def _cluster(target: ResolvedTarget, kubernetes: KubernetesTool) -> DiagnosisReport:
    if not kubernetes.is_available():
        return _unavailable(target, "kubectl is not installed.", "Install kubectl and configure a cluster context.")
    try:
        context, nodes = kubernetes.current_context(), kubernetes.get_nodes()
    except KubernetesToolError:
        return _unavailable(target, "kubectl cannot connect to the configured cluster.", "Check kubeconfig, current context, network access, and credentials.")
    ready = sum(_node_ready(node) for node in nodes)
    health = "healthy" if nodes and ready == len(nodes) else "warning"
    return DiagnosisReport(target, health, [f"Context: {context or 'unknown'}.", f"Nodes ready: {ready}/{len(nodes)}."], [], ["Run `kubectl get nodes` for node-level details."])


def _resource(target: ResolvedTarget, kubernetes: KubernetesTool) -> DiagnosisReport:
    resource = target.resource or {}
    kind = str(resource.get("kind", "resource"))
    status, findings = _resource_status(resource)
    findings.insert(0, f"Kind: {kind}.")
    try:
        events = _event_messages(kubernetes.get_events(target.name))
    except KubernetesToolError:
        events = []
    if kind.lower() == "pod":
        try:
            log_lines = _last_lines(kubernetes.logs(target.name, tail=20))
            events.extend(f"Log: {line}" for line in log_lines)
        except KubernetesToolError:
            pass
    if any("back-off" in event.lower() or "failed" in event.lower() for event in events):
        status = "unhealthy"
    actions = [f"Run `kubectl describe {kind.lower()} {target.name}` for full resource details."]
    if status != "healthy":
        actions.insert(0, "Review recent events and container logs for the underlying failure.")
    return DiagnosisReport(target, status, findings, events, actions)


def _resource_status(resource: dict[str, Any]) -> tuple[str, list[str]]:
    kind, state = str(resource.get("kind", "")).lower(), resource.get("status")
    status = state if isinstance(state, dict) else {}
    if kind == "pod":
        phase = str(status.get("phase", "Unknown"))
        containers = status.get("containerStatuses") if isinstance(status.get("containerStatuses"), list) else []
        restarts = sum(int(item.get("restartCount", 0)) for item in containers if isinstance(item, dict))
        level = "healthy" if phase == "Running" else "unhealthy" if phase in {"Failed", "Unknown"} else "warning"
        return level, [f"Phase: {phase}.", f"Container restarts: {restarts}."]
    if kind == "deployment":
        desired = int((resource.get("spec") or {}).get("replicas", 1))
        available = int(status.get("availableReplicas", 0))
        return ("healthy" if available >= desired else "warning"), [f"Available replicas: {available}/{desired}."]
    if kind == "service":
        return "healthy", [f"Cluster IP: {(resource.get('spec') or {}).get('clusterIP', 'none')}."]
    return "warning", ["No health rule is available for this resource kind."]


def _node_ready(node: dict[str, Any]) -> bool:
    return any(item.get("type") == "Ready" and item.get("status") == "True" for item in (node.get("status") or {}).get("conditions", []) if isinstance(item, dict))


def _container_matches(container: dict[str, Any], target: str) -> bool:
    container_id = str(container.get("ID", ""))
    return target == _container_name(container, "") or (
        bool(container_id) and (container_id.startswith(target) or target.startswith(container_id))
    )


def _container_name(container: dict[str, Any], fallback: str) -> str:
    return str(container.get("Name") or container.get("Names") or fallback).lstrip("/")


def _resource_name(resource: dict[str, Any]) -> str:
    metadata = resource.get("metadata")
    return str(metadata.get("name", "")) if isinstance(metadata, dict) else ""


def _container_suggestions(items: list[dict[str, Any]], target: str) -> list[str]:
    return [f"container/{_container_name(item, '')}" for item in items if target.lower() in _container_name(item, "").lower()][:3]


def _resource_suggestions(items: list[dict[str, Any]], target: str) -> list[str]:
    return [f"{str(item.get('kind', 'resource')).lower()}/{_resource_name(item)}" for item in items if target.lower() in _resource_name(item).lower()][:3]


def _event_messages(events: list[dict[str, Any]]) -> list[str]:
    ordered = sorted(events, key=lambda item: str(item.get("lastTimestamp") or item.get("eventTime") or ""), reverse=True)
    return [str(item.get("message") or item.get("reason") or "Kubernetes event") for item in ordered[:5]]


def _last_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.strip()][-5:]


def _tool_problem(tool: str, error: Exception) -> str:
    if isinstance(error, (DockerUnavailableError, KubernetesUnavailableError)):
        return f"{tool} is unavailable. Check that its CLI is installed and its service or cluster is reachable."
    return f"{tool} could not be queried. Check its current connection and permissions."


def _unavailable(target: ResolvedTarget, finding: str, action: str) -> DiagnosisReport:
    return DiagnosisReport(target, "unavailable", [finding], [], [action])
