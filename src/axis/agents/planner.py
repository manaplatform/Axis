"""Read-only planning for Docker and Kubernetes operations."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from axis.tools.docker import DockerTool, DockerToolError
from axis.tools.kubernetes import KubernetesTool, KubernetesToolError
from axis.core.llm import LLMClient, LLMError


class PlannerAgent:
    """Turn a natural-language goal into a safe, executable plan."""

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client

    def create_plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        docker: Optional[DockerTool] = None,
        kubernetes: Optional[KubernetesTool] = None,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Return a structured plan and lightweight, read-only observations."""
        normalized_goal = " ".join(goal.split())
        domain = self._detect_domain(normalized_goal)
        selected_namespace = self.namespace_from_goal(normalized_goal) or namespace
        collected = self._gather_context(domain, docker, kubernetes, selected_namespace)
        if context:
            collected.update(context)

        intent = self._detect_intent(normalized_goal)
        if intent == "containerize":
            plan = self._containerize_plan(normalized_goal, collected)
        elif intent == "scale":
            plan = self._scale_plan(normalized_goal, selected_namespace, collected)
        elif intent == "restart":
            plan = self._restart_plan(normalized_goal, selected_namespace, collected)
        elif intent == "docker_health":
            plan = self._docker_health_plan(normalized_goal, collected)
        else:
            plan = self._general_plan(normalized_goal, domain, collected)

        # A failed or absent model must never prevent local planning.  The local
        # result remains the safe fallback and also supplies domain metadata.
        if self.llm_client is not None and self.llm_client.configured:
            try:
                plan = {
                    "goal": normalized_goal,
                    **self.llm_client.generate_plan(normalized_goal, collected).model_dump(),
                }
                plan["engine"] = "llm"
            except LLMError as error:
                plan.setdefault("notes", []).append(f"LLM unavailable ({error}); using the local planner.")
                plan["engine"] = "local"
                plan["error"] = str(error)
        else:
            plan["engine"] = "local"
        plan["domain"] = domain
        plan["context"] = collected
        return plan

    @staticmethod
    def _detect_domain(goal: str) -> str:
        lowered = goal.lower()
        docker = any(word in lowered for word in ("docker", "container", "image", "dockerfile"))
        kubernetes = any(word in lowered for word in ("kubernetes", "k8s", "kubectl", "pod", "deployment", "namespace", "replica", "cluster"))
        if docker and kubernetes:
            return "mixed"
        if docker:
            return "docker"
        if kubernetes:
            return "kubernetes"
        return "general"

    @staticmethod
    def _detect_intent(goal: str) -> str:
        lowered = goal.lower()
        if any(phrase in lowered for phrase in ("create a docker container", "containerize", "dockerize", "create dockerfile")):
            return "containerize"
        if "scale" in lowered and ("deployment" in lowered or "replica" in lowered):
            return "scale"
        if "restart" in lowered and ("pod" in lowered or "kubernetes" in lowered or "k8s" in lowered):
            return "restart"
        if "docker" in lowered and any(word in lowered for word in ("unhealthy", "health", "why", "diagnose", "check")):
            return "docker_health"
        return "general"

    @staticmethod
    def namespace_from_goal(goal: str) -> Optional[str]:
        """Extract an explicitly named namespace from a goal, if present."""
        match = re.search(r"\bnamespace\s+([a-z0-9][a-z0-9._-]*)", goal, re.IGNORECASE)
        return match.group(1) if match else None

    def _gather_context(self, domain: str, docker: Optional[DockerTool], kubernetes: Optional[KubernetesTool], namespace: str) -> Dict[str, Any]:
        context: Dict[str, Any] = {"namespace": namespace}
        if domain in {"docker", "mixed"} and docker is not None:
            try:
                containers = docker.list_containers(all=True)
                running = sum(item.get("State") == "running" for item in containers)
                context["docker"] = f"available; {running} running, {len(containers)} total containers"
            except DockerToolError as error:
                context["docker"] = f"unavailable: {error}"
        if domain in {"kubernetes", "mixed"} and kubernetes is not None:
            try:
                current_context = kubernetes.current_context()
                deployments = kubernetes.get_deployments()
                context["kubernetes"] = f"available; context={current_context or 'unknown'}, namespace={namespace}, {len(deployments)} deployments"
            except KubernetesToolError as error:
                context["kubernetes"] = f"unavailable: {error}"
        return context

    @staticmethod
    def _containerize_plan(goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"goal": goal, "interpretation": "Package this repository as a Docker image and run it safely for local verification.", "assumptions": ["The repository contains an application with a documented start command.", "No production deployment is intended by this request."], "steps": ["Inspect the repository's runtime, start command, exposed port, and required configuration values.", "Add a minimal Dockerfile and .dockerignore using a supported base image and a non-root runtime user where practical.", "Build a locally tagged image, for example `docker build -t axis-app:local .`.", "Run the image with explicit port and configuration mappings; do not publish it or replace an existing container.", "Verify startup with `docker ps`, application health checks, and container logs."], "risk_level": "medium", "requires_approval": True, "notes": ["Building and running containers changes the local Docker state and requires approval.", context.get("docker", "Docker availability was not checked.")]}

    @staticmethod
    def _scale_plan(goal: str, namespace: str, context: Dict[str, Any]) -> Dict[str, Any]:
        deployment = PlannerAgent._resource_after(goal, "deployment") or "<deployment-name>"
        replicas = re.search(r"\bto\s+(\d+)\s+replicas?\b", goal, re.IGNORECASE)
        desired = replicas.group(1) if replicas else "<replica-count>"
        return {"goal": goal, "interpretation": f"Scale Kubernetes deployment `{deployment}` to {desired} replicas in namespace `{namespace}`.", "assumptions": [f"`{deployment}` is the intended deployment in namespace `{namespace}`.", "The cluster has capacity and any autoscaler will not immediately override the requested replica count."], "steps": [f"Confirm the current replica count, resource requests, quota, and autoscaling policy for deployment/{deployment} in `{namespace}`.", f"Request approval for `kubectl scale deployment/{deployment} --replicas={desired} --namespace {namespace}`.", "Apply the approved scale operation through the authorized deployment workflow.", f"Watch `kubectl rollout status deployment/{deployment} --namespace {namespace}` and confirm the requested ready replica count.", "Check application health and error rate; roll back to the previous replica count if the rollout degrades service."], "risk_level": "medium", "requires_approval": True, "notes": [context.get("kubernetes", "Kubernetes availability was not checked."), "This plan makes no cluster changes."]}

    @staticmethod
    def _restart_plan(goal: str, namespace: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"goal": goal, "interpretation": f"Investigate failed Kubernetes pods in namespace `{namespace}` and restart only a verified owning workload if needed.", "assumptions": ["The failed pods are managed by a Deployment, StatefulSet, Job, or another controller."], "steps": [f"List non-running pods in `{namespace}` and inspect their owner references, events, exit reasons, and recent logs.", "Determine whether the failure is caused by configuration, image pulls, capacity, or an application crash before restarting anything.", "If a managed workload needs a restart, request approval for a targeted rollout restart; do not delete pods blindly.", "Execute the approved restart and watch the rollout until replacement pods are Ready.", "Confirm the original failure does not recur and record any required configuration or capacity fix."], "risk_level": "medium", "requires_approval": True, "notes": [context.get("kubernetes", "Kubernetes availability was not checked."), "The initial investigation is read-only; restarting is a mutation."]}

    @staticmethod
    def _docker_health_plan(goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"goal": goal, "interpretation": "Diagnose Docker daemon or container health without changing Docker state.", "assumptions": ["The reported problem is local to the Docker CLI, daemon, or one of its containers."], "steps": ["Check Docker daemon connectivity and list running and stopped containers.", "Identify unhealthy, restarting, or exited containers and inspect their health status, exit code, and restart count.", "Review bounded recent logs and configuration for the affected container.", "Correlate findings with image availability, mounts, ports, configured values, and host resource pressure.", "Propose a targeted remediation; obtain approval before restarting, recreating, or removing any container."], "risk_level": "low", "requires_approval": False, "notes": [context.get("docker", "Docker availability was not checked."), "Any remediation that changes containers requires separate approval."]}

    @staticmethod
    def _general_plan(goal: str, domain: str, context: Dict[str, Any]) -> Dict[str, Any]:
        mutation = bool(re.search(r"\b(create|delete|deploy|restart|scale|update|apply|remove|run)\b", goal, re.IGNORECASE))
        return {"goal": goal, "interpretation": f"Clarify the requested {domain} operation, establish its current state, then propose a targeted next action.", "assumptions": ["The goal does not identify a specific resource or environment."], "steps": ["Identify the target resource, environment, desired outcome, and success criteria.", "Collect the minimum relevant read-only status, configuration, and recent error evidence.", "Prepare the smallest reversible change or investigation needed to meet the goal.", "Verify the outcome and document a rollback path before making any mutation."], "risk_level": "medium" if mutation else "low", "requires_approval": mutation, "notes": ["This plan performs no changes.", *[value for key, value in context.items() if key != "namespace"]]}

    @staticmethod
    def _resource_after(goal: str, resource_type: str) -> Optional[str]:
        match = re.search(rf"\b{resource_type}\s+([a-z0-9][a-z0-9._-]*)", goal, re.IGNORECASE)
        if match and match.group(1).lower() not in {"to", "in", "with"}:
            return match.group(1)
        # Natural phrasing commonly puts the resource name before its kind:
        # "scale the api deployment to 5 replicas".
        preceding = re.search(rf"\b(?:the\s+)?([a-z0-9][a-z0-9._-]*)\s+{resource_type}\b", goal, re.IGNORECASE)
        return preceding.group(1) if preceding else None
