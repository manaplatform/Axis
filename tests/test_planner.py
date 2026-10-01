"""Tests for structured, read-only plans."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from axis.agents.planner import PlannerAgent
from axis.core.llm import LLMError, Plan
from axis.tools.docker import DockerUnavailableError
from axis.tools.kubernetes import KubernetesUnavailableError


class PlannerAgentTests(unittest.TestCase):
    def test_uses_validated_llm_plan_when_a_provider_is_configured(self) -> None:
        llm = Mock()
        llm.configured = True
        llm.generate_plan.return_value = Plan(
            interpretation="Containerize the application for local testing.",
            assumptions=["A documented start command exists."],
            steps=["Inspect the runtime.", "Request approval before building or running a container."],
            risk_level="medium",
            requires_approval=True,
            notes=["No commands have been run."],
        )

        plan = PlannerAgent(llm_client=llm).create_plan("create a docker container for this repo")

        self.assertEqual(plan["engine"], "llm")
        self.assertEqual(plan["goal"], "create a docker container for this repo")
        self.assertEqual(plan["interpretation"], "Containerize the application for local testing.")
        self.assertEqual(plan["domain"], "docker")
        llm.generate_plan.assert_called_once()

    def test_llm_failure_falls_back_to_local_plan_with_a_clear_note(self) -> None:
        llm = Mock()
        llm.configured = True
        llm.generate_plan.side_effect = LLMError("LLM request timed out")

        plan = PlannerAgent(llm_client=llm).create_plan("create a docker container for this repo")

        self.assertEqual(plan["engine"], "local")
        self.assertIn("Package this repository", plan["interpretation"])
        self.assertTrue(any("LLM unavailable" in note for note in plan["notes"]))
        self.assertEqual(plan["error"], "LLM request timed out")

    def test_scale_plan_uses_light_kubernetes_context_and_requires_approval(self) -> None:
        kubernetes = Mock()
        kubernetes.current_context.return_value = "production-cluster"
        kubernetes.get_deployments.return_value = [{}, {}]

        plan = PlannerAgent().create_plan(
            "scale the api deployment to 5 replicas", kubernetes=kubernetes, namespace="production"
        )

        self.assertEqual(plan["domain"], "kubernetes")
        self.assertTrue(plan["requires_approval"])
        self.assertEqual(plan["risk_level"], "medium")
        self.assertIn("deployment `api` to 5 replicas", plan["interpretation"])
        self.assertIn("context=production-cluster", plan["context"]["kubernetes"])
        kubernetes.current_context.assert_called_once_with()
        kubernetes.get_deployments.assert_called_once_with()

    def test_docker_health_plan_is_read_only_and_handles_unavailable_daemon(self) -> None:
        docker = Mock()
        docker.list_containers.side_effect = DockerUnavailableError("daemon unavailable")

        plan = PlannerAgent().create_plan("check why docker is unhealthy", docker=docker)

        self.assertEqual(plan["domain"], "docker")
        self.assertFalse(plan["requires_approval"])
        self.assertEqual(plan["risk_level"], "low")
        self.assertIn("unavailable: daemon unavailable", plan["context"]["docker"])
        docker.list_containers.assert_called_once_with(all=True)

    def test_restart_plan_uses_namespace_in_goal_and_marks_mutation(self) -> None:
        kubernetes = Mock()
        kubernetes.current_context.side_effect = KubernetesUnavailableError("no cluster")

        plan = PlannerAgent().create_plan("restart failed pods in namespace payments", kubernetes=kubernetes)

        self.assertTrue(plan["requires_approval"])
        self.assertIn("namespace `payments`", plan["interpretation"])
        self.assertIn("unavailable: no cluster", plan["context"]["kubernetes"])
        kubernetes.get_deployments.assert_not_called()
