"""Tests for the subprocess-backed infrastructure tools."""

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from axis.tools.docker import DockerTool, DockerUnavailableError
from axis.tools.kubernetes import KubernetesCommandError, KubernetesTool, KubernetesUnavailableError


class KubernetesToolTests(unittest.TestCase):
    @patch("axis.tools.kubernetes.shutil.which", return_value="/usr/bin/kubectl")
    @patch("axis.tools.kubernetes.subprocess.run")
    def test_get_pods_uses_namespace_context_and_selector(self, run, _which) -> None:
        run.return_value = subprocess.CompletedProcess(
            [], 0, '{"items": [{"metadata": {"name": "api"}}]}', ""
        )
        tool = KubernetesTool(namespace="production", context="cluster-a")

        pods = tool.get_pods("app=api")

        self.assertEqual(pods[0]["metadata"]["name"], "api")
        run.assert_called_once_with(
            ["kubectl", "--context", "cluster-a", "--namespace", "production", "get", "pods", "-o", "json", "--selector", "app=api"],
            capture_output=True,
            text=True,
            check=False,
        )

    @patch("axis.tools.kubernetes.shutil.which", return_value=None)
    def test_missing_kubectl_has_helpful_error(self, _which) -> None:
        with self.assertRaisesRegex(KubernetesUnavailableError, "not installed"):
            KubernetesTool().get_nodes()

    @patch("axis.tools.kubernetes.shutil.which", return_value="/usr/bin/kubectl")
    @patch("axis.tools.kubernetes.subprocess.run")
    def test_kubectl_failure_uses_final_actionable_error(self, run, _which) -> None:
        run.return_value = subprocess.CompletedProcess([], 1, "", "debug detail\nUnable to connect to the server")

        with self.assertRaisesRegex(KubernetesCommandError, "^Unable to connect to the server$"):
            KubernetesTool().get_pods()


class DockerToolTests(unittest.TestCase):
    @patch("axis.tools.docker.shutil.which", return_value="/usr/bin/docker")
    @patch("axis.tools.docker.subprocess.run")
    def test_list_containers_parses_json_lines(self, run, _which) -> None:
        run.return_value = subprocess.CompletedProcess(
            [], 0, '{"ID":"abc","State":"running"}\n{"ID":"def","State":"exited"}\n', ""
        )

        containers = DockerTool().list_containers(all=True)

        self.assertEqual([container["ID"] for container in containers], ["abc", "def"])
        run.assert_called_once_with(
            ["docker", "ps", "--format", "{{json .}}", "--all"],
            capture_output=True,
            text=True,
            check=False,
        )

    @patch("axis.tools.docker.shutil.which", return_value=None)
    def test_missing_docker_has_helpful_error(self, _which) -> None:
        with self.assertRaisesRegex(DockerUnavailableError, "not installed"):
            DockerTool().list_containers()
