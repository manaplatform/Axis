"""CLI integration tests using fake read-only tool results."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from click.testing import CliRunner

from axis.cli import main


class StatusCommandTests(unittest.TestCase):
    @patch("axis.cli.DockerTool")
    @patch("axis.cli.KubernetesTool")
    def test_status_reports_real_collected_counts(self, kubernetes_type, docker_type) -> None:
        kubernetes = kubernetes_type.return_value
        kubernetes.get_pods.return_value = [{}, {}]
        kubernetes.get_deployments.return_value = [{}]
        kubernetes.get_nodes.return_value = [{}, {}, {}]
        docker_type.return_value.list_containers.return_value = [
            {"State": "running"},
            {"State": "exited"},
        ]

        result = CliRunner().invoke(main, ["status", "--namespace", "production"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("2 pods, 1 deployments, 3", result.output)
        self.assertIn("nodes", result.output)
        self.assertIn("1 running, 2 total containers", result.output)
        kubernetes_type.assert_called_once_with(namespace="production", context=None)
        docker_type.return_value.list_containers.assert_called_once_with(all=True)
