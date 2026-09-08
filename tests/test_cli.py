"""CLI integration tests using fake read-only tool results."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from click.testing import CliRunner
import yaml

from axis.cli import main
from axis.config import Settings


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


class PlanCommandTests(unittest.TestCase):
    @patch("axis.cli.DockerTool")
    @patch("axis.cli.KubernetesTool")
    def test_plan_renders_structured_scale_plan(self, kubernetes_type, docker_type) -> None:
        kubernetes = kubernetes_type.return_value
        kubernetes.current_context.return_value = "prod-cluster"
        kubernetes.get_deployments.return_value = [{}]

        result = CliRunner().invoke(main, ["plan", "scale the api deployment to 5 replicas", "-n", "production"])

        self.assertEqual(result.exit_code, 0, result.output)
        for heading in ("Goal", "Interpretation", "Assumptions", "Steps", "Risk level", "Requires approval"):
            self.assertIn(heading, result.output)
        self.assertIn("Yes", result.output)
        self.assertIn("context=prod-cluster", result.output)
        kubernetes_type.assert_called_once_with(namespace="production", context=None)
        docker_type.return_value.list_containers.assert_not_called()


class DiagnoseCommandTests(unittest.TestCase):
    @patch("axis.cli.DockerTool")
    @patch("axis.cli.KubernetesTool")
    def test_diagnose_docker_reports_engine_facts(self, kubernetes_type, docker_type) -> None:
        docker = docker_type.return_value
        docker.is_available.return_value = True
        docker.version_info.return_value = {"Server": {"Version": "27.0"}}
        docker.info.return_value = {}
        docker.list_containers.return_value = [{"State": "running"}, {"State": "exited"}]

        result = CliRunner().invoke(main, ["diagnose", "docker"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Docker engine", result.output)
        self.assertIn("healthy", result.output)
        self.assertIn("1 running container(s), 2 total", result.output)
        kubernetes_type.return_value.get_pods.assert_not_called()

    @patch("axis.cli.DockerTool")
    @patch("axis.cli.KubernetesTool")
    def test_diagnose_container_does_not_collect_kubernetes_details(self, kubernetes_type, docker_type) -> None:
        docker, kubernetes = docker_type.return_value, kubernetes_type.return_value
        docker.list_containers.return_value = [{"ID": "abc123def456", "Name": "api", "State": "running"}]
        docker.inspect.return_value = {"State": {"Status": "running", "ExitCode": 0}, "RestartCount": 1}
        docker.logs.return_value = "ready\n"
        kubernetes.get_pods.return_value = []
        kubernetes.get_deployments.return_value = []
        kubernetes.get_services.return_value = []

        result = CliRunner().invoke(main, ["diagnose", "abc123def456789"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Docker container", result.output)
        self.assertIn("Restart count: 1", result.output)
        kubernetes.get_events.assert_not_called()

    @patch("axis.cli.DockerTool")
    @patch("axis.cli.KubernetesTool")
    def test_diagnose_k8s_reports_cluster_connectivity(self, kubernetes_type, docker_type) -> None:
        kubernetes = kubernetes_type.return_value
        kubernetes.is_available.return_value = True
        kubernetes.current_context.return_value = "dev"
        kubernetes.get_nodes.return_value = [{"status": {"conditions": [{"type": "Ready", "status": "True"}]}}]

        result = CliRunner().invoke(main, ["diagnose", "k8s"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Kubernetes cluster", result.output)
        self.assertIn("Context: dev", result.output)
        docker_type.return_value.list_containers.assert_not_called()

    @patch("axis.cli.DockerTool")
    @patch("axis.cli.KubernetesTool")
    def test_diagnose_pod_reports_phase_events_and_logs(self, kubernetes_type, docker_type) -> None:
        docker, kubernetes = docker_type.return_value, kubernetes_type.return_value
        docker.list_containers.return_value = []
        kubernetes.get_pods.return_value = [{
            "kind": "Pod", "metadata": {"name": "api"},
            "status": {"phase": "Running", "containerStatuses": [{"restartCount": 2}]},
        }]
        kubernetes.get_deployments.return_value = []
        kubernetes.get_services.return_value = []
        kubernetes.get_events.return_value = [{"message": "Pulled image"}]
        kubernetes.logs.return_value = "application ready\n"

        result = CliRunner().invoke(main, ["diagnose", "api"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Kubernetes resource", result.output)
        self.assertIn("Container restarts: 2", result.output)
        self.assertIn("Pulled image", result.output)
        self.assertIn("application ready", result.output)

    @patch("axis.cli.DockerTool")
    @patch("axis.cli.KubernetesTool")
    def test_unknown_target_has_friendly_not_found_message(self, kubernetes_type, docker_type) -> None:
        docker, kubernetes = docker_type.return_value, kubernetes_type.return_value
        docker.list_containers.return_value = []
        kubernetes.get_pods.return_value = []
        kubernetes.get_deployments.return_value = []
        kubernetes.get_services.return_value = []

        result = CliRunner().invoke(main, ["diagnose", "missing"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Target not found", result.output)
        self.assertNotIn("inspect", result.output)


class LogsCommandTests(unittest.TestCase):
    @patch("axis.cli.DockerTool")
    @patch("axis.cli.KubernetesTool")
    def test_docker_engine_explains_that_it_has_no_container_log_stream(self, kubernetes_type, docker_type) -> None:
        result = CliRunner().invoke(main, ["logs", "docker"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("does not have a single container log stream", result.output)
        self.assertIn("docker ps", result.output)
        docker_type.return_value.logs.assert_not_called()
        kubernetes_type.return_value.logs.assert_not_called()
        docker_type.return_value.list_containers.assert_not_called()
        kubernetes_type.return_value.get_pods.assert_not_called()

    @patch("axis.cli.DockerTool")
    @patch("axis.cli.KubernetesTool")
    def test_kubernetes_cluster_explains_that_it_has_no_single_log_stream(self, kubernetes_type, docker_type) -> None:
        result = CliRunner().invoke(main, ["logs", "k8s"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("does not have a single log stream", result.output)
        self.assertIn("deployment", result.output)
        docker_type.return_value.logs.assert_not_called()
        kubernetes_type.return_value.logs.assert_not_called()
        docker_type.return_value.list_containers.assert_not_called()
        kubernetes_type.return_value.get_pods.assert_not_called()

    @patch("axis.cli.DockerTool")
    @patch("axis.cli.KubernetesTool")
    def test_container_logs_do_not_try_kubernetes_logs(self, kubernetes_type, docker_type) -> None:
        docker, kubernetes = docker_type.return_value, kubernetes_type.return_value
        docker.list_containers.return_value = [{"ID": "abc123def456", "Name": "api", "State": "running"}]
        docker.logs.return_value = "container ready\n"
        kubernetes.get_pods.return_value = []
        kubernetes.get_deployments.return_value = []
        kubernetes.get_services.return_value = []

        result = CliRunner().invoke(main, ["logs", "abc123def456789", "--tail", "25"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("container ready", result.output)
        docker.logs.assert_called_once_with("api", tail=25)
        kubernetes.logs.assert_not_called()

    @patch("axis.cli.DockerTool")
    @patch("axis.cli.KubernetesTool")
    def test_kubernetes_resource_logs_do_not_try_docker_logs(self, kubernetes_type, docker_type) -> None:
        docker, kubernetes = docker_type.return_value, kubernetes_type.return_value
        docker.list_containers.return_value = []
        kubernetes.get_pods.return_value = [{"kind": "Pod", "metadata": {"name": "api"}}]
        kubernetes.get_deployments.return_value = []
        kubernetes.get_services.return_value = []
        kubernetes.logs.return_value = "pod ready\n"

        result = CliRunner().invoke(main, ["logs", "api", "-n", "production"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("pod ready", result.output)
        kubernetes_type.assert_called_once_with(namespace="production")
        kubernetes.logs.assert_called_once_with("api", tail=100)
        docker.logs.assert_not_called()

    @patch("axis.cli.DockerTool")
    @patch("axis.cli.KubernetesTool")
    def test_unknown_target_has_suggestions_without_log_attempts(self, kubernetes_type, docker_type) -> None:
        docker, kubernetes = docker_type.return_value, kubernetes_type.return_value
        docker.list_containers.return_value = [{"Name": "api-worker"}]
        kubernetes.get_pods.return_value = []
        kubernetes.get_deployments.return_value = []
        kubernetes.get_services.return_value = []

        result = CliRunner().invoke(main, ["logs", "api"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Target not found", result.output)
        self.assertIn("container/api-worker", result.output)
        docker.logs.assert_not_called()
        kubernetes.logs.assert_not_called()


class ConfigureCommandTests(unittest.TestCase):
    def test_configure_saves_settings_and_masks_api_key(self) -> None:
        with TemporaryDirectory() as directory:
            config_dir = Path(directory) / ".axis"
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["configure"],
                input="openai\ngpt-test\nnot-a-real-api-key\nproduction\n/tmp/kubeconfig\ny\nn\nDEBUG\n",
                env={"AXIS_CONFIG_DIR": str(config_dir)},
            )

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Configuration saved successfully", result.output)
            self.assertNotIn("not-a-real-api-key", result.output)
            self.assertEqual(
                yaml.safe_load((config_dir / "config.yaml").read_text()),
                {
                    "default_model": "gpt-test",
                    "default_namespace": "production",
                    "kubeconfig": "/tmp/kubeconfig",
                    "llm_provider": "openai",
                    "log_level": "DEBUG",
                    "prefer_gitops": False,
                    "require_approval_for_mutations": True,
                },
            )
            self.assertEqual(
                yaml.safe_load((config_dir / "secrets.yaml").read_text()),
                {"openai_api_key": "not-a-real-api-key"},
            )
            self.assertEqual((config_dir / "secrets.yaml").stat().st_mode & 0o777, 0o600)

    def test_settings_load_files_but_environment_wins(self) -> None:
        with TemporaryDirectory() as directory:
            config_dir = Path(directory)
            (config_dir / "config.yaml").write_text(
                "default_model: saved-model\ndefault_namespace: saved-namespace\nprefer_gitops: false\n"
            )
            (config_dir / "secrets.yaml").write_text("openai_api_key: saved-key\n")
            with patch.dict(
                "os.environ",
                {"AXIS_MODEL": "environment-model", "AXIS_NAMESPACE": "environment-namespace"},
                clear=False,
            ):
                configured = Settings(config_dir=config_dir)

            self.assertEqual(configured.default_model, "environment-model")
            self.assertEqual(configured.default_namespace, "environment-namespace")
            self.assertEqual(configured.openai_api_key, "saved-key")
            self.assertFalse(configured.prefer_gitops)
