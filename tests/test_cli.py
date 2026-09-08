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
