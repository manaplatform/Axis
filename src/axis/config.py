"""Configuration management for Axis.

Non-secret settings live in ``~/.axis/config.yaml``. Provider API keys are kept
in ``~/.axis/secrets.yaml`` so they are not accidentally included with ordinary
configuration when it is shared or inspected.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


class Settings:
    """Load Axis settings from files, with environment variables taking priority."""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        configured_dir = os.getenv("AXIS_CONFIG_DIR")
        self.config_dir = config_dir or (Path(configured_dir) if configured_dir else Path.home() / ".axis")
        config = self._load_yaml(self.config_path)
        secrets = self._load_yaml(self.secrets_path)

        self.llm_provider: str = os.getenv("AXIS_LLM_PROVIDER", str(config.get("llm_provider", "openai")))
        self.default_model: str = os.getenv("AXIS_MODEL", str(config.get("default_model", "gpt-4o")))
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY") or self._optional_string(
            secrets.get("openai_api_key")
        )
        self.anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY") or self._optional_string(
            secrets.get("anthropic_api_key")
        )
        self.custom_api_key: Optional[str] = os.getenv("AXIS_CUSTOM_API_KEY") or self._optional_string(
            secrets.get("custom_api_key")
        )
        self.log_level: str = os.getenv("AXIS_LOG_LEVEL", str(config.get("log_level", "INFO")))

        self.require_approval_for_mutations = self._env_bool(
            "AXIS_REQUIRE_APPROVAL", config.get("require_approval_for_mutations", True)
        )
        self.prefer_gitops = self._env_bool("AXIS_PREFER_GITOPS", config.get("prefer_gitops", True))
        self.kubeconfig: Optional[str] = os.getenv("KUBECONFIG") or self._optional_string(
            config.get("kubeconfig")
        )
        self.default_namespace: str = os.getenv(
            "AXIS_NAMESPACE", str(config.get("default_namespace", "default"))
        )
        self.docker_host: Optional[str] = os.getenv("DOCKER_HOST")

    @property
    def config_path(self) -> Path:
        return self.config_dir / "config.yaml"

    @property
    def secrets_path(self) -> Path:
        return self.config_dir / "secrets.yaml"

    def ensure_config_dir(self) -> Path:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        return self.config_dir

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        try:
            with path.open(encoding="utf-8") as config_file:
                contents = yaml.safe_load(config_file)
        except (OSError, yaml.YAMLError):
            return {}
        return contents if isinstance(contents, dict) else {}

    @staticmethod
    def _optional_string(value: Any) -> Optional[str]:
        return str(value) if value not in (None, "") else None

    @staticmethod
    def _env_bool(name: str, default: Any) -> bool:
        value = os.getenv(name)
        if value is None:
            return bool(default)
        return value.strip().lower() in {"1", "true", "yes", "on"}


settings = Settings()
