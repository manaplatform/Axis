"""Configuration management for Axis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class Settings:
    """Simple settings loader (no external deps required for scaffold)."""

    def __init__(self) -> None:
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        self.default_model: str = os.getenv("AXIS_MODEL", "gpt-4o")

        self.config_dir: Path = Path.home() / ".axis"
        self.log_level: str = os.getenv("AXIS_LOG_LEVEL", "INFO")

        # Safety
        self.require_approval_for_mutations: bool = True
        self.prefer_gitops: bool = True

        # Kubernetes
        self.kubeconfig: Optional[str] = os.getenv("KUBECONFIG")
        self.default_namespace: str = os.getenv("AXIS_NAMESPACE", "default")

        # Docker
        self.docker_host: Optional[str] = os.getenv("DOCKER_HOST")

    def ensure_config_dir(self) -> Path:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        return self.config_dir


settings = Settings()
