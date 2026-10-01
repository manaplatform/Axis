"""Persistent configuration for Axis.

Axis reads all settings, including LLM credentials, from ``~/.axis``. Values
are changed through :meth:`Settings.set` or the ``axis configure`` wizard;
environment variables are deliberately not configuration inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


_DEFAULTS: dict[str, Any] = {
    "llm_provider": "openai", "default_model": "gpt-4o", "llm_base_url": None,
    "log_level": "INFO", "require_approval_for_mutations": True,
    "prefer_gitops": True, "kubeconfig": None, "default_namespace": "default",
    "docker_host": None,
}
_SECRET_KEYS = {"openai_api_key", "anthropic_api_key", "custom_api_key"}


class Settings:
    """Get and persist Axis configuration without environment-variable overrides."""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        self.config_dir = config_dir or Path.home() / ".axis"
        self._config = self._load_yaml(self.config_path)
        self._secrets = self._load_yaml(self.secrets_path)
        self._refresh_attributes()

    @property
    def config_path(self) -> Path:
        return self.config_dir / "config.yaml"

    @property
    def secrets_path(self) -> Path:
        return self.config_dir / "secrets.yaml"

    def ensure_config_dir(self) -> Path:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        return self.config_dir

    def get(self, key: str, default: Any = None) -> Any:
        """Return a configured value, or ``default`` when it is absent."""
        if key in _SECRET_KEYS:
            value = self._optional_string(self._secrets.get(key))
            return value if value is not None else default
        return self._config.get(key, _DEFAULTS.get(key, default))
    
    def set(self, key: str, value: Any) -> Any:
        """Persist one value and return the saved configuration value."""
        return self.set_many({key: value})[key]

    def set_many(self, values: dict[str, Any]) -> dict[str, Any]:
        """Persist multiple values and return their configured values.

        Provider API keys are placed in the private secrets file. All other
        values are stored in the ordinary configuration file.
        """
        for key, value in values.items():
            destination = self._secrets if key in _SECRET_KEYS else self._config
            if value is None:
                destination.pop(key, None)
            else:
                destination[key] = value
        self.ensure_config_dir()
        if any(key not in _SECRET_KEYS for key in values):
            self._write_yaml(self.config_path, self._config, secret=False)
        if any(key in _SECRET_KEYS for key in values):
            self._write_yaml(self.secrets_path, self._secrets, secret=True)
        self._refresh_attributes()
        return {key: self.get(key) for key in values}

    def _refresh_attributes(self) -> None:
        for key, default in _DEFAULTS.items():
            setattr(self, key, self.get(key, default))
        for key in _SECRET_KEYS:
            setattr(self, key, self._optional_string(self.get(key)))

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
    def _write_yaml(path: Path, values: dict[str, Any], *, secret: bool) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(values, config_file, default_flow_style=False, sort_keys=True)
        temporary_path.replace(path)
        if secret:
            path.chmod(0o600)


settings = Settings()


def get_configure(key: str, default: Any = None) -> Any:
    """Return a value from the process-wide Axis configuration."""
    return settings.get(key, default)


def set_configure(key: str, value: Any) -> Any:
    """Save a value in the process-wide Axis configuration and return it."""
    return settings.set(key, value)
