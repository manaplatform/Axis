"""Tests for file-backed Axis configuration."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from axis.config import Settings


class SettingsTests(unittest.TestCase):
    def test_get_and_set_persist_regular_and_secret_values(self) -> None:
        with TemporaryDirectory() as directory:
            configured = Settings(Path(directory))

            self.assertEqual(configured.get("default_model"), "gpt-4o")
            self.assertEqual(configured.set("default_model", "gpt-test"), "gpt-test")
            self.assertEqual(configured.set("openai_api_key", "test-key"), "test-key")

            reloaded = Settings(Path(directory))
            self.assertEqual(reloaded.get("default_model"), "gpt-test")
            self.assertEqual(reloaded.get("openai_api_key"), "test-key")
            self.assertEqual(reloaded.secrets_path.stat().st_mode & 0o777, 0o600)
