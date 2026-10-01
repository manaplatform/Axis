"""Tests for provider-independent structured-plan validation."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import httpx
from axis.config import Settings
from axis.core.llm import LLMClient, LLMError


class LLMClientTests(unittest.TestCase):
    def test_unconfigured_client_does_not_make_a_request(self) -> None:
        client = LLMClient("openai", "test-model", None)

        with self.assertRaisesRegex(LLMError, "no supported LLM API key"):
            client.generate_plan("inspect docker", {"docker": "available"})

    def test_rejects_non_json_or_invalid_plan_shapes(self) -> None:
        with self.assertRaisesRegex(LLMError, "invalid structured plan"):
            LLMClient._parse("not json")
        with self.assertRaisesRegex(LLMError, "invalid structured plan"):
            LLMClient._parse('{"interpretation": "missing required fields"}')

    def test_prompt_bounds_context_and_excludes_secret_named_values(self) -> None:
        prompt = LLMClient._prompt("x" * 2000, {"docker": "d" * 500, "api_key": "must-not-leave-axis"})

        self.assertNotIn("must-not-leave-axis", prompt)
        self.assertLessEqual(len(prompt), 1600)

    def test_mutating_model_step_always_requires_approval(self) -> None:
        plan = LLMClient._parse(
            '{"interpretation":"test", "assumptions":["test"], "steps":["Build the image."], '
            '"risk_level":"low", "requires_approval":false, "notes":[]}'
        )

        self.assertTrue(plan.requires_approval)

    def test_client_uses_base_url_saved_in_configuration(self) -> None:
        with TemporaryDirectory() as directory:
            configured = Settings(Path(directory))
            configured.set_many(
                {
                    "llm_provider": "custom",
                    "default_model": "local-model",
                    "llm_base_url": "https://llm.example/v1",
                    "custom_api_key": "test-key",
                }
            )

            client = LLMClient.from_settings(configured)

            self.assertEqual(client.base_url, "https://llm.example/v1")
            self.assertTrue(client.configured)

    def test_custom_provider_retries_without_unsupported_response_format(self) -> None:
        request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
        unsupported_format = httpx.Response(400, request=request)
        success = httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"interpretation":"test", "assumptions":["test"], '
                                '"steps":["Inspect the service."], "risk_level":"low", '
                                '"requires_approval":false, "notes":[]}'
                            )
                        }
                    }
                ]
            },
        )
        http_client = MagicMock()
        http_client.__enter__.return_value = http_client
        http_client.post.side_effect = [unsupported_format, success]

        with patch("axis.core.llm.httpx.Client", return_value=http_client):
            plan = LLMClient(
                "custom", "local-model", "test-key", base_url="https://llm.example/v1"
            ).generate_plan("inspect service", {})

        self.assertEqual(plan.interpretation, "test")
        self.assertEqual(http_client.post.call_count, 2)
        first_payload = http_client.post.call_args_list[0].kwargs["json"]
        retry_payload = http_client.post.call_args_list[1].kwargs["json"]
        self.assertIn("response_format", first_payload)
        self.assertNotIn("response_format", retry_payload)

    def test_openai_provider_uses_official_responses_client(self) -> None:
        response = MagicMock()
        response.output_text = (
            '{"interpretation":"test", "assumptions":["test"], '
            '"steps":["Inspect the service."], "risk_level":"low", '
            '"requires_approval":false, "notes":[]}'
        )
        openai_client = MagicMock()
        openai_client.responses.create.return_value = response

        with patch("axis.core.llm.OpenAI", return_value=openai_client) as openai:
            plan = LLMClient("openai", "test-model", "test-key").generate_plan(
                "inspect service", {}
            )

        self.assertEqual(plan.interpretation, "test")
        openai.assert_called_once_with(api_key="test-key", timeout=12.0)
        openai_client.responses.create.assert_called_once()
        request = openai_client.responses.create.call_args.kwargs
        self.assertEqual(request["model"], "test-model")
        self.assertIn("Goal: inspect service", request["input"])

    def test_openai_client_uses_configured_base_url(self) -> None:
        response = MagicMock()
        response.output_text = (
            '{"interpretation":"test", "assumptions":["test"], '
            '"steps":["Inspect the service."], "risk_level":"low", '
            '"requires_approval":false, "notes":[]}'
        )
        openai_client = MagicMock()
        openai_client.responses.create.return_value = response

        with patch("axis.core.llm.OpenAI", return_value=openai_client) as openai:
            LLMClient("openai", "test-model", "test-key", base_url="https://api.example/v1").generate_plan(
                "inspect service", {}
            )

        openai.assert_called_once_with(
            api_key="test-key", timeout=12.0, base_url="https://api.example/v1"
        )
