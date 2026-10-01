"""Small, provider-isolated client for read-only structured planning."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from axis.config import Settings


class LLMError(RuntimeError):
    """A provider response could not safely be used as a plan."""


class Plan(BaseModel):
    """The only model-generated portion of an Axis plan."""

    interpretation: str = Field(min_length=1, max_length=600)
    assumptions: list[str] = Field(min_length=1, max_length=8)
    steps: list[str] = Field(min_length=1, max_length=12)
    risk_level: str
    requires_approval: bool
    notes: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("assumptions", "steps", "notes")
    @classmethod
    def bounded_items(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            if not isinstance(value, str):
                raise ValueError("plan list items must be strings")
            item = value.strip()
            if not item or len(item) > 600:
                raise ValueError("plan list items must be non-empty, short strings")
            cleaned.append(item)
        return cleaned

    @field_validator("risk_level")
    @classmethod
    def valid_risk_level(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"low", "medium", "high"}:
            raise ValueError("risk_level must be low, medium, or high")
        return value

    @model_validator(mode="after")
    def require_approval_for_mutations(self) -> "Plan":
        """Do not allow a model to describe a mutation as approval-free."""
        mutation_words = (
            r"\b(build|run|create|delete|deploy|restart|scale|update|apply|"
            r"remove|write|modify|push|publish)\b"
        )
        if any(re.search(mutation_words, step, re.IGNORECASE) for step in self.steps):
            self.requires_approval = True
        return self


class LLMClient:
    """Generate a validated plan from OpenAI, compatible, or Anthropic APIs."""

    timeout_seconds = 12.0

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str],
        *,
        base_url: Optional[str] = None,
    ) -> None:
        self.provider = (provider or "openai").lower().strip()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @classmethod
    def from_settings(cls, configured: Settings) -> "LLMClient":
        # Be defensive: older Settings objects may not have all fields yet.
        provider = getattr(configured, "llm_provider", None) or "openai"
        model = getattr(configured, "default_model", None) or "gpt-4o"
        base_url = getattr(configured, "llm_base_url", None)

        provider = str(provider).lower().strip()
        if provider == "anthropic":
            api_key = getattr(configured, "anthropic_api_key", None)
        elif provider == "custom":
            api_key = getattr(configured, "custom_api_key", None) or getattr(
                configured, "openai_api_key", None
            )
        else:
            # default / openai
            provider = "openai"
            api_key = getattr(configured, "openai_api_key", None)

        return cls(provider, model, api_key, base_url=base_url)

    @property
    def configured(self) -> bool:
        if not self.api_key:
            return False
        if self.provider in {"openai", "anthropic"}:
            return True
        if self.provider == "custom":
            return bool(self.base_url)
        return False

    def generate_plan(self, goal: str, context: Mapping[str, Any]) -> Plan:
        """Call the configured provider. This method never executes plan steps."""
        if not self.configured:
            raise LLMError("no supported LLM API key is configured")

        prompt = self._prompt(goal, context)
        try:
            if self.provider == "openai":
                content = self._openai(prompt)
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    if self.provider == "custom":
                        content = self._custom_openai(client, prompt)
                    elif self.provider == "anthropic":
                        content = self._anthropic(client, prompt)
                    else:
                        raise LLMError(f"unsupported LLM provider: {self.provider}")
        except APITimeoutError as error:
            raise LLMError("LLM request timed out") from error
        except APIStatusError as error:
            raise LLMError(f"LLM provider request failed (HTTP {error.status_code})") from error
        except APIConnectionError as error:
            raise LLMError("LLM provider request failed") from error
        except httpx.TimeoutException as error:
            raise LLMError("LLM request timed out") from error
        except httpx.HTTPStatusError as error:
            # Status codes are useful configuration diagnostics but response
            # bodies can contain provider-specific details that Axis should not
            # surface (or accidentally persist in a plan).
            raise LLMError(f"LLM provider request failed (HTTP {error.response.status_code})") from error
        except httpx.HTTPError as error:
            raise LLMError("LLM provider request failed") from error

        return self._parse(content)

    def _openai(self, prompt: str) -> str:
        """Generate a plan through the official OpenAI Responses client."""
        client_options: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout_seconds,
        }
        if self.base_url:
            client_options["base_url"] = self.base_url
        client = OpenAI(**client_options)
        response = client.responses.create(
            model=self.model,
            input=f"{_SYSTEM_PROMPT}\n\n{prompt}",
        )
        content = response.output_text
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM provider returned empty content")
        return content

    def _custom_openai(self, client: httpx.Client, prompt: str) -> str:
        """Call a configured OpenAI-compatible endpoint over its chat API."""
        url = (self.base_url or "").rstrip("/") + "/chat/completions"
        response = client.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
            },
        )
        # ``response_format`` is optional. Some compatible endpoints reject it
        # even though they can return JSON in ordinary message content. Retry
        # once without the optional parameter.
        if response.status_code in {400, 422}:
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                },
            )
        response.raise_for_status()
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise LLMError("LLM provider returned an invalid response") from error
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM provider returned empty content")
        return content

    def _anthropic(self, client: httpx.Client, prompt: str) -> str:
        url = (self.base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        response = client.post(
            url,
            headers={
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 1200,
                "system": _SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
        )
        response.raise_for_status()
        try:
            blocks = response.json()["content"]
            content = next(block["text"] for block in blocks if block.get("type") == "text")
        except (KeyError, StopIteration, TypeError, ValueError) as error:
            raise LLMError("LLM provider returned an invalid response") from error
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM provider returned empty content")
        return content

    @staticmethod
    def _prompt(goal: str, context: Mapping[str, Any]) -> str:
        # Context is created from short local observations; bound it again before it leaves Axis.
        safe_context: dict[str, str] = {}
        for key, value in context.items():
            key_s = str(key)
            if "key" in key_s.lower() or "secret" in key_s.lower() or "token" in key_s.lower():
                continue
            safe_context[key_s[:80]] = str(value)[:400]
        return (
            f"Goal: {goal[:1000]}\n"
            f"Local read-only context: {json.dumps(safe_context, ensure_ascii=True)}"
        )

    @staticmethod
    def _parse(content: str) -> Plan:
        # Some models wrap JSON in markdown fences despite instructions.
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return Plan.model_validate(json.loads(cleaned))
        except (json.JSONDecodeError, ValidationError) as error:
            raise LLMError("LLM returned an invalid structured plan") from error


_SYSTEM_PROMPT = """You are Axis's infrastructure planning assistant. Return exactly one JSON object, no Markdown.
It must have: interpretation (string), assumptions (array of strings), steps (array of strings),
risk_level (low|medium|high), requires_approval (boolean), notes (array of strings).
Planning is read-only: never execute commands or claim actions were performed. Prefer safe, reversible
steps. Clearly mark any mutation as requiring approval. Do not request, repeat, or expose secrets."""
