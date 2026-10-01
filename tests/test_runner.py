"""Tests for the agent runner loop (mocked OpenAI client; no network)."""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List

import pytest

from axis.agents.registry import (
    RegisteredTool,
    ToolRegistry,
    UnknownToolError,
    default_registry,
)
from axis.agents.runner import AgentRunner
from axis.core.llm import LLMClient
from axis.safety.permissions import RiskLevel
from rich.console import Console


# ----------------------------------------------------------------------
# fakes
# ----------------------------------------------------------------------
class FakeCall:
    type = "function_call"

    def __init__(self, name: str, arguments: str, call_id: str = "call-1") -> None:
        self.name = name
        self.arguments = arguments
        self.call_id = call_id


class FakeResponse:
    def __init__(self, output: List[Any], output_text: str = "") -> None:
        self.output = output
        self.output_text = output_text


class ScriptedResponses:
    def __init__(self, responses: List[FakeResponse]) -> None:
        self._responses = list(responses)
        self.requests: List[Dict[str, Any]] = []

    def create(self, *, model: str, input: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> FakeResponse:
        self.requests.append({"model": model, "input": input, "tools": tools})
        return self._responses.pop(0)


class ScriptedClient:
    def __init__(self, responses: List[FakeResponse]) -> None:
        self.responses = ScriptedResponses(responses)


class FakeGate:
    require_approval = True

    def __init__(self, allow: bool = True) -> None:
        self.allow = allow
        self.checks: List[tuple] = []

    def check(self, action: str, risk: RiskLevel, details: str | None = None) -> bool:
        self.checks.append((action, risk, details))
        return self.allow


def _llm() -> LLMClient:
    return LLMClient(provider="openai", model="gpt-4o", api_key="test-key")


def _runner(registry: ToolRegistry, client: ScriptedClient, **kwargs: Any) -> AgentRunner:
    return AgentRunner(
        registry=registry,
        llm=_llm(),
        console=Console(file=io.StringIO(), width=100),
        client=client,
        **kwargs,
    )


def _stub_registry() -> tuple[ToolRegistry, dict]:
    seen: dict = {}
    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            name="search_directory",
            schema={"type": "function", "name": "search_directory", "parameters": {}},
            execute=lambda args: seen.update(args) or {"matches": []},
            mutating=False,
        )
    )
    registry.register(
        RegisteredTool(
            name="create_file",
            schema={"type": "function", "name": "create_file", "parameters": {}},
            execute=lambda args: seen.update(args) or {"created": True},
            mutating=True,
        )
    )
    return registry, seen


# ----------------------------------------------------------------------
# runner loop
# ----------------------------------------------------------------------
def test_run_executes_tool_and_returns_final_answer() -> None:
    registry, seen = _stub_registry()
    client = ScriptedClient(
        [
            FakeResponse([FakeCall("search_directory", '{"pattern": "*.py"}', "c1")]),
            FakeResponse([], "All done"),
        ]
    )
    runner = _runner(registry, client, gate=FakeGate())
    assert runner.run("find python files") == "All done"
    assert seen == {"pattern": "*.py"}

    second_input = client.responses.requests[1]["input"]
    outputs = [i for i in second_input if i.get("type") == "function_call_output"]
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "c1"
    assert json.loads(outputs[0]["output"]) == {"matches": []}


def test_run_passes_registry_schemas_as_tools() -> None:
    registry, _ = _stub_registry()
    client = ScriptedClient([FakeResponse([], "hi")])
    _runner(registry, client, gate=FakeGate()).run("hi")
    assert client.responses.requests[0]["tools"] == registry.schemas()
    assert client.responses.requests[0]["model"] == "gpt-4o"


def test_echoed_function_call_has_only_api_accepted_fields() -> None:
    """Regression: SDK output items carry read-only fields (status, id, ...)
    that the Responses API rejects with 400 if echoed back."""

    class SdkLikeCall:
        type = "function_call"

        def __init__(self) -> None:
            self.call_id = "call-1"
            self.name = "search_directory"
            self.arguments = "{}"

        def model_dump(self, mode: str = "json") -> Dict[str, Any]:
            return {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call-1",
                "name": "search_directory",
                "arguments": "{}",
                "status": "completed",
            }

    registry, _ = _stub_registry()
    client = ScriptedClient(
        [
            FakeResponse([SdkLikeCall()]),
            FakeResponse([], "ok"),
        ]
    )
    _runner(registry, client, gate=FakeGate()).run("x")
    echoed = [
        i for i in client.responses.requests[1]["input"] if i.get("type") == "function_call"
    ]
    assert len(echoed) == 1
    assert set(echoed[0].keys()) == {"type", "call_id", "name", "arguments"}


def test_unknown_tool_is_reported_to_model() -> None:
    registry, seen = _stub_registry()
    client = ScriptedClient(
        [
            FakeResponse([FakeCall("nope", "{}", "c1")]),
            FakeResponse([], "ok"),
        ]
    )
    assert _runner(registry, client, gate=FakeGate()).run("x") == "ok"
    assert seen == {}
    output = client.responses.requests[1]["input"][-1]["output"]
    assert "unknown tool" in output


def test_invalid_arguments_json_is_reported() -> None:
    registry, seen = _stub_registry()
    client = ScriptedClient(
        [
            FakeResponse([FakeCall("search_directory", "not-json", "c1")]),
            FakeResponse([], "ok"),
        ]
    )
    _runner(registry, client, gate=FakeGate()).run("x")
    assert seen == {}
    output = client.responses.requests[1]["input"][-1]["output"]
    assert "invalid arguments" in output


def test_denied_mutating_tool_is_not_executed() -> None:
    registry, seen = _stub_registry()
    gate = FakeGate(allow=False)
    client = ScriptedClient(
        [
            FakeResponse([FakeCall("create_file", '{"path": "a.txt", "content": "hi"}', "c1")]),
            FakeResponse([], "ok"),
        ]
    )
    _runner(registry, client, gate=gate).run("x")
    assert seen == {}
    assert len(gate.checks) == 1
    output = client.responses.requests[1]["input"][-1]["output"]
    assert json.loads(output)["denied"] is True


def test_approved_mutating_tool_executes() -> None:
    registry, seen = _stub_registry()
    gate = FakeGate(allow=True)
    client = ScriptedClient(
        [
            FakeResponse([FakeCall("create_file", '{"path": "a.txt", "content": "hi"}', "c1")]),
            FakeResponse([], "ok"),
        ]
    )
    _runner(registry, client, gate=gate).run("x")
    assert seen == {"path": "a.txt", "content": "hi"}
    assert len(gate.checks) == 1


def test_max_steps_stops_the_loop() -> None:
    registry, _ = _stub_registry()
    client = ScriptedClient(
        [FakeResponse([FakeCall("search_directory", "{}", f"c{i}")]) for i in range(5)]
    )
    runner = _runner(registry, client, gate=FakeGate(), max_steps=2)
    assert runner.run("x") == ""
    assert len(client.responses.requests) == 2


def test_tool_exception_is_reported_not_raised() -> None:
    registry = ToolRegistry()

    def boom(args: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("disk on fire")

    registry.register(
        RegisteredTool(
            name="search_directory",
            schema={"type": "function", "name": "search_directory", "parameters": {}},
            execute=boom,
        )
    )
    client = ScriptedClient(
        [
            FakeResponse([FakeCall("search_directory", "{}", "c1")]),
            FakeResponse([], "ok"),
        ]
    )
    assert _runner(registry, client, gate=FakeGate()).run("x") == "ok"
    output = client.responses.requests[1]["input"][-1]["output"]
    assert "disk on fire" in output


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------
def test_registry_rejects_duplicates_and_unknown() -> None:
    registry = ToolRegistry()
    tool = RegisteredTool(name="a", schema={}, execute=lambda args: {})
    registry.register(tool)
    with pytest.raises(ValueError):
        registry.register(tool)
    with pytest.raises(UnknownToolError):
        registry.get("missing")


def test_default_registry_has_filesystem_tools() -> None:
    registry = default_registry(workspace_root=".")
    assert registry.names() == ["search_directory", "create_file"]
    assert all(s["type"] == "function" for s in registry.schemas())
    assert not registry.get("search_directory").mutating
    assert registry.get("create_file").mutating
