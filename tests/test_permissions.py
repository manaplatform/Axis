"""Tests for the human-in-the-loop permission gate."""

from __future__ import annotations

from axis.safety import permissions
from axis.safety.permissions import PermissionGate, RiskLevel


def test_mutation_prompt_asks_for_approval_and_defaults_to_rejection(monkeypatch) -> None:
    observed = {}

    def ask(prompt, *, default):
        observed.update(prompt=prompt, default=default)
        return False

    monkeypatch.setattr(permissions.Confirm, "ask", ask)

    assert PermissionGate().check("write file", RiskLevel.MEDIUM) is False
    assert observed == {"prompt": "Approve?", "default": False}


def test_gate_returns_confirmation_for_mutation(monkeypatch) -> None:
    monkeypatch.setattr(permissions.Confirm, "ask", lambda prompt, *, default: True)

    assert PermissionGate().check("write file", RiskLevel.MEDIUM) is True
