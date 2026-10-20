"""Tests for the human-in-the-loop permission gate."""

from __future__ import annotations

from axis.safety import permissions
from axis.safety.permissions import PermissionGate, RiskLevel


def test_mutation_prompt_rejects_on_n_keystroke_without_enter(monkeypatch) -> None:
    keys = iter(["x", "\n", "n"])
    observed = []
    monkeypatch.setattr(permissions.click, "getchar", lambda *, echo: observed.append(echo) or next(keys))

    assert PermissionGate().check("write file", RiskLevel.MEDIUM) is False
    assert observed == [False, False, False]


def test_gate_accepts_uppercase_y_keystroke_without_enter(monkeypatch) -> None:
    monkeypatch.setattr(permissions.click, "getchar", lambda *, echo: "Y")

    assert PermissionGate().check("write file", RiskLevel.MEDIUM) is True
