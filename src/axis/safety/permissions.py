"""Permission and approval gates."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import click
from rich.console import Console

console = Console()


class RiskLevel(str, Enum):
    READ = "read"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PermissionGate:
    """Simple human-in-the-loop approval system."""

    def __init__(self, require_approval: bool = True):
        self.require_approval = require_approval

    def check(self, action: str, risk: RiskLevel, details: Optional[str] = None) -> bool:
        """Return True if the action is allowed to proceed."""
        if risk == RiskLevel.READ:
            return True

        if not self.require_approval:
            return True

        console.print()
        console.print(f"[bold yellow]Approval required[/bold yellow]")
        console.print(f"Action : {action}")
        console.print(f"Risk   : [bold]{risk.value}[/bold]")
        if details:
            console.print(f"Details: {details}")

        return _confirm_approval()


def _confirm_approval() -> bool:
    """Wait for a y/n keystroke without requiring the user to press Enter."""
    console.print("Approve? [y/n] ", end="")
    while True:
        choice = click.getchar(echo=False).lower()
        if choice in ("y", "n"):
            console.print(choice)
            return choice == "y"
