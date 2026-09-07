"""Permission and approval gates."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from rich.console import Console
from rich.prompt import Confirm

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

        return Confirm.ask("Do you want to proceed?", default=False)
