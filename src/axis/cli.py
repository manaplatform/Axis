"""Axis CLI entrypoint."""

from __future__ import annotations

import sys
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from axis import __version__
from axis.config import settings

console = Console()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="axis")
def main() -> None:
    """Axis – AI Agent for Cloud, Kubernetes & Docker Operations."""
    pass


@main.command()
def configure() -> None:
    """Interactive configuration wizard."""
    console.print(Panel.fit("[bold]Axis Configuration[/bold]", border_style="cyan"))
    settings.ensure_config_dir()

    console.print(f"Config directory: [cyan]{settings.config_dir}[/cyan]")
    console.print("\n[yellow]Configuration wizard will be expanded in the next iteration.[/yellow]")
    console.print("For now you can set environment variables or edit ~/.axis/config.toml later.")


@main.command()
@click.option("--namespace", "-n", default=None, help="Kubernetes namespace")
@click.option("--context", default=None, help="Kubernetes context")
def status(namespace: Optional[str], context: Optional[str]) -> None:
    """Show high-level status of the current environment (K8s + Docker)."""
    ns = namespace or settings.default_namespace

    console.print(Panel.fit("[bold]Axis Status[/bold]", border_style="green"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Details")

    # Placeholders – real checks will be added when tools are implemented
    table.add_row("Kubernetes", "[yellow]pending[/yellow]", f"namespace={ns}")
    table.add_row("Docker", "[yellow]pending[/yellow]", "local engine")
    table.add_row("Cloud CLIs", "[yellow]pending[/yellow]", "aws / gcloud / az")

    console.print(table)
    console.print("\n[dim]Real status collection will be implemented next.[/dim]")


@main.command()
@click.argument("target")
@click.option("--namespace", "-n", default=None)
def diagnose(target: str, namespace: Optional[str]) -> None:
    """Diagnose a service, deployment, pod or container."""
    ns = namespace or settings.default_namespace
    console.print(Panel.fit(f"[bold]Diagnosing:[/bold] {target}", border_style="blue"))
    console.print(f"Namespace: [cyan]{ns}[/cyan]")
    console.print("\n[yellow]Diagnosis engine is under construction.[/yellow]")


@main.command()
@click.argument("target")
@click.option("--namespace", "-n", default=None)
@click.option("--tail", default=100, help="Number of log lines")
def logs(target: str, namespace: Optional[str], tail: int) -> None:
    """Collect and summarize logs for a target."""
    ns = namespace or settings.default_namespace
    console.print(Panel.fit(f"[bold]Logs:[/bold] {target}", border_style="blue"))
    console.print(f"Namespace: {ns} | Tail: {tail}")
    console.print("\n[yellow]Log collection & summarization coming soon.[/yellow]")


@main.command()
@click.argument("goal", required=False)
def suggest(goal: Optional[str]) -> None:
    """Suggest actions based on current state or a goal."""
    console.print(Panel.fit("[bold]Suggestions[/bold]", border_style="magenta"))
    if goal:
        console.print(f"Goal: [cyan]{goal}[/cyan]\n")
    console.print("[yellow]Suggestion engine will be connected to the Analyzer agent next.[/yellow]")


@main.command()
@click.argument("goal")
def plan(goal: str) -> None:
    """Create an execution plan for a goal."""
    console.print(Panel.fit(f"[bold]Plan for:[/bold] {goal}", border_style="cyan"))
    console.print("\n[yellow]Planner agent will generate a structured plan here.[/yellow]")


@main.command()
def doctor() -> None:
    """Check local environment and tool availability."""
    console.print(Panel.fit("[bold]Axis Doctor[/bold]", border_style="green"))

    checks = [
        ("Python", sys.version.split()[0]),
        ("Config dir", str(settings.config_dir)),
        ("kubectl", "not checked yet"),
        ("docker", "not checked yet"),
        ("cloud CLIs", "not checked yet"),
    ]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Result")

    for name, result in checks:
        table.add_row(name, result)

    console.print(table)


if __name__ == "__main__":
    main()
