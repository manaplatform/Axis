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
from axis.tools.docker import DockerTool, DockerToolError
from axis.tools.kubernetes import KubernetesTool, KubernetesToolError

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

    kubernetes = KubernetesTool(namespace=ns, context=context)
    try:
        pods = kubernetes.get_pods()
        deployments = kubernetes.get_deployments()
        nodes = kubernetes.get_nodes()
        table.add_row(
            "Kubernetes",
            "[green]available[/green]",
            f"namespace={ns}; {len(pods)} pods, {len(deployments)} deployments, {len(nodes)} nodes",
        )
    except KubernetesToolError as error:
        table.add_row("Kubernetes", "[yellow]unavailable[/yellow]", str(error))

    docker = DockerTool()
    try:
        containers = docker.list_containers(all=True)
        running = sum(container.get("State") == "running" for container in containers)
        table.add_row("Docker", "[green]available[/green]", f"{running} running, {len(containers)} total containers")
    except DockerToolError as error:
        table.add_row("Docker", "[yellow]unavailable[/yellow]", str(error))

    console.print(table)


@main.command()
@click.argument("target")
@click.option("--namespace", "-n", default=None)
def diagnose(target: str, namespace: Optional[str]) -> None:
    """Diagnose a service, deployment, pod or container."""
    ns = namespace or settings.default_namespace
    console.print(Panel.fit(f"[bold]Diagnosing:[/bold] {target}", border_style="blue"))

    results = Table(show_header=True, header_style="bold magenta")
    results.add_column("Source")
    results.add_column("Result")
    kubernetes = KubernetesTool(namespace=ns)
    try:
        description = kubernetes.describe("pod", target)
        results.add_row("Kubernetes pod", description.strip() or "No description returned")
    except KubernetesToolError as error:
        results.add_row("Kubernetes pod", f"[yellow]{error}[/yellow]")

    docker = DockerTool()
    try:
        inspection = docker.inspect(target)
        state = inspection.get("State", {})
        status = state.get("Status", "unknown") if isinstance(state, dict) else "unknown"
        name = str(inspection.get("Name", target)).lstrip("/")
        results.add_row("Docker container", f"name={name}; status={status}")
    except DockerToolError as error:
        results.add_row("Docker container", f"[yellow]{error}[/yellow]")
    console.print(results)


@main.command()
@click.argument("target")
@click.option("--namespace", "-n", default=None)
@click.option("--tail", default=100, help="Number of log lines")
def logs(target: str, namespace: Optional[str], tail: int) -> None:
    """Collect and summarize logs for a target."""
    ns = namespace or settings.default_namespace
    if tail < 0:
        raise click.BadParameter("must be zero or greater", param_hint="--tail")
    console.print(Panel.fit(f"[bold]Logs:[/bold] {target}", border_style="blue"))

    kubernetes = KubernetesTool(namespace=ns)
    try:
        output = kubernetes.logs(target, tail=tail)
        console.print(Panel(output.rstrip() or "No Kubernetes log lines returned.", title=f"Kubernetes pod ({ns})"))
        return
    except KubernetesToolError as kubernetes_error:
        docker = DockerTool()
        try:
            output = docker.logs(target, tail=tail)
            console.print(Panel(output.rstrip() or "No Docker log lines returned.", title="Docker container"))
            return
        except DockerToolError as docker_error:
            console.print(f"[yellow]Unable to collect logs for {target}.[/yellow]")
            console.print(f"Kubernetes: {kubernetes_error}")
            console.print(f"Docker: {docker_error}")


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
        ("kubectl", "available" if KubernetesTool.is_available() else "not found on PATH"),
        ("docker", "available" if DockerTool.is_available() else "not found on PATH"),
    ]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Result")

    for name, result in checks:
        table.add_row(name, result)

    console.print(table)


if __name__ == "__main__":
    main()
