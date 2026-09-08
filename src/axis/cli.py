"""Axis CLI entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from axis import __version__
from axis.agents.planner import PlannerAgent
from axis.config import Settings, settings
from axis.diagnosis import DiagnosisReport, TargetType, diagnose as collect_diagnosis, resolve_target
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
    wizard_settings = Settings()
    wizard_settings.ensure_config_dir()
    console.print(Panel.fit("[bold]Axis Configuration Wizard[/bold]", border_style="cyan"))
    console.print(f"Config directory: [cyan]{wizard_settings.config_dir}[/cyan]\n")

    provider = click.prompt(
        "LLM Provider",
        type=click.Choice(["openai", "anthropic", "custom"], case_sensitive=False),
        default=wizard_settings.llm_provider,
        show_default=True,
    ).lower()
    model = click.prompt("Default Model", default=wizard_settings.default_model, show_default=True)
    api_key = _prompt_api_key(provider, _provider_key(wizard_settings, provider))
    namespace = click.prompt("Default Namespace", default=wizard_settings.default_namespace, show_default=True)
    kubeconfig = click.prompt(
        "Preferred kubeconfig path (optional)", default=wizard_settings.kubeconfig or "", show_default=bool(wizard_settings.kubeconfig)
    )
    require_approval = click.confirm(
        "Require approval for mutations", default=wizard_settings.require_approval_for_mutations
    )
    prefer_gitops = click.confirm("Prefer GitOps (create PRs)", default=wizard_settings.prefer_gitops)
    log_level = click.prompt(
        "Log level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
        default=wizard_settings.log_level.upper(), show_default=True,
    ).upper()

    config = {
        "llm_provider": provider,
        "default_model": model,
        "default_namespace": namespace,
        "kubeconfig": kubeconfig or None,
        "require_approval_for_mutations": require_approval,
        "prefer_gitops": prefer_gitops,
        "log_level": log_level,
    }
    secrets = wizard_settings._load_yaml(wizard_settings.secrets_path)
    if api_key:
        secrets[f"{provider}_api_key"] = api_key
    _write_yaml(wizard_settings.config_path, config, secret=False)
    _write_yaml(wizard_settings.secrets_path, secrets, secret=True)

    console.print("\n[green]Configuration saved successfully.[/green]")
    console.print(f"Settings: [cyan]{wizard_settings.config_path}[/cyan]")
    console.print(f"Secrets: [cyan]{wizard_settings.secrets_path}[/cyan] (API keys hidden)")
    approval_summary = "required" if require_approval else "not required"
    gitops_summary = "preferred" if prefer_gitops else "direct changes allowed"
    console.print(
        f"Provider: {provider}; model: {model}; namespace: {namespace}; "
        f"approval: {approval_summary}; GitOps: {gitops_summary}"
    )


def _provider_key(current_settings: Settings, provider: str) -> Optional[str]:
    return getattr(current_settings, f"{provider}_api_key")


def _prompt_api_key(provider: str, current_key: Optional[str]) -> str:
    """Prompt without displaying either an entered or an existing secret."""
    hint = f" [{_mask_secret(current_key)}]" if current_key else " (optional)"
    value = click.prompt(
        f"{provider.title()} API Key{hint}", default="", show_default=False, hide_input=True
    )
    return value or current_key or ""


def _mask_secret(value: str) -> str:
    return "*" * 6 + value[-4:] if len(value) > 4 else "*" * 6


def _write_yaml(path: Path, values: dict, *, secret: bool) -> None:
    """Write configuration atomically and make the secrets file owner-readable only."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(values, config_file, default_flow_style=False, sort_keys=True)
    temporary_path.replace(path)
    if secret:
        path.chmod(0o600)


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

    kubernetes = KubernetesTool(namespace=ns)
    docker = DockerTool()
    resolution = resolve_target(target, docker, kubernetes)
    if not resolution.targets:
        details = [f"[yellow]No Docker container or Kubernetes pod, deployment, or service named '{target}' was found.[/yellow]"]
        if resolution.suggestions:
            details.append("Similar targets: " + ", ".join(dict.fromkeys(resolution.suggestions)))
        if resolution.docker_problem:
            details.append(resolution.docker_problem)
        if resolution.kubernetes_problem:
            details.append(resolution.kubernetes_problem)
        console.print(Panel("\n".join(details), title="Target not found", border_style="yellow"))
        return
    for resolved in resolution.targets:
        _print_diagnosis(collect_diagnosis(resolved, docker, kubernetes))


def _print_diagnosis(report: DiagnosisReport) -> None:
    """Render a diagnosis without exposing command output or tracebacks."""
    color = {"healthy": "green", "warning": "yellow", "unhealthy": "red", "unavailable": "yellow"}[report.status]
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Target", report.target.name)
    table.add_row("Type", report.target.target_type.value)
    table.add_row("Status", f"[{color}]{report.status}[/{color}]")
    table.add_row("Findings", "\n".join(f"• {item}" for item in report.findings))
    if report.events:
        table.add_row("Recent events / logs", "\n".join(f"• {item}" for item in report.events))
    table.add_row("Next actions", "\n".join(f"• {item}" for item in report.actions))
    console.print(Panel(table, title="Diagnosis", border_style=color))


@main.command()
@click.argument("target")
@click.option("--namespace", "-n", default=None)
@click.option("--tail", default=100, help="Number of log lines")
def logs(target: str, namespace: Optional[str], tail: int) -> None:
    """Collect logs for a resolved Docker container or Kubernetes resource."""
    ns = namespace or settings.default_namespace
    if tail < 0:
        raise click.BadParameter("must be zero or greater", param_hint="--tail")
    console.print(Panel.fit(f"[bold]Logs:[/bold] {target}", border_style="blue"))

    kubernetes = KubernetesTool(namespace=ns)
    docker = DockerTool()
    resolution = resolve_target(target, docker, kubernetes)

    if not resolution.targets:
        _print_log_target_not_found(target, resolution.suggestions, resolution.docker_problem, resolution.kubernetes_problem)
        return

    for resolved in resolution.targets:
        if resolved.target_type is TargetType.DOCKER_ENGINE:
            console.print(
                Panel(
                    "The Docker engine does not have a single container log stream. "
                    "Choose a running container with `docker ps`, then run `axis logs <container>`.",
                    title="Docker engine logs",
                    border_style="yellow",
                )
            )
        elif resolved.target_type is TargetType.KUBERNETES_CLUSTER:
            console.print(
                Panel(
                    "A Kubernetes cluster does not have a single log stream. "
                    "Choose a pod or deployment, then run `axis logs <target> -n <namespace>`.",
                    title="Kubernetes cluster logs",
                    border_style="yellow",
                )
            )
        elif resolved.target_type is TargetType.DOCKER_CONTAINER:
            _print_docker_logs(resolved.name, docker, tail)
        else:
            _print_kubernetes_logs(resolved.name, kubernetes, ns, tail)


def _print_docker_logs(container: str, docker: DockerTool, tail: int) -> None:
    try:
        output = docker.logs(container, tail=tail)
    except DockerToolError as error:
        console.print(Panel(str(error), title=f"Unable to collect Docker logs for {container}", border_style="yellow"))
        return
    console.print(Panel(output.rstrip() or "No Docker log lines returned.", title="Docker container"))


def _print_kubernetes_logs(resource: str, kubernetes: KubernetesTool, namespace: str, tail: int) -> None:
    try:
        output = kubernetes.logs(resource, tail=tail)
    except KubernetesToolError as error:
        console.print(
            Panel(str(error), title=f"Unable to collect Kubernetes logs for {resource}", border_style="yellow")
        )
        return
    console.print(Panel(output.rstrip() or "No Kubernetes log lines returned.", title=f"Kubernetes resource ({namespace})"))


def _print_log_target_not_found(
    target: str, suggestions: list[str], docker_problem: Optional[str], kubernetes_problem: Optional[str]
) -> None:
    details = [f"[yellow]No Docker container or Kubernetes pod, deployment, or service named '{target}' was found.[/yellow]"]
    if suggestions:
        details.append("Similar targets: " + ", ".join(dict.fromkeys(suggestions)))
    if docker_problem:
        details.append(docker_problem)
    if kubernetes_problem:
        details.append(kubernetes_problem)
    console.print(Panel("\n".join(details), title="Target not found", border_style="yellow"))


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
@click.option("--namespace", "namespace", "-n", default=None, help="Kubernetes namespace for plan context")
@click.option("--context", "kube_context", default=None, help="Kubernetes context for plan context")
def plan(goal: str, namespace: Optional[str], kube_context: Optional[str]) -> None:
    """Create an execution plan for a goal."""
    ns = PlannerAgent.namespace_from_goal(goal) or namespace or settings.default_namespace
    console.print(Panel.fit(f"[bold]Plan for:[/bold] {goal}", border_style="cyan"))
    plan_result = PlannerAgent().create_plan(
        goal,
        docker=DockerTool(),
        kubernetes=KubernetesTool(namespace=ns, context=kube_context),
        namespace=ns,
    )
    _print_plan(plan_result)


def _print_plan(plan_result: dict) -> None:
    """Render a planner result as a human-executable checklist."""
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    table.add_row("Goal", plan_result["goal"])
    table.add_row("Domain", plan_result["domain"].title())
    table.add_row("Interpretation", plan_result["interpretation"])
    table.add_row("Assumptions", "\n".join(f"• {item}" for item in plan_result["assumptions"]))
    table.add_row("Steps", "\n".join(f"{index}. {item}" for index, item in enumerate(plan_result["steps"], start=1)))
    table.add_row("Risk level", plan_result["risk_level"].title())
    table.add_row("Requires approval", "Yes" if plan_result["requires_approval"] else "No")
    table.add_row("Notes / warnings", "\n".join(f"• {item}" for item in plan_result["notes"]))
    console.print(Panel(table, title="Execution plan", border_style="cyan"))


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
