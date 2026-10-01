"""Read-only Docker operations implemented through the Docker CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any


class DockerToolError(RuntimeError):
    """Base error raised while invoking docker."""


class DockerUnavailableError(DockerToolError):
    """Raised when docker cannot be found on PATH."""


class DockerCommandError(DockerToolError):
    """Raised when Docker reports an error."""


class DockerTool:
    """Small, read-only wrapper around the locally installed Docker CLI."""

    @staticmethod
    def is_available() -> bool:
        """Return whether the Docker CLI is available on PATH."""
        return shutil.which("docker") is not None

    def list_containers(self, all: bool = False) -> list[dict[str, Any]]:
        """Return containers as records from Docker's JSON lines output."""
        args = ["ps", "--format", "{{json .}}"]
        if all:
            args.append("--all")
        output = self._run(args)
        containers: list[dict[str, Any]] = []
        for line in output.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise DockerCommandError("docker returned invalid container JSON") from error
            if not isinstance(item, dict):
                raise DockerCommandError("docker returned an unexpected container record")
            containers.append(item)
        return containers

    def logs(self, container_id: str, tail: int = 100) -> str:
        """Return the most recent log lines for a container."""
        if tail < 0:
            raise ValueError("tail must be zero or greater")
        return self._run(["logs", "--tail", str(tail), container_id])

    def inspect(self, container_id: str) -> dict[str, Any]:
        """Return the Docker inspection document for a container."""
        output = self._run(["inspect", container_id])
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as error:
            raise DockerCommandError("docker returned invalid inspection JSON") from error
        if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
            raise DockerCommandError("docker returned an unexpected inspection document")
        return payload[0]

    def version_info(self) -> dict[str, Any]:
        """Return Docker client and server version details."""
        return self._run_json(["version", "--format", "{{json .}}"])

    def info(self) -> dict[str, Any]:
        """Return a machine-readable Docker daemon summary."""
        return self._run_json(["info", "--format", "{{json .}}"])

    def _run_json(self, args: list[str]) -> dict[str, Any]:
        output = self._run(args)
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as error:
            raise DockerCommandError("docker returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise DockerCommandError("docker returned an unexpected JSON document")
        return payload

    def _run(self, args: list[str]) -> str:
        if not self.is_available():
            raise DockerUnavailableError("docker is not installed or is not on PATH")

        try:
            result = subprocess.run(
                ["docker", *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            raise DockerUnavailableError(f"could not run docker: {error}") from error

        if result.returncode:
            detail = _last_error_line(result.stderr) or _last_error_line(result.stdout) or "unknown docker error"
            lowered = detail.lower()

            # Common daemon / socket problems
            if any(x in lowered for x in [
                "cannot connect to the docker daemon",
                "is the docker daemon running",
                "no such file or directory",
                "permission denied",
                "connect: no such file",
            ]):
                raise DockerUnavailableError(
                    "Docker CLI is installed but cannot connect to the Docker daemon. "
                    "Make sure Docker Desktop is running."
                )

            raise DockerCommandError(detail)

        return result.stdout


def _last_error_line(output: str) -> str:
    """Return the actionable final line from CLI error output."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""
