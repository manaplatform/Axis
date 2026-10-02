"""Infrastructure tools (Kubernetes, Docker, Cloud CLIs, GitOps, filesystem, shell)."""

from axis.tools.docker import DOCKER_FUNCTION_SCHEMAS
from axis.tools.filesystem import (
    OPENAI_FUNCTION_SCHEMAS,
    FileExistsError,
    FilesystemError,
    FilesystemTool,
)
from axis.tools.kubernetes import KUBERNETES_FUNCTION_SCHEMAS
from axis.tools.shell import OPENAI_FUNCTION_SCHEMAS as SHELL_FUNCTION_SCHEMAS
from axis.tools.shell import ShellBlockedError, ShellError, ShellTool, classify_command
from axis.tools.workspace import PathOutsideWorkspaceError, WorkspaceError

#: All deployable OpenAI function-calling schemas (filesystem + shell +
#: kubernetes + docker). The agent registry composes per-family entries
#: instead of using this flat aggregate.
ALL_FUNCTION_SCHEMAS = (
    OPENAI_FUNCTION_SCHEMAS
    + SHELL_FUNCTION_SCHEMAS
    + KUBERNETES_FUNCTION_SCHEMAS
    + DOCKER_FUNCTION_SCHEMAS
)

__all__ = [
    "OPENAI_FUNCTION_SCHEMAS",
    "SHELL_FUNCTION_SCHEMAS",
    "DOCKER_FUNCTION_SCHEMAS",
    "KUBERNETES_FUNCTION_SCHEMAS",
    "ALL_FUNCTION_SCHEMAS",
    "FileExistsError",
    "FilesystemError",
    "FilesystemTool",
    "PathOutsideWorkspaceError",
    "ShellBlockedError",
    "ShellError",
    "ShellTool",
    "WorkspaceError",
    "classify_command",
]
