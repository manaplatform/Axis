"""Infrastructure tools (Kubernetes, Docker, Cloud CLIs, GitOps, filesystem, shell)."""

from axis.tools.filesystem import (
    OPENAI_FUNCTION_SCHEMAS as _FILESYSTEM_SCHEMAS,
)
from axis.tools.filesystem import (
    FileExistsError,
    FilesystemError,
    FilesystemTool,
)
from axis.tools.shell import OPENAI_FUNCTION_SCHEMAS as _SHELL_SCHEMAS
from axis.tools.shell import ShellBlockedError, ShellError, ShellTool, classify_command
from axis.tools.workspace import PathOutsideWorkspaceError, WorkspaceError

#: All deployable OpenAI function-calling tools (Responses API ``tools``).
OPENAI_FUNCTION_SCHEMAS = _FILESYSTEM_SCHEMAS + _SHELL_SCHEMAS

__all__ = [
    "OPENAI_FUNCTION_SCHEMAS",
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
