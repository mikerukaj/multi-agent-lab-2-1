"""File tools confined to a run's workspace directory (contracts/tools.md)."""

from pathlib import Path

ToolOutput = tuple[str, bool]
"""Text returned to the model, and whether it reports an error."""

READ_LIMIT_BYTES = 200_000

READ_FILE_TOOL = {
    "name": "read_file",
    "description": (
        "Read a UTF-8 text file from the task workspace. "
        "Use a path relative to the workspace root."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path"}
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}

WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": (
        "Create or overwrite a UTF-8 text file in the task workspace. "
        "Parent folders are created as needed."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path"},
            "content": {"type": "string", "description": "Full file contents"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
}


def resolve_in_workspace(workspace: Path, path: str) -> Path:
    """Return the absolute location of ``path`` inside ``workspace``.

    Raises:
        ValueError: if ``path`` is absolute or resolves (through ``..`` or
            symlinks) to a location outside the workspace.
    """
    root = workspace.resolve()
    if Path(path).is_absolute():
        raise ValueError(f"Path is outside the workspace: {path}")
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Path is outside the workspace: {path}")
    return target


def read_file(workspace: Path, path: str) -> ToolOutput:
    """Read a workspace file, truncating anything beyond 200 KB."""
    try:
        target = resolve_in_workspace(workspace, path)
    except ValueError as error:
        return str(error), True
    if not target.is_file():
        return f"File not found: {path}", True

    data = target.read_bytes()
    text = data[:READ_LIMIT_BYTES].decode("utf-8", errors="replace")
    if len(data) > READ_LIMIT_BYTES:
        text += f"\n[truncated: file is {len(data)} bytes]"
    return text, False


def write_file(workspace: Path, path: str, content: str) -> ToolOutput:
    """Create or overwrite a workspace file, creating parent folders."""
    try:
        target = resolve_in_workspace(workspace, path)
    except ValueError as error:
        return str(error), True

    data = content.encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return f"Wrote {len(data)} bytes to {path}", False
