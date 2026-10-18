"""Tools the Coder Agent can call, and a dispatcher that never raises."""

import json
from pathlib import Path

from tools.file_tools import READ_FILE_TOOL, WRITE_FILE_TOOL, read_file, write_file
from tools.python_exec import EXEC_PYTHON_TOOL, SANDBOX_IMAGE, ExecutionResult, exec_python

PURPOSE_PROPERTY = {
    "type": "string",
    "description": "One short sentence: what this call is for and why.",
}


def _with_purpose(tool: dict) -> dict:
    """Add a required ``purpose`` argument, so every step's trace says why it ran."""
    schema = tool["input_schema"]
    return {
        **tool,
        "input_schema": {
            **schema,
            "properties": {**schema["properties"], "purpose": PURPOSE_PROPERTY},
            "required": [*schema["required"], "purpose"],
        },
    }


TOOL_DEFINITIONS = [_with_purpose(t) for t in (READ_FILE_TOOL, WRITE_FILE_TOOL, EXEC_PYTHON_TOOL)]


def dispatch_tool(
    name: str,
    tool_input: dict,
    workspace: Path,
    timeout_seconds: float,
    image: str = SANDBOX_IMAGE,
) -> tuple[str, bool, ExecutionResult | None]:
    """Run the named tool and describe the outcome for the model.

    Returns the text for the ``tool_result`` block, whether it is an error,
    and the ``ExecutionResult`` when ``exec_python`` ran. Tool failures come
    back as error text so the model can recover; they are never raised.
    """
    try:
        if name == "read_file":
            return (*read_file(workspace, tool_input["path"]), None)
        if name == "write_file":
            text, is_error = write_file(workspace, tool_input["path"], tool_input["content"])
            return text, is_error, None
        if name == "exec_python":
            result = exec_python(workspace, tool_input["path"], timeout_seconds, image)
            return json.dumps(result.to_dict()), False, result
    except (ValueError, FileNotFoundError) as error:
        return str(error), True, None
    except KeyError as error:
        return f"Missing tool argument: {error}", True, None
    return f"Unknown tool: {name}", True, None
