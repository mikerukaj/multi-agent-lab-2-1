"""Offline stand-ins for the PM's Anthropic client and for the Coder Agent.

Shared by the team tests. The file name does not match ``test*.py``, so
``unittest discover`` never runs it as a test module.
"""

import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

from agents.coder_agent import CoderResult

PRIORITY_CYCLE = ("high", "medium", "low")


def pm_response(
    payload: dict | None = None, stop_reason: str = "end_turn", stop_details=None
) -> SimpleNamespace:
    """A model response whose text is ``payload`` as JSON (no content if ``None``)."""
    content = [] if payload is None else [SimpleNamespace(type="text", text=json.dumps(payload))]
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )


def task_item(number: int, criteria_prefix: str = "Task") -> dict:
    """A model-facing task (no ``task_id``) with unique acceptance criteria."""
    criteria_count = 1 if number % 2 else 2
    return {
        "title": f"Task {number}",
        "description": f"Build part {number} of the program.",
        "acceptance_criteria": [
            f"{criteria_prefix} {number} criterion {j}" for j in range(1, criteria_count + 1)
        ],
        "priority": PRIORITY_CYCLE[(number - 1) % 3],
    }


def spec_payload(task_count: int, rejection_reason: str = "") -> dict:
    """A model-facing spec as in contracts/pm-agent-api.md ``SPEC_SCHEMA``."""
    tasks = [] if rejection_reason else [task_item(n) for n in range(1, task_count + 1)]
    return {
        "name": "Temperature converter",
        "description": "Converts temperatures between Celsius and Fahrenheit.",
        "acceptance_criteria": ["100 C converts to 212 F", "32 F converts to 0 C"],
        "assumptions": ["Input is a number followed by C or F"],
        "tasks": tasks,
        "rejection_reason": rejection_reason,
    }


def tasks_payload(task_count: int) -> dict:
    """A consolidation reply with fresh criteria (for cases where criteria do not matter)."""
    return {"tasks": [task_item(n, "Other") for n in range(1, task_count + 1)]}


def regroup_payload(tasks: list[dict], group_count: int) -> dict:
    """A well-behaved consolidation: contiguous groups, criteria copied in order."""
    size, extra = divmod(len(tasks), group_count)
    merged, start = [], 0
    for index in range(group_count):
        group = tasks[start : start + size + (1 if index < extra else 0)]
        start += len(group)
        merged.append(
            {
                "title": " + ".join(task["title"] for task in group),
                "description": " ".join(task["description"] for task in group),
                "acceptance_criteria": [c for task in group for c in task["acceptance_criteria"]],
                "priority": group[0]["priority"],
            }
        )
    return {"tasks": merged}


class FakePMClient:
    """Stands in for ``anthropic.Anthropic``: replays scripted responses or errors."""

    def __init__(self, script: list) -> None:
        self._script = iter(script)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs)))
        item = next(self._script)
        if isinstance(item, Exception):
            raise item
        return item


class FakeCoder:
    """Stands in for ``CoderAgent``: writes one file per task and reports a result."""

    def __init__(
        self,
        fail_on: frozenset[str] = frozenset(),
        failed_status: str = "step_limit",
        delay_seconds: float = 0.0,
    ) -> None:
        self.fail_on = fail_on
        self.failed_status = failed_status
        self.delay_seconds = delay_seconds
        self.calls: list[tuple[str, str, Path]] = []

    def run(self, task: str, workspace: Path) -> CoderResult:
        """Record the call, write ``<task id>.py``, and return a scripted result."""
        task_id = re.search(r"Task (T\d+) of", task).group(1)
        self.calls.append((task_id, task, workspace))
        time.sleep(self.delay_seconds)
        file_name = f"{task_id.lower()}.py"
        code = f"print('{task_id}')\n"
        (workspace / file_name).write_text(code, encoding="utf-8")
        failed = task_id in self.fail_on
        return CoderResult(
            code=code,
            explanation=f"Built {task_id}",
            plan=[f"Write {task_id}"],
            execution_result=None,
            status=self.failed_status if failed else "completed",
            files_written=[file_name],
            steps_used=2,
            duration_seconds=0.5,
            usage={"input_tokens": 10, "output_tokens": 5},
            workspace=str(workspace),
            error="Step limit reached" if failed else None,
        )
