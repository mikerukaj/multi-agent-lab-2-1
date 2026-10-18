"""Shared LangGraph state that the PM and Coder agents read and write.

Every value is a ``TypedDict``, a ``Literal`` or a plain JSON type, so any state
converts to JSON with ``json.dumps`` and back without loss
(specs/001-autonomous-dev-team/data-model.md, "Increment 2: Shared Team State").
"""

from typing import Literal, TypedDict

Priority = Literal["high", "medium", "low"]
TaskStatus = Literal["pending", "in_progress", "completed", "failed", "skipped"]
RunStatus = Literal["running", "completed", "failed", "rejected", "error"]
ConsolidationMethod = Literal["pm", "merge"]


class Task(TypedDict):
    """One engineering task; also the handoff message (contracts/handoff-protocol.md)."""

    task_id: str
    title: str
    description: str
    acceptance_criteria: list[str]
    priority: Priority


class TechnicalSpec(TypedDict):
    """The PM's specification of the requirement, including its ordered tasks."""

    name: str
    description: str
    acceptance_criteria: list[str]
    assumptions: list[str]
    tasks: list[Task]


class TaskProgress(TypedDict):
    """Status of one task, written by the PM side and then by the Coder."""

    task_id: str
    status: TaskStatus
    coder_status: str | None
    summary: str
    files_written: list[str]
    steps_used: int
    duration_seconds: float
    error: str | None


class CodeOutput(TypedDict):
    """One file a task wrote, with its contents read from disk after the task."""

    task_id: str
    path: str
    content: str


class Consolidation(TypedDict):
    """How a plan with more than the task limit was merged down."""

    proposed_count: int
    final_count: int
    method: ConsolidationMethod


class TokenUsage(TypedDict):
    """Tokens used by model calls, summed over the run."""

    input_tokens: int
    output_tokens: int


class TeamState(TypedDict):
    """The state of one team run, shared by every node of the team graph."""

    run_id: str
    workspace: str
    requirement: str
    spec: TechnicalSpec | None
    tasks: list[Task]
    handoff: Task | None
    task_progress: dict[str, TaskProgress]
    code_output: list[CodeOutput]
    consolidation: Consolidation | None
    status: RunStatus
    error: str | None
    usage: TokenUsage
    duration_seconds: float
