"""The PM -> Coder team: one LangGraph graph over the shared ``TeamState``.

The PM writes the spec and tasks, publishes one task at a time on the
``handoff`` key, and the Coder builds each task in one shared workspace and
reports back. See specs/001-autonomous-dev-team/contracts/handoff-protocol.md.

Usage:
    python -m orchestration.team "<requirement>" [--max-steps N] [--timeout SECONDS]
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agents.coder_agent import CoderAgent, CoderConfig, new_run_id, print_step_progress
from agents.pm_agent import MAX_TASKS, PMAgent, PMConfig, merge_tasks
from orchestration.state import (
    CodeOutput,
    Consolidation,
    Task,
    TaskProgress,
    TeamState,
    TechnicalSpec,
    TokenUsage,
)
from tools.file_tools import read_file
from tools.python_exec import SandboxUnavailableError

# pm + consolidate + (dispatch + coder) per task + finish, with room to spare.
RECURSION_LIMIT = 2 * MAX_TASKS + 5


def initial_team_state(requirement: str, run_id: str, workspace: Path) -> TeamState:
    """Return the state a run starts from.

    Raises:
        ValueError: if ``requirement`` is empty.
    """
    if not requirement.strip():
        raise ValueError("requirement must not be empty")
    return TeamState(
        run_id=run_id,
        workspace=str(workspace),
        requirement=requirement,
        spec=None,
        tasks=[],
        handoff=None,
        task_progress={},
        code_output=[],
        consolidation=None,
        status="running",
        error=None,
        usage=TokenUsage(input_tokens=0, output_tokens=0),
        duration_seconds=0.0,
    )


def add_usage(total: TokenUsage, extra: TokenUsage) -> TokenUsage:
    """Return ``total`` plus ``extra``."""
    return TokenUsage(
        input_tokens=total["input_tokens"] + extra["input_tokens"],
        output_tokens=total["output_tokens"] + extra["output_tokens"],
    )


def pending_progress(tasks: list[Task]) -> dict[str, TaskProgress]:
    """Return a fresh ``pending`` progress entry for every task."""
    return {
        task["task_id"]: TaskProgress(
            task_id=task["task_id"],
            status="pending",
            coder_status=None,
            summary="",
            files_written=[],
            steps_used=0,
            duration_seconds=0.0,
            error=None,
        )
        for task in tasks
    }


def pm_node(state: TeamState, pm: PMAgent) -> dict:
    """PM side: write the spec and task list into shared state."""
    result = pm.write_spec(state["requirement"])
    usage = add_usage(state["usage"], result.usage)
    if result.error:
        return {"status": "error", "error": result.error, "usage": usage}
    if result.rejection_reason:
        return {"status": "rejected", "error": result.rejection_reason, "usage": usage}
    return {
        "spec": result.spec,
        "tasks": result.spec["tasks"],
        "task_progress": pending_progress(result.spec["tasks"]),
        "usage": usage,
    }


def consolidate_node(state: TeamState, pm: PMAgent) -> dict:
    """PM side: bring a plan of more than ``MAX_TASKS`` tasks down to the limit.

    The PM's merge is used when it passed ``PMAgent.consolidate``'s checks;
    otherwise the original tasks are merged deterministically (FR-006a).
    """
    result = pm.consolidate(state["spec"], state["requirement"])
    if result.error:
        tasks, method = merge_tasks(state["tasks"]), "merge"
    else:
        tasks, method = result.spec["tasks"], "pm"
    return {
        "spec": {**state["spec"], "tasks": tasks},
        "tasks": tasks,
        "task_progress": pending_progress(tasks),
        "consolidation": Consolidation(
            proposed_count=len(state["tasks"]), final_count=len(tasks), method=method
        ),
        "usage": add_usage(state["usage"], result.usage),
    }


def route_after_pm(state: TeamState) -> str:
    """Stop on rejection or error, consolidate an oversized plan, else hand off."""
    if state["status"] != "running":
        return "finish"
    if len(state["tasks"]) > MAX_TASKS:
        return "consolidate"
    return "dispatch"


def dispatch_node(state: TeamState) -> dict:
    """PM side: publish the next pending task, in list order, on ``handoff``."""
    progress = state["task_progress"]
    task = next(t for t in state["tasks"] if progress[t["task_id"]]["status"] == "pending")
    started = TaskProgress(**progress[task["task_id"]])
    started["status"] = "in_progress"
    return {"handoff": Task(**task), "task_progress": {**progress, task["task_id"]: started}}


def render_coder_task(
    handoff: Task, spec: TechnicalSpec, position: int, total: int, existing_files: list[str]
) -> str:
    """Turn a handoff into the Coder's task text (layout: contracts/handoff-protocol.md)."""
    criteria = "\n".join(f"- {criterion}" for criterion in handoff["acceptance_criteria"])
    parts = [
        f"You are building: {spec['name']}. {spec['description']}",
        f"Task {handoff['task_id']} of {total} (priority: {handoff['priority']}): "
        f"{handoff['title']}\n{handoff['description']}",
        f"Acceptance criteria:\n{criteria}",
    ]
    if existing_files:
        parts.append(
            f"Files already in the workspace from earlier tasks: {', '.join(existing_files)}.\n"
            "Read a file before you change it, and keep earlier tasks' behavior working."
        )
    return "\n\n".join(parts)


def _workspace_files(workspace: Path) -> list[str]:
    """Workspace-relative paths of the regular files, skipping hidden names."""
    relative = (path.relative_to(workspace) for path in workspace.rglob("*") if path.is_file())
    return sorted(
        str(path) for path in relative if not any(part.startswith(".") for part in path.parts)
    )


def coder_node(state: TeamState, coder) -> dict:
    """Coder side: build the task on ``handoff`` and report progress and code back."""
    handoff, workspace = state["handoff"], Path(state["workspace"])
    position = [task["task_id"] for task in state["tasks"]].index(handoff["task_id"]) + 1
    text = render_coder_task(
        handoff, state["spec"], position, len(state["tasks"]), _workspace_files(workspace)
    )
    result = coder.run(text, workspace)
    progress = TaskProgress(
        task_id=handoff["task_id"],
        status="completed" if result.status == "completed" else "failed",
        coder_status=result.status,
        summary=result.explanation,
        files_written=list(result.files_written),
        steps_used=result.steps_used,
        duration_seconds=result.duration_seconds,
        error=result.error,
    )
    return {
        "handoff": None,
        "task_progress": {**state["task_progress"], handoff["task_id"]: progress},
        "code_output": state["code_output"] + _read_outputs(workspace, handoff, result),
        "usage": add_usage(state["usage"], TokenUsage(**result.usage)),
    }


def _read_outputs(workspace: Path, handoff: Task, result) -> list[CodeOutput]:
    """Read each file the task wrote from disk (ground truth, not the model's claim)."""
    outputs = []
    for path in result.files_written:
        content, missing = read_file(workspace, path)
        if not missing:
            outputs.append(CodeOutput(task_id=handoff["task_id"], path=path, content=content))
    return outputs


def route_after_coder(state: TeamState) -> str:
    """Hand off the next task unless one failed or none are left (FR-009a)."""
    statuses = [progress["status"] for progress in state["task_progress"].values()]
    if "failed" in statuses or "pending" not in statuses:
        return "finish"
    return "dispatch"


def finish_node(state: TeamState) -> dict:
    """Set the run's final status; after a failure, mark the remaining tasks skipped."""
    if state["status"] in ("rejected", "error"):
        return {"handoff": None, "status": state["status"]}
    progress = {
        task_id: TaskProgress(**{**entry, "status": "skipped"})
        if entry["status"] == "pending"
        else entry
        for task_id, entry in state["task_progress"].items()
    }
    in_order = [progress[task["task_id"]] for task in state["tasks"]]
    failed = [entry for entry in in_order if entry["status"] == "failed"]
    if not failed:
        return {"handoff": None, "task_progress": progress, "status": "completed"}
    first = failed[0]
    reason = first["error"] or first["summary"]
    error = f"Task {first['task_id']} failed ({first['coder_status']}): {reason}"
    return {"handoff": None, "task_progress": progress, "status": "failed", "error": error}


def build_team_graph(pm: PMAgent, coder):
    """Compile the team graph: pm -> (consolidate) -> dispatch <-> coder -> finish."""
    graph = StateGraph(TeamState)
    graph.add_node("pm", partial(pm_node, pm=pm))
    graph.add_node("consolidate", partial(consolidate_node, pm=pm))
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("coder", partial(coder_node, coder=coder))
    graph.add_node("finish", finish_node)
    graph.add_edge(START, "pm")
    graph.add_conditional_edges("pm", route_after_pm, ["finish", "consolidate", "dispatch"])
    graph.add_edge("consolidate", "dispatch")
    graph.add_edge("dispatch", "coder")
    graph.add_conditional_edges("coder", route_after_coder, ["dispatch", "finish"])
    graph.add_edge("finish", END)
    return graph.compile()


def run_team(
    requirement: str,
    pm: PMAgent | None = None,
    coder=None,
    workspace_root: Path = Path("workspace"),
    on_update: Callable[[str, dict], None] | None = None,
) -> TeamState:
    """Run the PM -> Coder team on ``requirement`` and return the final state.

    ``coder`` is any object with ``run(task, workspace) -> CoderResult``.
    ``on_update(node, update)`` is called after every node.

    Raises:
        ValueError: if ``requirement`` is empty.
        SandboxUnavailableError: if no ``coder`` is given and Docker is unavailable.
    """
    started = time.monotonic()
    run_id = new_run_id()
    workspace = (workspace_root / run_id).resolve()
    state = initial_team_state(requirement, run_id, workspace)
    graph = build_team_graph(pm or PMAgent(), coder or CoderAgent())
    workspace.mkdir(parents=True)
    final = state
    for mode, chunk in graph.stream(
        state, {"recursion_limit": RECURSION_LIMIT}, stream_mode=["updates", "values"]
    ):
        if mode == "values":
            final = chunk
        elif on_update:
            for node, update in chunk.items():
                on_update(node, update)
    # Only the caller knows when the whole run ended, so this field is set outside a node.
    return {**final, "duration_seconds": round(time.monotonic() - started, 3)}


def format_update(node: str, update: dict, current: Task | None) -> str | None:
    """Return the progress line for one node's update, or ``None`` if there is none.

    ``current`` is the task most recently handed off, which the ``coder`` line
    reports on (the Coder's update holds every task's progress).
    """
    if node == "pm":
        if update.get("spec"):
            count = len(update["spec"]["tasks"])
            return f'[pm] "{update["spec"]["name"]}": {count} task{"" if count == 1 else "s"}'
        return f"[pm] {update['status']}: {update['error']}"
    if node == "consolidate":
        merge = update["consolidation"]
        return (
            f"[pm] consolidated {merge['proposed_count']} → {merge['final_count']} tasks "
            f"({merge['method']})"
        )
    if node == "dispatch":
        handoff = update["handoff"]
        return f"[handoff] {handoff['task_id']} ({handoff['priority']}) {handoff['title']}"
    if node == "coder" and current:
        progress = update["task_progress"][current["task_id"]]
        if progress["status"] == "completed":
            return (
                f"[coder] {current['task_id']} completed in {progress['steps_used']} steps "
                f"({progress['duration_seconds']} s)"
            )
        return f"[coder] {current['task_id']} failed ({progress['coder_status']})"
    if node == "finish":
        return f"[finish] {update['status']}"
    return None


def _progress_printer() -> Callable[[str, dict], None]:
    """Return an ``on_update`` callback that prints progress lines to stderr."""
    current: Task | None = None

    def on_update(node: str, update: dict) -> None:
        nonlocal current
        if node == "dispatch":
            current = update["handoff"]
        line = format_update(node, update, current)
        if line:
            print(line, file=sys.stderr, flush=True)

    return on_update


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns 0 if the run completed, 1 otherwise, 2 for usage errors."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="python -m orchestration.team",
        description="Turn a plain-English requirement into a spec, tasks, and working code.",
    )
    parser.add_argument("requirement", help="what the software should do")
    parser.add_argument("--max-steps", type=int, help="Coder step cap per task (1-10)")
    parser.add_argument("--timeout", type=float, help="seconds before a code run is killed")
    args = parser.parse_args(argv)

    try:
        config = CoderConfig.from_env()
        if args.max_steps is not None:
            config = replace(config, max_steps=args.max_steps)
        if args.timeout is not None:
            config = replace(config, exec_timeout_seconds=args.timeout)
        pm = PMAgent(PMConfig.from_env())
        coder = CoderAgent(config, on_step=print_step_progress)
        final = run_team(
            args.requirement, pm, coder, config.workspace_root, on_update=_progress_printer()
        )
    except (ValueError, SandboxUnavailableError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(final, indent=2))
    return 0 if final["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
