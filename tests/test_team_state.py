"""Offline tests for the shared team state schema (orchestration/state.py)."""

import json
import typing
import unittest

from orchestration.state import Priority, Task, TaskProgress, TeamState

# Copied from contracts/task-handoff.schema.json ("required"); specs/ is not in git.
HANDOFF_KEYS = {"task_id", "title", "description", "acceptance_criteria", "priority"}

TEAM_STATE_KEYS = {
    "run_id",
    "workspace",
    "requirement",
    "spec",
    "tasks",
    "handoff",
    "task_progress",
    "code_output",
    "consolidation",
    "status",
    "error",
    "usage",
    "duration_seconds",
}

TASK_PROGRESS_KEYS = {
    "task_id",
    "status",
    "coder_status",
    "summary",
    "files_written",
    "steps_used",
    "duration_seconds",
    "error",
}


def sample_state() -> TeamState:
    task = Task(
        task_id="T1",
        title="Convert temperatures",
        description="Write convert.py.",
        acceptance_criteria=["convert.py 100 C prints 212.0 F"],
        priority="high",
    )
    return TeamState(
        run_id="20261017-120000-abcdef",
        workspace="/tmp/workspace/20261017-120000-abcdef",
        requirement="A temperature converter",
        spec={
            "name": "Temperature converter",
            "description": "Converts between Celsius and Fahrenheit.",
            "acceptance_criteria": ["Both directions work"],
            "assumptions": ["Input is a number and a unit letter"],
            "tasks": [task],
        },
        tasks=[task],
        handoff=None,
        task_progress={
            "T1": {
                "task_id": "T1",
                "status": "completed",
                "coder_status": "completed",
                "summary": "Wrote convert.py.",
                "files_written": ["convert.py"],
                "steps_used": 3,
                "duration_seconds": 12.5,
                "error": None,
            }
        },
        code_output=[{"task_id": "T1", "path": "convert.py", "content": "print(1)\n"}],
        consolidation={"proposed_count": 9, "final_count": 1, "method": "merge"},
        status="completed",
        error=None,
        usage={"input_tokens": 1200, "output_tokens": 300},
        duration_seconds=20.0,
    )


class TeamStateSchemaTests(unittest.TestCase):
    def test_task_has_exactly_the_handoff_contract_keys(self) -> None:
        self.assertEqual(Task.__required_keys__, HANDOFF_KEYS)

    def test_priority_values_match_the_contract(self) -> None:
        self.assertEqual(typing.get_args(Priority), ("high", "medium", "low"))

    def test_team_state_and_task_progress_fields(self) -> None:
        self.assertEqual(TeamState.__required_keys__, TEAM_STATE_KEYS)
        self.assertEqual(TaskProgress.__required_keys__, TASK_PROGRESS_KEYS)

    def test_state_round_trips_through_json(self) -> None:
        state = sample_state()
        self.assertEqual(json.loads(json.dumps(state)), state)


if __name__ == "__main__":
    unittest.main()
