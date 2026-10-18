"""Offline tests for the PM -> Coder handoff over shared state (contracts/handoff-protocol.md)."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import anthropic
import httpx2

from agents.pm_agent import PMAgent, PMConfig
from orchestration.team import (
    RECURSION_LIMIT,
    build_team_graph,
    coder_node,
    dispatch_node,
    initial_team_state,
    pm_node,
    render_coder_task,
    route_after_pm,
    run_team,
)
from tests.team_fakes import (
    FakeCoder,
    FakePMClient,
    pm_response,
    regroup_payload,
    spec_payload,
    tasks_payload,
)

REQUIREMENT = "Build a temperature converter CLI"
# Copied from contracts/task-handoff.schema.json; specs/ is not in git.
HANDOFF_KEYS = {"task_id", "title", "description", "acceptance_criteria", "priority"}
PRIORITIES = ("high", "medium", "low")


def make_pm(*responses) -> PMAgent:
    return PMAgent(PMConfig(), client=FakePMClient(list(responses)))


def api_error() -> anthropic.APIConnectionError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(request=request)


def flat_criteria(tasks: list[dict]) -> list[str]:
    return [criterion for task in tasks for criterion in task["acceptance_criteria"]]


class TeamTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name) / "run"
        self.workspace.mkdir()
        self.state = initial_team_state(REQUIREMENT, "test-run", self.workspace)

    def run_graph(self, pm: PMAgent, coder: FakeCoder) -> dict:
        graph = build_team_graph(pm, coder)
        return graph.invoke(self.state, {"recursion_limit": RECURSION_LIMIT})

    def after_pm(self, task_count: int = 3) -> dict:
        update = pm_node(self.state, make_pm(pm_response(spec_payload(task_count))))
        return {**self.state, **update}


class PmNodeHandoffTests(TeamTestCase):
    """The user's required test: the PM node leaves a schema-correct task list in state."""

    def test_pm_node_writes_a_non_empty_schema_correct_task_list(self) -> None:
        before = copy.deepcopy(self.state)
        after = self.after_pm(3)

        self.assertIsInstance(after["tasks"], list)
        self.assertTrue(after["tasks"])
        for task in after["tasks"]:
            self.assertEqual(set(task), HANDOFF_KEYS)
            for field in ("task_id", "title", "description"):
                self.assertIsInstance(task[field], str)
                self.assertTrue(task[field].strip())
            self.assertIsInstance(task["acceptance_criteria"], list)
            self.assertTrue(task["acceptance_criteria"])
            self.assertTrue(all(isinstance(c, str) for c in task["acceptance_criteria"]))
            self.assertIn(task["priority"], PRIORITIES)
        self.assertEqual([task["task_id"] for task in after["tasks"]], ["T1", "T2", "T3"])
        self.assertEqual(after["tasks"], after["spec"]["tasks"])
        for field in ("name", "description", "acceptance_criteria"):
            self.assertTrue(after["spec"][field])
        self.assertEqual(set(after["task_progress"]), {"T1", "T2", "T3"})
        self.assertTrue(all(p["status"] == "pending" for p in after["task_progress"].values()))
        self.assertIsNone(after["handoff"])
        self.assertEqual(after["status"], "running")
        self.assertEqual(self.state, before)

    def test_route_after_pm(self) -> None:
        self.assertEqual(route_after_pm(self.after_pm(3)), "dispatch")
        self.assertEqual(route_after_pm(self.after_pm(9)), "consolidate")
        for status in ("rejected", "error"):
            self.assertEqual(route_after_pm({**self.state, "status": status}), "finish")


class DispatchAndCoderNodeTests(TeamTestCase):
    def test_dispatch_publishes_the_first_pending_task(self) -> None:
        after_pm = self.after_pm(3)

        update = dispatch_node(after_pm)

        handoff = update["handoff"]
        self.assertEqual(handoff, after_pm["tasks"][0])
        self.assertEqual(set(handoff), HANDOFF_KEYS)
        self.assertEqual(json.loads(json.dumps(handoff)), handoff)
        statuses = {tid: p["status"] for tid, p in update["task_progress"].items()}
        self.assertEqual(statuses, {"T1": "in_progress", "T2": "pending", "T3": "pending"})
        self.assertEqual(after_pm["task_progress"]["T1"]["status"], "pending")

    def test_coder_node_reports_progress_and_clears_the_handoff(self) -> None:
        state = self.after_pm(3)
        state = {**state, **dispatch_node(state)}
        coder = FakeCoder()

        update = coder_node(state, coder)

        self.assertIsNone(update["handoff"])
        progress = update["task_progress"]["T1"]
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(progress["coder_status"], "completed")
        self.assertEqual(progress["summary"], "Built T1")
        self.assertEqual(progress["files_written"], ["t1.py"])
        self.assertEqual(progress["steps_used"], 2)
        self.assertEqual(progress["duration_seconds"], 0.5)
        self.assertEqual(
            update["code_output"], [{"task_id": "T1", "path": "t1.py", "content": "print('T1')\n"}]
        )
        self.assertEqual(update["usage"], {"input_tokens": 110, "output_tokens": 55})

        task_id, task_text, workspace = coder.calls[0]
        task = state["tasks"][0]
        self.assertEqual(task_id, "T1")
        self.assertIn("Task T1 of 3", task_text)
        self.assertIn(f"priority: {task['priority']}", task_text)
        for criterion in task["acceptance_criteria"]:
            self.assertIn(criterion, task_text)
        self.assertIn(state["spec"]["name"], task_text)
        self.assertEqual(workspace, self.workspace)

    def test_render_coder_task_lists_existing_files_only_when_there_are_some(self) -> None:
        state = self.after_pm(2)
        task = state["tasks"][1]

        without = render_coder_task(task, state["spec"], 2, 2, [])
        with_files = render_coder_task(task, state["spec"], 2, 2, ["a.py", "b.py"])

        self.assertNotIn("Files already in the workspace", without)
        self.assertIn("Files already in the workspace from earlier tasks: a.py, b.py.", with_files)
        self.assertIn("Task T2 of 2 (priority: medium): Task 2", without)


class FullGraphTests(TeamTestCase):
    def test_tasks_run_in_order_in_one_workspace(self) -> None:
        coder = FakeCoder()

        final = self.run_graph(make_pm(pm_response(spec_payload(3))), coder)

        self.assertEqual(final["status"], "completed")
        self.assertEqual([call[0] for call in coder.calls], ["T1", "T2", "T3"])
        self.assertEqual({call[2] for call in coder.calls}, {self.workspace})
        self.assertTrue(all(p["status"] == "completed" for p in final["task_progress"].values()))
        self.assertEqual(len(final["code_output"]), 3)
        self.assertIsNone(final["handoff"])
        self.assertEqual(json.loads(json.dumps(final)), final)

    def test_one_task_in_flight_and_completed_in_order(self) -> None:
        graph = build_team_graph(make_pm(pm_response(spec_payload(3))), FakeCoder())
        completion_order: list[str] = []

        for state in graph.stream(
            self.state, {"recursion_limit": RECURSION_LIMIT}, stream_mode="values"
        ):
            in_progress = [
                tid for tid, p in state["task_progress"].items() if p["status"] == "in_progress"
            ]
            if state["handoff"] is not None:
                self.assertEqual(in_progress, [state["handoff"]["task_id"]])
            else:
                self.assertEqual(in_progress, [])
            for tid, progress in state["task_progress"].items():
                if progress["status"] == "completed" and tid not in completion_order:
                    completion_order.append(tid)
            self.assertEqual(json.loads(json.dumps(state)), state)

        self.assertEqual(completion_order, ["T1", "T2", "T3"])


class TaskCapTests(TeamTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.original = spec_payload(11)["tasks"]

    def assert_capped(self, final: dict, coder: FakeCoder, method: str, count: int) -> None:
        self.assertEqual(final["status"], "completed")
        self.assertEqual(len(final["tasks"]), count)
        self.assertEqual(final["tasks"], final["spec"]["tasks"])
        self.assertEqual(
            final["consolidation"], {"proposed_count": 11, "final_count": count, "method": method}
        )
        self.assertEqual(len(coder.calls), count)
        self.assertFalse(any(" of 11" in call[1] for call in coder.calls))
        # FR-006a: every original criterion is still there, in order.
        self.assertEqual(flat_criteria(final["tasks"]), flat_criteria(self.original))

    def test_pm_consolidation_is_used_when_it_passes_the_checks(self) -> None:
        coder = FakeCoder()
        pm = make_pm(
            pm_response(spec_payload(11)), pm_response(regroup_payload(self.original, 6))
        )

        final = self.run_graph(pm, coder)

        self.assert_capped(final, coder, "pm", 6)
        self.assertEqual(final["usage"], {"input_tokens": 260, "output_tokens": 130})

    def test_failed_consolidation_falls_back_to_merging_the_original_tasks(self) -> None:
        dropped = copy.deepcopy(regroup_payload(self.original, 6))
        dropped["tasks"][0]["acceptance_criteria"].pop()
        failures = {
            "too many tasks": pm_response(tasks_payload(10)),
            "api error": api_error(),
            "empty task list": pm_response({"tasks": []}),
            "dropped criterion": pm_response(dropped),
        }
        for name, reply in failures.items():
            with self.subTest(name):
                coder = FakeCoder()
                pm = make_pm(pm_response(spec_payload(11)), reply)

                final = self.run_graph(pm, coder)

                self.assert_capped(final, coder, "merge", 8)


class FailureAndRejectionTests(TeamTestCase):
    def test_run_stops_at_the_first_failed_task(self) -> None:
        coder = FakeCoder(fail_on=frozenset({"T2"}))

        final = self.run_graph(make_pm(pm_response(spec_payload(4))), coder)

        self.assertEqual(final["status"], "failed")
        statuses = {tid: p["status"] for tid, p in final["task_progress"].items()}
        self.assertEqual(
            statuses, {"T1": "completed", "T2": "failed", "T3": "skipped", "T4": "skipped"}
        )
        self.assertEqual(final["task_progress"]["T2"]["coder_status"], "step_limit")
        self.assertEqual([call[0] for call in coder.calls], ["T1", "T2"])
        self.assertTrue((self.workspace / "t1.py").is_file())
        self.assertTrue((self.workspace / "t2.py").is_file())
        self.assertIsNone(final["handoff"])

    def test_rejected_requirement_never_reaches_the_coder(self) -> None:
        coder = FakeCoder()
        pm = make_pm(pm_response(spec_payload(0, rejection_reason="Not a software request.")))

        final = self.run_graph(pm, coder)

        self.assertEqual(final["status"], "rejected")
        self.assertEqual(final["error"], "Not a software request.")
        self.assertEqual(final["tasks"], [])
        self.assertEqual(coder.calls, [])

    def test_pm_error_never_reaches_the_coder(self) -> None:
        coder = FakeCoder()

        final = self.run_graph(make_pm(api_error()), coder)

        self.assertEqual(final["status"], "error")
        self.assertTrue(final["error"])
        self.assertEqual(coder.calls, [])

    def test_empty_requirement_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            initial_team_state("  ", "test-run", self.workspace)
        with self.assertRaises(ValueError):
            run_team("  ", pm=make_pm(), coder=FakeCoder(), workspace_root=self.workspace)


if __name__ == "__main__":
    unittest.main()
