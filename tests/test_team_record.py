"""Offline tests for the team run record, progress lines, and CLI (contracts/team-api.md)."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.pm_agent import PMAgent, PMConfig
from orchestration import team
from orchestration.team import format_update, run_team
from tests.team_fakes import FakeCoder, FakePMClient, pm_response, regroup_payload, spec_payload
from tools.python_exec import SandboxUnavailableError

REQUIREMENT = "Build a temperature converter CLI"


def make_pm(*responses) -> PMAgent:
    return PMAgent(PMConfig(), client=FakePMClient(list(responses)))


class RunRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def run_team(self, coder: FakeCoder, task_count: int = 3, on_update=None) -> dict:
        pm = make_pm(pm_response(spec_payload(task_count)))
        return run_team(REQUIREMENT, pm, coder, self.root, on_update)

    def test_progress_is_reported_after_every_node(self) -> None:
        updates: list[tuple[str, dict]] = []

        final = self.run_team(FakeCoder(), on_update=lambda node, u: updates.append((node, u)))

        nodes = [node for node, _ in updates]
        self.assertEqual(
            nodes, ["pm", "dispatch", "coder", "dispatch", "coder", "dispatch", "coder", "finish"]
        )
        coder_updates = [update for node, update in updates if node == "coder"]
        for task_id, update in zip(["T1", "T2", "T3"], coder_updates):
            self.assertEqual(update["task_progress"][task_id], final["task_progress"][task_id])

    def test_record_keeps_assumptions_priorities_and_usage(self) -> None:
        final = self.run_team(FakeCoder())

        payload = spec_payload(3)
        self.assertEqual(final["spec"]["assumptions"], payload["assumptions"])
        self.assertEqual(
            [task["priority"] for task in final["tasks"]],
            [task["priority"] for task in payload["tasks"]],
        )
        self.assertEqual(final["usage"], {"input_tokens": 130, "output_tokens": 65})
        self.assertEqual(Path(final["workspace"]).parent, self.root.resolve())
        self.assertEqual(json.loads(json.dumps(final)), final)

    def test_failed_run_names_the_failed_task(self) -> None:
        final = self.run_team(FakeCoder(fail_on=frozenset({"T2"})))

        self.assertEqual(final["status"], "failed")
        self.assertTrue(final["error"].startswith("Task T2 failed (step_limit)"))

    def test_durations_are_recorded(self) -> None:
        final = self.run_team(FakeCoder(delay_seconds=0.01))

        self.assertIsInstance(final["duration_seconds"], float)
        self.assertGreater(final["duration_seconds"], 0)
        for progress in final["task_progress"].values():
            self.assertEqual(progress["duration_seconds"], 0.5)


class FormatUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        pm = make_pm(pm_response(spec_payload(11)))
        self.spec = pm.write_spec(REQUIREMENT).spec
        self.task = self.spec["tasks"][1]

    def test_pm_and_consolidate_lines(self) -> None:
        three = {**self.spec, "tasks": self.spec["tasks"][:3]}
        self.assertEqual(
            format_update("pm", {"spec": three}, None), '[pm] "Temperature converter": 3 tasks'
        )
        self.assertEqual(
            format_update("pm", {"status": "rejected", "error": "Not software."}, None),
            "[pm] rejected: Not software.",
        )
        self.assertEqual(
            format_update("pm", {"status": "error", "error": "Model call failed"}, None),
            "[pm] error: Model call failed",
        )
        merged = regroup_payload(self.spec["tasks"], 6)["tasks"]
        consolidated = {
            "consolidation": {"proposed_count": 11, "final_count": 6, "method": "pm"},
            "tasks": merged,
        }
        self.assertEqual(
            format_update("consolidate", consolidated, None), "[pm] consolidated 11 → 6 tasks (pm)"
        )

    def test_handoff_and_coder_lines(self) -> None:
        self.assertEqual(
            format_update("dispatch", {"handoff": self.task}, None), "[handoff] T2 (medium) Task 2"
        )
        done = {
            "task_id": "T2",
            "status": "completed",
            "coder_status": "completed",
            "steps_used": 2,
            "duration_seconds": 0.5,
        }
        failed = {**done, "status": "failed", "coder_status": "step_limit"}
        self.assertEqual(
            format_update("coder", {"task_progress": {"T2": done}}, self.task),
            "[coder] T2 completed in 2 steps (0.5 s)",
        )
        self.assertEqual(
            format_update("coder", {"task_progress": {"T2": failed}}, self.task),
            "[coder] T2 failed (step_limit)",
        )

    def test_finish_line_and_silent_updates(self) -> None:
        for status in ("completed", "failed", "rejected", "error"):
            self.assertEqual(
                format_update("finish", {"status": status, "handoff": None}, None),
                f"[finish] {status}",
            )
        self.assertIsNone(format_update("unknown", {}, None))


class CliTests(unittest.TestCase):
    def run_main(self, argv: list[str], final: dict | None = None, coder_error=None) -> tuple:
        stdout, stderr = io.StringIO(), io.StringIO()
        coder_agent = mock.Mock(side_effect=coder_error)
        patches = [
            mock.patch.object(team, "PMAgent"),
            mock.patch.object(team, "CoderAgent", coder_agent),
        ]
        if final is not None:
            patches.append(mock.patch.object(team, "run_team", return_value=final))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            code = team.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_exit_codes_follow_the_run_status(self) -> None:
        for status, expected in [("completed", 0), ("failed", 1), ("rejected", 1), ("error", 1)]:
            with self.subTest(status):
                code, out, _ = self.run_main([REQUIREMENT], final={"status": status})
                self.assertEqual(code, expected)
                self.assertEqual(json.loads(out), {"status": status})

    def test_usage_errors_exit_2(self) -> None:
        code, _, err = self.run_main(["  "])
        self.assertEqual(code, 2)
        self.assertIn("error:", err)

        code, _, err = self.run_main(
            [REQUIREMENT], coder_error=SandboxUnavailableError("Docker is not available")
        )
        self.assertEqual(code, 2)
        self.assertIn("error: Docker is not available", err)


if __name__ == "__main__":
    unittest.main()
