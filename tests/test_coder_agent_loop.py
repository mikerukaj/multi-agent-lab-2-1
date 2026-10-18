"""Offline tests for the Coder Agent's ReAct loop, using a scripted fake client."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import anthropic
import httpx2

from agents import coder_agent
from agents.coder_agent import CoderAgent, CoderConfig, CoderResult, Step, ToolCall
from tools.python_exec import SandboxUnavailableError

SUMMARY = {
    "plan": ["Write main.py", "Run main.py"],
    "explanation": "Wrote main.py, which prints hi.",
    "main_file": "main.py",
}


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def thinking_block(summary: str) -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=summary, signature="sig")


def tool_use(block_id: str, name: str, tool_input: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def response(content: list, stop_reason: str, stop_details=None) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class FakeClient:
    """Stands in for ``anthropic.Anthropic``; replays scripted loop responses.

    Requests carrying ``output_config.format`` are the finalize call and get
    ``summary`` back. ``loop`` is a list of responses or exceptions, or a
    callable returning the next one.
    """

    def __init__(self, loop, summary: dict = SUMMARY, events: list | None = None) -> None:
        self._loop = iter(loop) if isinstance(loop, list) else None
        self._loop_fn = loop if callable(loop) else None
        self._summary = summary
        self.events = events if events is not None else []
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        self.events.append("create")
        if "format" in kwargs["output_config"]:
            return response([text_block(json.dumps(self._summary))], "end_turn")
        item = self._loop_fn() if self._loop_fn else next(self._loop)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def loop_requests(self) -> list[dict]:
        return [r for r in self.requests if "format" not in r["output_config"]]


class FakeMemory:
    """Records calls instead of embedding with Chroma."""

    events: list = []
    records: list = []

    def __init__(self, path: Path, collection: str = "coder_memory") -> None:
        pass

    def add(self, run_id: str, kind: str, text: str) -> str:
        FakeMemory.records.append((run_id, kind, text))
        return f"{run_id}-{len(FakeMemory.records)}"

    def query(self, text: str, top_k: int, exclude_run_id: str) -> list:
        FakeMemory.events.append("query")
        return [("old-1", "The user's name is Priya.")]


class AgentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.config = CoderConfig(
            workspace_root=root / "workspace", chroma_path=root / "chroma", max_steps=3
        )
        FakeMemory.events = []
        FakeMemory.records = []
        patcher = mock.patch.object(coder_agent, "SemanticMemory", FakeMemory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make_agent(self, client: FakeClient, **kwargs) -> CoderAgent:
        return CoderAgent(self.config, client=client, **kwargs)


def happy_path_script() -> list:
    return [
        response(
            [
                thinking_block("I should write the file first."),
                tool_use("t1", "write_file", {"path": "main.py", "content": "print('hi')\n"}),
            ],
            "tool_use",
        ),
        response(
            [tool_use("t2", "exec_python", {"path": "main.py", "purpose": "Check the output."})],
            "tool_use",
        ),
        response([text_block("Done.")], "end_turn"),
    ]


class HappyPathTests(AgentTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.client = FakeClient(happy_path_script(), events=FakeMemory.events)
        self.steps_seen: list[Step] = []
        self.result = self.make_agent(self.client, on_step=self.steps_seen.append).run(
            "Write main.py that prints hi"
        )

    def test_completes_with_ground_truth_code_and_execution(self) -> None:
        result = self.result
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.main_file, "main.py")
        workspace = Path(result.workspace)
        self.assertEqual(result.code, (workspace / "main.py").read_text(encoding="utf-8"))
        self.assertEqual(result.execution_result.exit_code, 0)
        self.assertEqual(result.execution_result.stdout, "hi\n")
        self.assertEqual(result.files_written, ["main.py"])
        self.assertEqual(result.plan, SUMMARY["plan"])
        self.assertEqual(result.explanation, SUMMARY["explanation"])
        self.assertEqual(result.steps_used, 3)
        self.assertIsNone(result.error)

    def test_required_fields_have_contract_types(self) -> None:
        data = self.result.to_dict()
        self.assertIsInstance(data["code"], str)
        self.assertIsInstance(data["explanation"], str)
        self.assertIsInstance(data["plan"], list)
        self.assertIsInstance(data["execution_result"], dict)
        json.dumps(data)

    def test_memory_is_queried_before_every_loop_model_call(self) -> None:
        loop_events = [e for e in FakeMemory.events]
        # The finalize call is the last "create"; every earlier one follows a query.
        creates = [i for i, e in enumerate(loop_events) if e == "create"][:-1]
        self.assertEqual(len(creates), 3)
        for index in creates:
            self.assertEqual(loop_events[index - 1], "query")

    def test_memories_are_injected_once_as_system_messages(self) -> None:
        messages = self.client.requests[-1]["messages"]
        system_messages = [m for m in messages if m["role"] == "system"]
        self.assertEqual(len(system_messages), 1)
        self.assertIn("Priya", system_messages[0]["content"])

    def test_history_is_append_only(self) -> None:
        first = self.client.requests[0]
        for previous, current in zip(self.client.requests, self.client.requests[1:]):
            self.assertEqual(current["system"], first["system"])
            self.assertEqual(current["tools"], first["tools"])
            prefix = current["messages"][: len(previous["messages"])]
            self.assertEqual(prefix, previous["messages"])

    def test_usage_and_duration_are_recorded(self) -> None:
        self.assertEqual(self.result.usage, {"input_tokens": 40, "output_tokens": 20})
        self.assertGreater(self.result.duration_seconds, 0)

    def test_requests_use_research_settings(self) -> None:
        request = self.client.requests[0]
        self.assertEqual(request["model"], "claude-sonnet-5-5")
        self.assertEqual(request["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(request["output_config"], {"effort": "medium"})
        self.assertEqual(request["fallbacks"], "default")
        self.assertEqual(request["betas"], ["server-side-fallback-2026-07-01"])

    def test_trace_records_each_step(self) -> None:
        steps = self.result.steps
        self.assertEqual([s.number for s in steps], [1, 2, 3])
        self.assertIn("write the file first", steps[0].thought)
        self.assertEqual(steps[1].thought, "Check the output.")  # purpose fallback
        self.assertEqual(steps[2].thought, "Done.")
        (write,) = steps[0].actions
        self.assertEqual(
            write,
            ToolCall(
                tool="write_file",
                input={"path": "main.py", "content": "print('hi')\n"},
                output="Wrote 12 bytes to main.py",
                is_error=False,
            ),
        )
        (run,) = steps[1].actions
        self.assertEqual(json.loads(run.output)["stdout"], "hi\n")
        self.assertEqual(steps[2].actions, [])

    def test_on_step_receives_steps_with_actions_in_order(self) -> None:
        self.assertEqual([s.number for s in self.steps_seen], [1, 2])
        self.assertEqual(self.steps_seen[0].actions[0].tool, "write_file")

    def test_task_and_result_are_stored_in_memory(self) -> None:
        kinds = [kind for _, kind, _ in FakeMemory.records]
        self.assertEqual(kinds[0], "task")
        self.assertIn("observation", kinds)
        self.assertEqual(kinds[-1], "result")


class LimitAndFailureTests(AgentTestCase):
    def test_step_limit_ends_the_run_with_a_full_result(self) -> None:
        counter = iter(range(1000))

        def always_tool():
            n = next(counter)
            return response(
                [tool_use(f"t{n}", "write_file", {"path": "main.py", "content": f"print({n})\n"})],
                "tool_use",
            )

        client = FakeClient(always_tool)
        result = self.make_agent(client).run("Loop forever")
        self.assertEqual(result.status, "step_limit")
        self.assertEqual(len(client.loop_requests), 3)
        self.assertEqual(result.steps_used, 3)
        self.assertEqual(result.execution_result.stdout, "1\n")
        # The finalize request answers the pending tool call before asking for the summary.
        closing = client.requests[-1]["messages"][-1]["content"]
        self.assertEqual(closing[0]["type"], "tool_result")
        self.assertEqual(closing[0]["tool_use_id"], "t2")
        self.assertTrue(closing[0]["is_error"])

    def test_refusal_stops_without_a_summary_call(self) -> None:
        details = SimpleNamespace(category="cyber", explanation="Declined.")
        client = FakeClient([response([], "refusal", stop_details=details)])
        result = self.make_agent(client).run("Something declined")
        self.assertEqual(result.status, "refused")
        self.assertIn("cyber", result.error)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(result.code, "")
        self.assertIsNone(result.execution_result)

    def test_api_error_is_reported_not_raised(self) -> None:
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        client = FakeClient([anthropic.APIConnectionError(request=request)])
        result = self.make_agent(client).run("Anything")
        self.assertEqual(result.status, "error")
        self.assertIn("Connection error", result.error)

    def test_empty_task_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.make_agent(FakeClient([])).run("   ")

    def test_finalize_runs_main_file_when_the_model_did_not(self) -> None:
        client = FakeClient(
            [
                response(
                    [tool_use("t1", "write_file", {"path": "main.py", "content": "print(42)\n"})],
                    "tool_use",
                ),
                response([text_block("Wrote it.")], "end_turn"),
            ]
        )
        result = self.make_agent(client).run("Print 42")
        self.assertEqual(result.execution_result.stdout, "42\n")


class SandboxRequiredTests(AgentTestCase):
    def test_agent_refuses_to_start_without_docker(self) -> None:
        with mock.patch("tools.python_exec.shutil.which", return_value=None):
            with self.assertRaises(SandboxUnavailableError):
                self.make_agent(FakeClient([]))

    def test_cli_exits_2_without_docker(self) -> None:
        stderr = io.StringIO()
        with mock.patch("tools.python_exec.shutil.which", return_value=None), \
                contextlib.redirect_stderr(stderr):
            self.assertEqual(coder_agent.main(["Write a.py"]), 2)
        self.assertIn("Docker is required", stderr.getvalue())


class ConfigTests(unittest.TestCase):
    def test_env_overrides_and_validation(self) -> None:
        env = {"CODER_MAX_STEPS": "4", "CODER_EFFORT": "high", "CODER_USE_FALLBACKS": "false"}
        with mock.patch.dict("os.environ", env):
            config = CoderConfig.from_env()
        self.assertEqual(
            (config.max_steps, config.effort, config.use_fallbacks), (4, "high", False)
        )
        invalid = (
            {"CODER_MAX_STEPS": "11"},
            {"CODER_EFFORT": "huge"},
            {"CODER_EXEC_TIMEOUT": "0"},
        )
        for bad in invalid:
            with self.subTest(env=bad), mock.patch.dict("os.environ", bad):
                with self.assertRaises(ValueError):
                    CoderConfig.from_env()


class CliTests(unittest.TestCase):
    def test_prints_json_and_one_progress_line_per_step(self) -> None:
        steps = [
            Step(1, "t", [ToolCall("write_file", {"path": "a.py", "content": "x"}, "ok", False)]),
            Step(
                2,
                "t",
                [
                    ToolCall("exec_python", {"path": "a.py"}, "{}", False),
                    ToolCall("read_file", {"path": "b.txt"}, "b", False),
                ],
            ),
        ]

        class StubAgent:
            def __init__(self, config, on_step=None) -> None:
                self.on_step = on_step

            def run(self, task: str) -> CoderResult:
                for step in steps:
                    self.on_step(step)
                return CoderResult(code="x", explanation="e", plan=["p"], execution_result=None,
                                   status="completed", steps=steps)

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(coder_agent, "CoderAgent", StubAgent), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = coder_agent.main(["Write a.py"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "completed")
        self.assertEqual(
            stderr.getvalue().splitlines(),
            ["[step 1] write_file a.py", "[step 2] exec_python a.py, read_file b.txt"],
        )


if __name__ == "__main__":
    unittest.main()
