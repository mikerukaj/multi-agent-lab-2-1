"""Coder Agent: turns a plain-English coding task into working, executed Python code.

A LangGraph ReAct loop (recall -> think -> act, then finalize) drives Claude
through the Anthropic SDK. See specs/001-autonomous-dev-team/ for the design.

Usage:
    python -m agents.coder_agent "<task>" [--max-steps N] [--timeout SECONDS]
"""

import argparse
import json
import os
import secrets
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypedDict

import anthropic
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agents.memory import SemanticMemory, SlidingWindowBuffer, Turn
from tools import TOOL_DEFINITIONS, dispatch_tool
from tools.file_tools import read_file
from tools.python_exec import (
    SANDBOX_IMAGE,
    ExecutionResult,
    SandboxUnavailableError,
    ensure_sandbox,
    exec_python,
)

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
MAX_STEPS_CAP = 10
MAX_TOKENS = 16_000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
TERMINAL_STATUSES = ("refused", "error")

SYSTEM_PROMPT = """\
You are the Coder Agent in an autonomous software team. You receive one coding task \
and complete it on your own by writing Python files into your workspace and running them.

How to work:
- Before each tool call, say in one short sentence what you are about to do and why. \
These notes are the run's step-by-step record, so a reader can follow your work.
- Use paths relative to the workspace root with read_file, write_file and exec_python.
- Run the program's main file with exec_python, read the output, and fix any problem \
before you finish. Finish only when the program does what the task asks.
- Do not write test files. Tests belong to the QA Agent; check your work by running \
the program.
- Use only the Python standard library. Code runs in a sandbox with no network, so \
packages cannot be installed and the program cannot reach the internet.
- Write clean, modular code: PEP 8, type hints and docstrings on public functions, \
small functions with descriptive names, and an `if __name__ == "__main__":` guard \
on the entry point.

You work with the same user across many tasks. System messages may bring in notes \
from your earlier tasks with them. Treat what the user told you there (such as their \
name or preferences) as still true unless the current task says otherwise.

When you are done, reply with a short note on what you built."""

SUMMARY_PROMPT = """\
Summarize the work for the run record: the ordered steps you actually took, a short \
explanation of the solution, and the workspace-relative path of the program's main \
file (an empty string if you wrote none)."""

SUMMARY_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "plan": {"type": "array", "items": {"type": "string"}},
            "explanation": {"type": "string"},
            "main_file": {"type": "string"},
        },
        "required": ["plan", "explanation", "main_file"],
        "additionalProperties": False,
    },
}


@dataclass
class CoderConfig:
    """Settings for the Coder Agent; ``from_env`` applies ``CODER_*`` overrides."""

    model: str = "claude-sonnet-5-5"
    effort: str = "medium"
    max_steps: int = MAX_STEPS_CAP
    exec_timeout_seconds: float = 30.0
    window_turns: int = 5
    memory_top_k: int = 3
    workspace_root: Path = Path("workspace")
    chroma_path: Path = Path(".chroma")
    use_fallbacks: bool = True
    sandbox_image: str = SANDBOX_IMAGE

    def __post_init__(self) -> None:
        if not self.model or not self.sandbox_image:
            raise ValueError("model and sandbox_image must not be empty")
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {', '.join(EFFORT_LEVELS)}")
        if not 1 <= self.max_steps <= MAX_STEPS_CAP:
            raise ValueError(f"max_steps must be between 1 and {MAX_STEPS_CAP}")
        if self.exec_timeout_seconds <= 0:
            raise ValueError("exec_timeout_seconds must be greater than 0")
        if self.window_turns < 1 or self.memory_top_k < 1:
            raise ValueError("window_turns and memory_top_k must be at least 1")

    @classmethod
    def from_env(cls) -> "CoderConfig":
        """Build a config from defaults plus any ``CODER_*`` environment variables.

        Raises:
            ValueError: if a variable is malformed or out of range.
        """
        env = os.environ
        defaults = cls()
        return cls(
            model=env.get("CODER_MODEL", defaults.model),
            effort=env.get("CODER_EFFORT", defaults.effort),
            max_steps=int(env.get("CODER_MAX_STEPS", defaults.max_steps)),
            exec_timeout_seconds=float(
                env.get("CODER_EXEC_TIMEOUT", defaults.exec_timeout_seconds)
            ),
            workspace_root=Path(env.get("CODER_WORKSPACE", defaults.workspace_root)),
            chroma_path=Path(env.get("CODER_CHROMA_PATH", defaults.chroma_path)),
            use_fallbacks=env.get("CODER_USE_FALLBACKS", "true").lower()
            not in ("0", "false", "no"),
            sandbox_image=env.get("CODER_SANDBOX_IMAGE", defaults.sandbox_image),
        )


@dataclass
class ToolCall:
    """One tool the agent ran during a step, and what it returned."""

    tool: str
    input: dict
    output: str
    is_error: bool


@dataclass
class Step:
    """One Thought -> Action -> Observation step of the ReAct loop."""

    number: int
    thought: str
    actions: list[ToolCall] = field(default_factory=list)


@dataclass
class CoderResult:
    """Structured outcome of one Coder Agent run (contracts/coder-agent-api.md)."""

    code: str
    explanation: str
    plan: list[str]
    execution_result: ExecutionResult | None
    status: str = "completed"
    main_file: str | None = None
    files_written: list[str] = field(default_factory=list)
    steps_used: int = 0
    duration_seconds: float = 0.0
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    steps: list[Step] = field(default_factory=list)
    run_id: str = ""
    workspace: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable copy of this result."""
        return asdict(self)


class AgentState(TypedDict):
    """LangGraph state for one run. Nodes return the fields they change."""

    run_id: str
    task: str
    workspace: str
    window: list[Turn]
    messages: list[dict]
    step_count: int
    injected_memory_ids: list[str]
    steps: list[Step]
    files_written: list[str]
    last_execution: ExecutionResult | None
    stop_reason: str | None
    status: str
    error: str | None
    usage: dict
    result: CoderResult | None


def new_run_id() -> str:
    """Return a sortable, unique run id such as ``20261015-142233-a1b2c3``."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def build_request(config: CoderConfig, messages: list, output_format: dict | None = None) -> dict:
    """Assemble the keyword arguments for one ``beta.messages.create`` call.

    ``system`` and ``tools`` never change within a run, which keeps the history
    valid for Sonnet 5.5's preserved thinking (research R3).
    """
    output_config: dict[str, Any] = {"effort": config.effort}
    if output_format:
        output_config["format"] = output_format
    request = {
        "model": config.model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "tools": TOOL_DEFINITIONS,
        "messages": messages,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": output_config,
    }
    if config.use_fallbacks:
        request["betas"] = [FALLBACK_BETA]
        request["fallbacks"] = "default"
    return request


class CoderAgent:
    """A Coder Agent whose short-term memory lasts across ``run`` calls.

    Raises:
        SandboxUnavailableError: on construction, if Docker cannot run the sandbox
            that all generated code executes in.
    """

    def __init__(
        self,
        config: CoderConfig | None = None,
        client=None,
        on_step: Callable[[Step], None] | None = None,
    ) -> None:
        self.config = config or CoderConfig.from_env()
        ensure_sandbox(self.config.sandbox_image)
        self._client = client or anthropic.Anthropic(max_retries=3)
        self._on_step = on_step
        self._window = SlidingWindowBuffer(self.config.window_turns)
        self._memory = SemanticMemory(self.config.chroma_path)
        self._graph = self._build_graph()

    def run(self, task: str) -> CoderResult:
        """Complete ``task`` and return the structured result.

        Model, tool, and limit failures are reported in the result's ``status``
        and ``error``; they are never raised.

        Raises:
            ValueError: if ``task`` is empty.
        """
        if not task.strip():
            raise ValueError("task must not be empty")
        started = time.monotonic()
        run_id = new_run_id()
        workspace = (self.config.workspace_root / run_id).resolve()
        workspace.mkdir(parents=True)
        self._memory.add(run_id, "task", task)

        final = self._graph.invoke(
            _initial_state(run_id, task, workspace, self._window.get()),
            {"recursion_limit": 4 * self.config.max_steps + 5},
        )
        result = replace(final["result"], duration_seconds=round(time.monotonic() - started, 3))
        self._window.add(Turn(task=task, reply=result.explanation))
        if result.explanation:
            self._memory.add(run_id, "result", f"Task: {task}\nResult: {result.explanation}")
        return result

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("recall", self._recall)
        graph.add_node("think", self._think)
        graph.add_node("act", self._act)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "recall")
        graph.add_edge("recall", "think")
        graph.add_conditional_edges("think", route_after_think, ["act", "finalize"])
        graph.add_edge("act", "recall")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _recall(self, state: AgentState) -> dict:
        """Append memories from earlier runs that this run has not seen yet."""
        messages = list(state["messages"])
        if not messages:
            messages.append(
                {"role": "user", "content": _first_user_message(state["task"], state["window"])}
            )
        query = f"{state['task']}\n{_latest_observation(messages)}"
        hits = self._memory.query(
            query, top_k=self.config.memory_top_k, exclude_run_id=state["run_id"]
        )
        new_hits = [(id_, doc) for id_, doc in hits if id_ not in state["injected_memory_ids"]]
        if new_hits:
            messages.append({"role": "system", "content": _render_memories(new_hits)})
        return {
            "messages": messages,
            "injected_memory_ids": state["injected_memory_ids"] + [id_ for id_, _ in new_hits],
        }

    def _think(self, state: AgentState) -> dict:
        """Ask the model for the next Thought and Action."""
        try:
            response = self._client.beta.messages.create(
                **build_request(self.config, state["messages"])
            )
        except anthropic.APIError as error:
            return {"status": "error", "error": f"Model call failed: {error}"}

        step_count = state["step_count"] + 1
        update = {
            "messages": state["messages"] + [{"role": "assistant", "content": response.content}],
            "step_count": step_count,
            "usage": _add_usage(state["usage"], response.usage),
            "stop_reason": response.stop_reason,
            "steps": state["steps"] + [Step(step_count, _thought_text(response.content))],
        }
        if response.stop_reason == "refusal":
            update.update(status="refused", error=_refusal_message(response.stop_details))
        elif response.stop_reason == "tool_use" and step_count >= self.config.max_steps:
            update["status"] = "step_limit"
        elif response.stop_reason not in ("tool_use", "end_turn"):
            update.update(status="error", error=f"Unexpected stop reason: {response.stop_reason}")
        return update

    def _act(self, state: AgentState) -> dict:
        """Run every requested tool and return all results in one user message."""
        workspace = Path(state["workspace"])
        files_written = list(state["files_written"])
        last_execution = state["last_execution"]
        results, actions = [], []
        for block in _tool_uses(state["messages"][-1]):
            text, is_error, execution = dispatch_tool(
                block.name,
                block.input,
                workspace,
                self.config.exec_timeout_seconds,
                self.config.sandbox_image,
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": text,
                    "is_error": is_error,
                }
            )
            actions.append(ToolCall(block.name, dict(block.input), text, is_error))
            path = block.input.get("path")
            if block.name == "write_file" and not is_error and path not in files_written:
                files_written.append(path)
            last_execution = execution or last_execution
            observation = f"{_describe_call(block.name, block.input)} -> {text}"
            self._memory.add(state["run_id"], "observation", observation)

        step = replace(state["steps"][-1], actions=actions)
        if self._on_step:
            self._on_step(step)
        return {
            "messages": state["messages"] + [{"role": "user", "content": results}],
            "steps": state["steps"][:-1] + [step],
            "files_written": files_written,
            "last_execution": last_execution,
        }

    def _finalize(self, state: AgentState) -> dict:
        """Ask for the summary, then fill code and execution from ground truth."""
        status = "completed" if state["status"] == "running" else state["status"]
        error, usage, messages = state["error"], state["usage"], state["messages"]
        summary = {"plan": [], "explanation": error or "", "main_file": ""}
        if status not in TERMINAL_STATUSES:
            messages = messages + [{"role": "user", "content": _closing_content(messages[-1])}]
            try:
                response = self._client.beta.messages.create(
                    **build_request(self.config, messages, SUMMARY_FORMAT)
                )
                usage = _add_usage(usage, response.usage)
                summary = _parse_summary(response)
            except anthropic.APIError as api_error:
                status, error = "error", f"Summary call failed: {api_error}"
                summary["explanation"] = error

        workspace = Path(state["workspace"])
        main_file = _choose_main_file(summary["main_file"], state)
        code, missing = read_file(workspace, main_file) if main_file else ("", True)
        result = CoderResult(
            code="" if missing else code,
            explanation=summary["explanation"],
            plan=summary["plan"],
            execution_result=self._execution_for(workspace, main_file, state["last_execution"]),
            status=status,
            main_file=main_file,
            files_written=state["files_written"],
            steps_used=state["step_count"],
            usage=usage,
            steps=state["steps"],
            run_id=state["run_id"],
            workspace=str(workspace),
            error=error,
        )
        return {"messages": messages, "status": status, "usage": usage, "result": result}

    def _execution_for(
        self, workspace: Path, main_file: str | None, last: ExecutionResult | None
    ) -> ExecutionResult | None:
        """Return a real run of ``main_file``, reusing the last one if it matches."""
        if not main_file or not main_file.endswith(".py"):
            return None
        if last and os.path.normpath(last.path) == os.path.normpath(main_file):
            return last
        try:
            return exec_python(
                workspace, main_file, self.config.exec_timeout_seconds, self.config.sandbox_image
            )
        except (ValueError, FileNotFoundError):
            return None


def run_coder_agent(task: str, config: CoderConfig | None = None) -> CoderResult:
    """Run ``task`` once on a fresh ``CoderAgent``."""
    return CoderAgent(config).run(task)


def route_after_think(state: AgentState) -> str:
    """Send tool calls to ``act`` until a limit, refusal, or error ends the loop."""
    if state["status"] == "running" and state["stop_reason"] == "tool_use":
        return "act"
    return "finalize"


def _initial_state(run_id: str, task: str, workspace: Path, window: list[Turn]) -> AgentState:
    return AgentState(
        run_id=run_id,
        task=task,
        workspace=str(workspace),
        window=window,
        messages=[],
        step_count=0,
        injected_memory_ids=[],
        steps=[],
        files_written=[],
        last_execution=None,
        stop_reason=None,
        status="running",
        error=None,
        usage={"input_tokens": 0, "output_tokens": 0},
        result=None,
    )


def _first_user_message(task: str, window: list[Turn]) -> str:
    """Render the task, preceded by the recent conversation if there is one."""
    if not window:
        return f"Task:\n{task}"
    history = "\n".join(f"- Task: {turn.task}\n  Result: {turn.reply}" for turn in window)
    return f"Recent conversation (oldest first):\n{history}\n\nTask:\n{task}"


def _render_memories(hits: list[tuple[str, str]]) -> str:
    notes = "\n".join(f"- {document}" for _, document in hits)
    return f"Notes from your earlier tasks with this user (long-term memory):\n{notes}"


def _latest_observation(messages: list[dict]) -> str:
    """Return the text of the most recent tool results, or an empty string."""
    content = messages[-1]["content"]
    if messages[-1]["role"] != "user" or isinstance(content, str):
        return ""
    return "\n".join(block["content"] for block in content if block["type"] == "tool_result")


def _tool_uses(message: dict) -> list:
    if message["role"] != "assistant":
        return []
    return [block for block in message["content"] if block.type == "tool_use"]


def _closing_content(last_message: dict) -> list[dict]:
    """Answer any tool calls left pending by a limit, then ask for the summary."""
    pending = [
        {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": "Not run: the step limit was reached.",
            "is_error": True,
        }
        for block in _tool_uses(last_message)
    ]
    return pending + [{"type": "text", "text": SUMMARY_PROMPT}]


def _parse_summary(response) -> dict:
    """Read the structured summary, tolerating a refusal or malformed text."""
    text = "".join(block.text for block in response.content if block.type == "text")
    if response.stop_reason == "refusal":
        explanation = _refusal_message(response.stop_details)
        return {"plan": [], "explanation": explanation, "main_file": ""}
    try:
        data = json.loads(text)
        return {
            "plan": [str(step) for step in data["plan"]],
            "explanation": str(data["explanation"]),
            "main_file": str(data["main_file"]),
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"plan": [], "explanation": text, "main_file": ""}


def _choose_main_file(reported: str, state: AgentState) -> str | None:
    """Prefer the model's main file if it exists; else the last file run or written."""
    if reported in state["files_written"]:
        return reported
    if state["last_execution"]:
        return state["last_execution"].path
    python_files = [path for path in state["files_written"] if path.endswith(".py")]
    return python_files[-1] if python_files else None


def _thought_text(content: list) -> str:
    """Join the step's reasoning summaries and notes.

    When the model wrote none, fall back to the ``purpose`` of each tool call,
    which the strict tool schemas require, so a step's thought is never empty.
    """
    parts = []
    for block in content:
        if block.type == "thinking" and block.thinking:
            parts.append(block.thinking)
        elif block.type == "text" and block.text:
            parts.append(block.text)
    if not parts:
        parts = [
            block.input["purpose"]
            for block in content
            if block.type == "tool_use" and block.input.get("purpose")
        ]
    return "\n".join(parts)


def _add_usage(total: dict, usage) -> dict:
    return {
        "input_tokens": total["input_tokens"] + usage.input_tokens,
        "output_tokens": total["output_tokens"] + usage.output_tokens,
    }


def _refusal_message(stop_details) -> str:
    if stop_details is None:
        return "The model declined the request."
    return (
        f"The model declined the request (category: {stop_details.category}): "
        f"{stop_details.explanation}"
    )


def _describe_call(name: str, tool_input: dict) -> str:
    """Short label for a tool call, e.g. ``write_file fizzbuzz.py``."""
    return f"{name} {tool_input.get('path', '')}".strip()


def _print_progress(step: Step) -> None:
    calls = ", ".join(_describe_call(action.tool, action.input) for action in step.actions)
    print(f"[step {step.number}] {calls}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns 0 if the task completed, 1 otherwise, 2 for usage errors."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="python -m agents.coder_agent",
        description="Turn a plain-English coding task into working Python code.",
    )
    parser.add_argument("task", help="what the code should do")
    parser.add_argument("--max-steps", type=int, help=f"ReAct step cap (1-{MAX_STEPS_CAP})")
    parser.add_argument("--timeout", type=float, help="seconds before a code run is killed")
    args = parser.parse_args(argv)

    try:
        config = CoderConfig.from_env()
        if args.max_steps is not None:
            config = replace(config, max_steps=args.max_steps)
        if args.timeout is not None:
            config = replace(config, exec_timeout_seconds=args.timeout)
        result = CoderAgent(config, on_step=_print_progress).run(args.task)
    except (ValueError, SandboxUnavailableError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
