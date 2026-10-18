"""PM Agent: turns a plain-English requirement into a technical spec and ordered tasks.

One structured-output Claude call writes the spec; a second, only when the plan
has more than ``MAX_TASKS`` tasks, merges them. Both replies are checked in code
before they are used. See specs/001-autonomous-dev-team/contracts/pm-agent-api.md.
"""

import json
import os
import typing
from dataclasses import dataclass

import anthropic

from agents.llm import EFFORT_LEVELS, FALLBACK_BETA, MAX_TOKENS, refusal_message
from orchestration.state import Priority, Task, TechnicalSpec, TokenUsage

MAX_TASKS = 8
PRIORITIES = typing.get_args(Priority)
_PRIORITY_RANK = {priority: rank for rank, priority in enumerate(PRIORITIES)}

TASK_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
        "priority": {"type": "string", "enum": list(PRIORITIES)},
    },
    "required": ["title", "description", "acceptance_criteria", "priority"],
    "additionalProperties": False,
}

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "tasks": {"type": "array", "items": TASK_ITEM_SCHEMA},
        "rejection_reason": {"type": "string"},
    },
    "required": [
        "name",
        "description",
        "acceptance_criteria",
        "assumptions",
        "tasks",
        "rejection_reason",
    ],
    "additionalProperties": False,
}

CONSOLIDATION_SCHEMA = {
    "type": "object",
    "properties": {"tasks": {"type": "array", "items": TASK_ITEM_SCHEMA}},
    "required": ["tasks"],
    "additionalProperties": False,
}

PM_SYSTEM_PROMPT = f"""\
You are the Product Manager Agent in an autonomous software team. You receive one \
software requirement in plain English and turn it into a technical specification \
for the Coder Agent, who builds it task by task without asking anyone anything.

Write the specification:
- name: a short name for the software.
- description: the intended behavior, its inputs and its outputs.
- acceptance_criteria: how to tell that the whole program works.
- assumptions: every assumption you made where the requirement is unclear. Never ask \
questions; decide, and record the decision here.
- tasks: the work for the Coder, in the order it must be built. Put tasks that \
others depend on first, and higher-priority tasks before lower-priority ones where \
the order is otherwise free. Give each task a short imperative title, a description \
with its inputs and outputs, at least one acceptance criterion that can be checked \
by running the program, and a priority (high, medium or low). Use at most \
{MAX_TASKS} tasks; when the work is large, make each task bigger instead of adding \
more.
- rejection_reason: leave this empty. Fill it in, and leave the lists empty, only \
when the requirement is empty, nonsense, or not a request for software. Never \
reject a requirement because it is large or complex.

Limits of the current team:
- Do not write tasks for writing tests. Tests belong to the QA Agent.
- The Coder can use only the Python standard library, and its code runs with no \
network access. When the requirement needs more (a package, a database server, an \
outside service), plan a standard-library version and record that as an assumption."""

CONSOLIDATE_PROMPT = f"""\
The plan below has more than {MAX_TASKS} tasks. Merge it into at most {MAX_TASKS} \
tasks by combining neighbouring tasks. Keep the task order and the full scope of \
the requirement. Copy every acceptance criterion word for word, in its original \
order, into the merged task that covers it. The system checks this, and rejects a \
merge that drops, rewords or reorders a criterion."""


@dataclass
class PMConfig:
    """Settings for the PM Agent; ``from_env`` applies ``PM_*`` overrides."""

    model: str = "claude-sonnet-5-5"
    effort: str = "high"
    use_fallbacks: bool = True

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {', '.join(EFFORT_LEVELS)}")

    @classmethod
    def from_env(cls) -> "PMConfig":
        """Build a config from defaults plus any ``PM_*`` environment variables.

        Raises:
            ValueError: if a variable is malformed or out of range.
        """
        env = os.environ
        defaults = cls()
        return cls(
            model=env.get("PM_MODEL", defaults.model),
            effort=env.get("PM_EFFORT", defaults.effort),
            use_fallbacks=env.get("PM_USE_FALLBACKS", "true").lower()
            not in ("0", "false", "no"),
        )


@dataclass
class PMResult:
    """Outcome of one PM call: exactly one of ``spec``, ``rejection_reason`` or ``error``."""

    spec: TechnicalSpec | None
    rejection_reason: str | None
    error: str | None
    usage: TokenUsage


class PMAgent:
    """The Product Manager Agent. Every call stands alone; no history is kept."""

    def __init__(self, config: PMConfig | None = None, client=None) -> None:
        self.config = config or PMConfig.from_env()
        self._client = client or anthropic.Anthropic(max_retries=3)

    def write_spec(self, requirement: str) -> PMResult:
        """Write the technical specification and ordered tasks for ``requirement``.

        Model, refusal and output problems are reported in ``PMResult.error``.
        The number of tasks is not capped here; the team graph consolidates.

        Raises:
            ValueError: if ``requirement`` is empty.
        """
        if not requirement.strip():
            raise ValueError("requirement must not be empty")
        data, error, usage = self._call(f"Requirement:\n{requirement}", SPEC_SCHEMA)
        if error is None and data["rejection_reason"].strip():
            return PMResult(None, data["rejection_reason"].strip(), None, usage)
        error = error or _validate_spec(data)
        if error:
            return PMResult(None, None, error, usage)
        spec = TechnicalSpec(
            name=data["name"],
            description=data["description"],
            acceptance_criteria=data["acceptance_criteria"],
            assumptions=data["assumptions"],
            tasks=number_tasks(data["tasks"]),
        )
        return PMResult(spec, None, None, usage)

    def consolidate(self, spec: TechnicalSpec, requirement: str) -> PMResult:
        """Ask the PM to merge ``spec``'s tasks into at most ``MAX_TASKS``.

        The reply is used only if its tasks are valid, number at most
        ``MAX_TASKS``, and keep every original criterion in order (FR-006a).
        Otherwise ``PMResult.error`` says which check failed.
        """
        user_text = (
            f"Requirement:\n{requirement}\n\n{CONSOLIDATE_PROMPT}\n\n"
            f"Current tasks:\n{json.dumps(spec['tasks'], indent=2)}"
        )
        data, error, usage = self._call(user_text, CONSOLIDATION_SCHEMA)
        if error is None:
            error = _consolidation_problem(spec["tasks"], data["tasks"])
        if error:
            return PMResult(None, None, f"Consolidation rejected: {error}", usage)
        return PMResult({**spec, "tasks": number_tasks(data["tasks"])}, None, None, usage)

    def _request(self, user_text: str, schema: dict) -> dict:
        """Assemble the keyword arguments for one ``beta.messages.create`` call."""
        request = {
            "model": self.config.model,
            "max_tokens": MAX_TOKENS,
            "system": PM_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_text}],
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": self.config.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        if self.config.use_fallbacks:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    def _call(self, user_text: str, schema: dict) -> tuple[dict | None, str | None, TokenUsage]:
        """Make one model call and return its parsed JSON, an error, and the usage."""
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        try:
            response = self._client.beta.messages.create(**self._request(user_text, schema))
        except anthropic.APIError as error:
            return None, f"Model call failed: {error}", usage
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
        )
        if response.stop_reason == "refusal":
            return None, refusal_message(response.stop_details), usage
        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            return json.loads(text), None, usage
        except json.JSONDecodeError:
            return None, f"Model output is not valid JSON: {text[:200]!r}", usage


def number_tasks(items: list[dict]) -> list[Task]:
    """Return the tasks with IDs ``T1``…``Tn`` assigned by position."""
    return [
        Task(
            task_id=f"T{number}",
            title=item["title"],
            description=item["description"],
            acceptance_criteria=list(item["acceptance_criteria"]),
            priority=item["priority"],
        )
        for number, item in enumerate(items, start=1)
    ]


def merge_tasks(tasks: list[Task], limit: int = MAX_TASKS) -> list[Task]:
    """Merge neighbouring tasks until there are at most ``limit``, keeping order.

    The list is split into ``limit`` contiguous groups whose sizes differ by at
    most one (larger groups first). Each group keeps every acceptance criterion
    in order and takes its highest priority.
    """
    if len(tasks) <= limit:
        return number_tasks(tasks)
    size, extra = divmod(len(tasks), limit)
    groups, start = [], 0
    for index in range(limit):
        end = start + size + (1 if index < extra else 0)
        groups.append(tasks[start:end])
        start = end
    return number_tasks([_merge_group(group) for group in groups])


def _merge_group(group: list[Task]) -> dict:
    if len(group) == 1:
        return dict(group[0])
    return {
        "title": " + ".join(task["title"] for task in group),
        "description": "\n".join(
            f"{number}. {task['title']}: {task['description']}"
            for number, task in enumerate(group, start=1)
        ),
        "acceptance_criteria": [c for task in group for c in task["acceptance_criteria"]],
        "priority": min((task["priority"] for task in group), key=_PRIORITY_RANK.__getitem__),
    }


def validate_tasks(tasks: list[dict]) -> str | None:
    """Return the first problem with ``tasks``, or ``None`` if every task is valid."""
    if not tasks:
        return "the plan has no tasks"
    for number, task in enumerate(tasks, start=1):
        if not task["title"].strip():
            return f"task {number} has an empty title"
        if not task["description"].strip():
            return f"task {number} has an empty description"
        if not [c for c in task["acceptance_criteria"] if c.strip()]:
            return f"task {number} has no acceptance criteria"
        if task["priority"] not in PRIORITIES:
            return f"task {number} has an invalid priority {task['priority']!r}"
    return None


def criteria_preserved(original: list[Task], merged: list[dict]) -> bool:
    """Whether every original criterion appears in ``merged``, in the same order.

    Criteria are compared ignoring letter case and spacing. Extra criteria in
    ``merged`` are allowed.
    """
    remaining = iter(_flat_criteria(merged))
    # `in` on an iterator consumes it up to the match, so this checks an in-order subsequence.
    return all(criterion in remaining for criterion in _flat_criteria(original))


def _flat_criteria(tasks: list[dict]) -> list[str]:
    return [
        " ".join(criterion.split()).casefold()
        for task in tasks
        for criterion in task["acceptance_criteria"]
    ]


def _validate_spec(data: dict) -> str | None:
    for field in ("name", "description"):
        if not data[field].strip():
            return f"the spec has an empty {field}"
    if not data["acceptance_criteria"]:
        return "the spec has no acceptance criteria"
    return validate_tasks(data["tasks"])


def _consolidation_problem(original: list[Task], merged: list[dict]) -> str | None:
    problem = validate_tasks(merged)
    if problem:
        return problem
    if len(merged) > MAX_TASKS:
        return f"{len(merged)} tasks is still more than {MAX_TASKS}"
    if not criteria_preserved(original, merged):
        return "an acceptance criterion was dropped, reworded or moved out of order"
    return None
