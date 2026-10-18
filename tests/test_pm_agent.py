"""Offline tests for the PM Agent (contracts/pm-agent-api.md), using a fake client."""

import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import anthropic
import httpx2

from agents.llm import FALLBACK_BETA
from agents.pm_agent import (
    MAX_TASKS,
    SPEC_SCHEMA,
    PMAgent,
    PMConfig,
    criteria_preserved,
    merge_tasks,
    number_tasks,
    validate_tasks,
)
from orchestration.state import Task, TechnicalSpec
from tests.team_fakes import (
    FakePMClient,
    pm_response,
    regroup_payload,
    spec_payload,
    tasks_payload,
)

REQUIREMENT = "Build a temperature converter CLI"


def api_error() -> anthropic.APIConnectionError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(request=request)


def make_pm(script: list, **config) -> tuple[PMAgent, FakePMClient]:
    client = FakePMClient(script)
    return PMAgent(PMConfig(**config), client=client), client


def flat_criteria(tasks: list[dict]) -> list[str]:
    return [criterion for task in tasks for criterion in task["acceptance_criteria"]]


class WriteSpecTests(unittest.TestCase):
    def test_spec_is_parsed_and_tasks_are_numbered_in_order(self) -> None:
        payload = spec_payload(3)
        pm, _ = make_pm([pm_response(payload)])

        result = pm.write_spec(REQUIREMENT)

        self.assertIsNone(result.error)
        self.assertIsNone(result.rejection_reason)
        self.assertEqual([task["task_id"] for task in result.spec["tasks"]], ["T1", "T2", "T3"])
        for task, original in zip(result.spec["tasks"], payload["tasks"]):
            self.assertEqual({**original, "task_id": task["task_id"]}, task)
        self.assertEqual(result.spec["assumptions"], payload["assumptions"])
        self.assertEqual(result.usage, {"input_tokens": 100, "output_tokens": 50})

    def test_request_uses_structured_output_and_no_tools(self) -> None:
        pm, client = make_pm([pm_response(spec_payload(1))])

        pm.write_spec(REQUIREMENT)

        request = client.requests[0]
        self.assertNotIn("tools", request)
        self.assertEqual(request["model"], "claude-sonnet-5-5")
        self.assertEqual(request["thinking"], {"type": "adaptive"})
        self.assertEqual(request["output_config"]["effort"], "high")
        self.assertEqual(request["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(request["betas"], [FALLBACK_BETA])
        self.assertEqual(request["fallbacks"], "default")
        self.assertIn(REQUIREMENT, request["messages"][0]["content"])

    def test_fallbacks_can_be_switched_off(self) -> None:
        pm, client = make_pm([pm_response(spec_payload(1))], use_fallbacks=False)

        pm.write_spec(REQUIREMENT)

        self.assertNotIn("betas", client.requests[0])
        self.assertNotIn("fallbacks", client.requests[0])

    def test_rejection_is_reported_without_a_spec(self) -> None:
        pm, _ = make_pm([pm_response(spec_payload(0, rejection_reason="Not software."))])

        result = pm.write_spec("What's a good name for a cat?")

        self.assertIsNone(result.spec)
        self.assertEqual(result.rejection_reason, "Not software.")
        self.assertIsNone(result.error)

    def test_failures_are_reported_not_raised(self) -> None:
        no_tasks = spec_payload(0)
        no_criteria = spec_payload(2)
        no_criteria["tasks"][1]["acceptance_criteria"] = []
        refusal = SimpleNamespace(category="cyber", explanation="Declined.")
        cases = {
            "api error": api_error(),
            "refusal": pm_response(None, stop_reason="refusal", stop_details=refusal),
            "not json": SimpleNamespace(
                content=[SimpleNamespace(type="text", text="not json")],
                stop_reason="end_turn",
                stop_details=None,
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            ),
            "no tasks": pm_response(no_tasks),
            "task without criteria": pm_response(no_criteria),
        }
        for name, item in cases.items():
            with self.subTest(name):
                pm, _ = make_pm([item])
                result = pm.write_spec(REQUIREMENT)
                self.assertIsNone(result.spec)
                self.assertTrue(result.error)

    def test_refusal_error_names_the_category(self) -> None:
        refusal = SimpleNamespace(category="cyber", explanation="Declined.")
        pm, _ = make_pm([pm_response(None, stop_reason="refusal", stop_details=refusal)])

        self.assertIn("cyber", pm.write_spec(REQUIREMENT).error)

    def test_blank_requirement_raises(self) -> None:
        pm, client = make_pm([])

        with self.assertRaises(ValueError):
            pm.write_spec("  ")
        self.assertEqual(client.requests, [])


class ConsolidateTests(unittest.TestCase):
    def setUp(self) -> None:
        pm, _ = make_pm([pm_response(spec_payload(11))])
        self.spec: TechnicalSpec = pm.write_spec(REQUIREMENT).spec
        self.original = self.spec["tasks"]
        self.good_reply = regroup_payload([dict(task) for task in self.original], 6)

    def consolidate(self, reply) -> tuple:
        item = reply if isinstance(reply, Exception) else pm_response(reply)
        pm, client = make_pm([item])
        return pm.consolidate(self.spec, REQUIREMENT), client

    def test_good_merge_is_accepted_and_renumbered(self) -> None:
        result, client = self.consolidate(self.good_reply)

        self.assertIsNone(result.error)
        self.assertEqual(
            [task["task_id"] for task in result.spec["tasks"]], [f"T{n}" for n in range(1, 7)]
        )
        self.assertEqual(flat_criteria(result.spec["tasks"]), flat_criteria(self.original))
        self.assertEqual(result.spec["name"], self.spec["name"])
        request = client.requests[0]
        self.assertIn(REQUIREMENT, request["messages"][0]["content"])
        self.assertIn(json.dumps(self.original, indent=2), request["messages"][0]["content"])
        schema = request["output_config"]["format"]["schema"]
        self.assertEqual(set(schema["properties"]), {"tasks"})

    def test_case_and_spacing_differences_are_accepted(self) -> None:
        reply = copy.deepcopy(self.good_reply)
        first = reply["tasks"][0]["acceptance_criteria"]
        first[0] = "  " + first[0].upper().replace(" ", "   ") + " "

        result, _ = self.consolidate(reply)

        self.assertIsNone(result.error)

    def test_bad_replies_are_errors_not_exceptions(self) -> None:
        def mutated(change) -> dict:
            reply = copy.deepcopy(self.good_reply)
            change(reply["tasks"])
            return reply

        cases = {
            "too many tasks": tasks_payload(MAX_TASKS + 1),
            "no tasks": {"tasks": []},
            "empty criteria": mutated(lambda t: t[0].update(acceptance_criteria=[])),
            "invalid priority": mutated(lambda t: t[0].update(priority="urgent")),
            "dropped criterion": mutated(lambda t: t[0]["acceptance_criteria"].pop()),
            "reworded criterion": mutated(
                lambda t: t[1]["acceptance_criteria"].__setitem__(0, "Something else")
            ),
            "swapped criteria": mutated(
                lambda t: t.__setitem__(slice(0, 2), [t[1], t[0]])
            ),
            "api error": api_error(),
        }
        for name, reply in cases.items():
            with self.subTest(name):
                result, _ = self.consolidate(reply)
                self.assertIsNone(result.spec)
                self.assertTrue(result.error)


class HelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks: list[Task] = number_tasks(spec_payload(11)["tasks"])

    def test_criteria_preserved(self) -> None:
        merged = regroup_payload(self.tasks, 4)["tasks"]
        self.assertTrue(criteria_preserved(self.tasks, merged))

        extra = copy.deepcopy(merged)
        extra[0]["acceptance_criteria"].append("An extra combined check")
        self.assertTrue(criteria_preserved(self.tasks, extra))

        missing = copy.deepcopy(merged)
        missing[2]["acceptance_criteria"].pop(0)
        self.assertFalse(criteria_preserved(self.tasks, missing))

        reordered = copy.deepcopy(merged)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        self.assertFalse(criteria_preserved(self.tasks, reordered))

    def test_validate_tasks(self) -> None:
        self.assertIsNone(validate_tasks(self.tasks))
        self.assertIn("no tasks", validate_tasks([]))
        broken = {
            "title": dict(self.tasks[0], title=""),
            "description": dict(self.tasks[0], description=" "),
            "acceptance criteria": dict(self.tasks[0], acceptance_criteria=[]),
            "priority": dict(self.tasks[0], priority="urgent"),
        }
        for problem, task in broken.items():
            with self.subTest(problem):
                self.assertIn(problem, validate_tasks([task]))

    def test_merge_tasks_keeps_order_criteria_and_highest_priority(self) -> None:
        merged = merge_tasks(self.tasks, MAX_TASKS)

        self.assertEqual([task["task_id"] for task in merged], [f"T{n}" for n in range(1, 9)])
        self.assertEqual(flat_criteria(merged), flat_criteria(self.tasks))
        # 11 tasks into 8 groups: sizes 2, 2, 2, 1, 1, 1, 1, 1.
        self.assertEqual(merged[0]["title"], "Task 1 + Task 2")
        self.assertEqual(merged[0]["priority"], "high")
        self.assertEqual(merged[2]["title"], "Task 5 + Task 6")
        self.assertEqual(merged[2]["priority"], "medium")
        self.assertEqual(merged[3]["title"], "Task 7")
        self.assertIn("1. Task 1:", merged[0]["description"])
        self.assertEqual(merged[3]["description"], self.tasks[6]["description"])

    def test_merge_tasks_leaves_short_lists_alone_apart_from_numbering(self) -> None:
        tasks = number_tasks(spec_payload(3)["tasks"])[::-1]

        merged = merge_tasks(tasks)

        self.assertEqual([task["title"] for task in merged], ["Task 3", "Task 2", "Task 1"])
        self.assertEqual([task["task_id"] for task in merged], ["T1", "T2", "T3"])


class SchemaAndConfigTests(unittest.TestCase):
    def test_spec_schema_matches_the_state_types(self) -> None:
        task_properties = set(SPEC_SCHEMA["properties"]["tasks"]["items"]["properties"])
        self.assertEqual(task_properties | {"task_id"}, Task.__required_keys__)
        top_level = set(SPEC_SCHEMA["properties"]) - {"rejection_reason"}
        self.assertEqual(top_level, TechnicalSpec.__required_keys__)

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            PMConfig(effort="extreme")
        with self.assertRaises(ValueError):
            PMConfig(model="")

    def test_config_from_env(self) -> None:
        env = {"PM_MODEL": "claude-opus-5-5", "PM_EFFORT": "max", "PM_USE_FALLBACKS": "false"}
        with mock.patch.dict(os.environ, env):
            config = PMConfig.from_env()

        self.assertEqual(config, PMConfig("claude-opus-5-5", "max", False))


if __name__ == "__main__":
    unittest.main()
