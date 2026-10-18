"""Tests for the Coder Agent's two-tier memory (agents/memory.py)."""

import tempfile
import unittest
from pathlib import Path

from agents.memory import SemanticMemory, SlidingWindowBuffer, Turn


class SlidingWindowBufferTests(unittest.TestCase):
    def test_sixth_turn_evicts_the_oldest(self) -> None:
        window = SlidingWindowBuffer(max_turns=5)
        for i in range(6):
            window.add(Turn(task=f"task {i}", reply=f"reply {i}"))
        tasks = [turn.task for turn in window.get()]
        self.assertEqual(tasks, ["task 1", "task 2", "task 3", "task 4", "task 5"])

    def test_get_returns_a_copy(self) -> None:
        window = SlidingWindowBuffer(max_turns=2)
        window.add(Turn(task="a", reply="b"))
        window.get().clear()
        self.assertEqual(len(window.get()), 1)

    def test_rejects_non_positive_size(self) -> None:
        with self.assertRaises(ValueError):
            SlidingWindowBuffer(max_turns=0)


class SemanticMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.memory = SemanticMemory(Path(self._tmp.name) / "chroma")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_empty_collection_returns_nothing(self) -> None:
        self.assertEqual(self.memory.query("anything", top_k=3, exclude_run_id="r0"), [])

    def test_retrieves_semantically_similar_record(self) -> None:
        self.memory.add("run-a", "task", "The user's name is Priya and she likes tea.")
        self.memory.add("run-a", "task", "Compute the area of a circle with radius 3.")
        results = self.memory.query("What is my name?", top_k=1, exclude_run_id="run-b")
        self.assertEqual(len(results), 1)
        self.assertIn("Priya", results[0][1])

    def test_excludes_the_given_run(self) -> None:
        self.memory.add("current", "observation", "Wrote fizzbuzz.py")
        self.memory.add("earlier", "observation", "Wrote fizzbuzz.py earlier")
        results = self.memory.query("fizzbuzz", top_k=5, exclude_run_id="current")
        self.assertEqual([doc for _, doc in results], ["Wrote fizzbuzz.py earlier"])

    def test_ids_are_unique_and_prefixed_by_run(self) -> None:
        first = self.memory.add("run-x", "task", "one")
        second = self.memory.add("run-x", "task", "two")
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("run-x-"))

    def test_truncates_documents_to_2000_chars(self) -> None:
        self.memory.add("run-a", "observation", "z" * 5000)
        (_, document), = self.memory.query("z", top_k=1, exclude_run_id="run-b")
        self.assertEqual(len(document), 2000)

    def test_persists_across_instances(self) -> None:
        self.memory.add("run-a", "result", "Persisted fact about penguins.")
        reopened = SemanticMemory(Path(self._tmp.name) / "chroma")
        results = reopened.query("penguins", top_k=1, exclude_run_id="run-b")
        self.assertIn("penguins", results[0][1])


if __name__ == "__main__":
    unittest.main()
