"""Live end-to-end check of the Coder Agent against the real model.

Skipped unless ANTHROPIC_API_KEY is set (in the environment or .env).
"""

import os
import tempfile
import unittest
from pathlib import Path

from dotenv import load_dotenv

from agents.coder_agent import CoderConfig, run_coder_agent

load_dotenv()


@unittest.skipUnless(os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set")
class CoderAgentSmokeTest(unittest.TestCase):
    def test_fibonacci_task_returns_structured_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = CoderConfig(
                workspace_root=Path(tmp) / "workspace", chroma_path=Path(tmp) / "chroma"
            )
            result = run_coder_agent(
                "Write a function that returns the n-th Fibonacci number and print fib(10)",
                config,
            ).to_dict()

        self.assertEqual(result["status"], "completed", result["error"])
        self.assertIsInstance(result["code"], str)
        self.assertIsInstance(result["explanation"], str)
        self.assertIsInstance(result["plan"], list)
        self.assertTrue(all(isinstance(step, str) for step in result["plan"]))
        self.assertIsInstance(result["execution_result"], dict)
        self.assertEqual(result["execution_result"]["exit_code"], 0)
        self.assertIn("55", result["execution_result"]["stdout"])


if __name__ == "__main__":
    unittest.main()
