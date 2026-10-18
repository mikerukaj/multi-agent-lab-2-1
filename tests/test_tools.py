"""Tests for the Coder Agent's file and code-execution tools (contracts/tools.md)."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import TOOL_DEFINITIONS, dispatch_tool
from tools.file_tools import read_file, write_file
from tools.python_exec import SANDBOX_IMAGE, ensure_sandbox, exec_python, sandbox_command

requires_docker = unittest.skipUnless(shutil.which("docker"), "Docker is not available")


class WorkspaceTestCase(unittest.TestCase):
    """Gives each test a fresh, empty workspace directory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name) / "workspace"
        self.workspace.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, relative: str, content: str) -> Path:
        path = self.workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


class ReadFileTests(WorkspaceTestCase):
    def test_returns_contents(self) -> None:
        self.write("hello.txt", "hi there")
        self.assertEqual(read_file(self.workspace, "hello.txt"), ("hi there", False))

    def test_truncates_files_over_200_kb(self) -> None:
        self.write("big.txt", "a" * 300_000)
        text, is_error = read_file(self.workspace, "big.txt")
        self.assertFalse(is_error)
        self.assertTrue(text.startswith("a" * 200_000))
        self.assertTrue(text.endswith("\n[truncated: file is 300000 bytes]"))

    def test_missing_file_is_an_error(self) -> None:
        self.assertEqual(
            read_file(self.workspace, "nope.txt"), ("File not found: nope.txt", True)
        )

    def test_rejects_absolute_path(self) -> None:
        text, is_error = read_file(self.workspace, "/etc/passwd")
        self.assertTrue(is_error)
        self.assertEqual(text, "Path is outside the workspace: /etc/passwd")

    def test_rejects_parent_escape(self) -> None:
        (self.workspace.parent / "secret.txt").write_text("s", encoding="utf-8")
        text, is_error = read_file(self.workspace, "../secret.txt")
        self.assertTrue(is_error)
        self.assertIn("outside the workspace", text)

    def test_rejects_symlink_out_of_workspace(self) -> None:
        outside = self.workspace.parent / "outside.txt"
        outside.write_text("s", encoding="utf-8")
        (self.workspace / "link.txt").symlink_to(outside)
        text, is_error = read_file(self.workspace, "link.txt")
        self.assertTrue(is_error)
        self.assertIn("outside the workspace", text)


class WriteFileTests(WorkspaceTestCase):
    def test_creates_parent_folders_and_reports_bytes(self) -> None:
        text, is_error = write_file(self.workspace, "pkg/mod.py", "x = 1\n")
        self.assertEqual((text, is_error), ("Wrote 6 bytes to pkg/mod.py", False))
        self.assertEqual(
            (self.workspace / "pkg" / "mod.py").read_text(encoding="utf-8"), "x = 1\n"
        )

    def test_rejects_paths_outside_workspace(self) -> None:
        for path in ("/tmp/evil.py", "../evil.py"):
            with self.subTest(path=path):
                text, is_error = write_file(self.workspace, path, "x")
                self.assertTrue(is_error)
                self.assertIn("outside the workspace", text)
        self.assertFalse((self.workspace.parent / "evil.py").exists())


@requires_docker
class ExecPythonTests(WorkspaceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_sandbox()

    def test_returns_exit_code_and_output(self) -> None:
        self.write("ok.py", "import sys\nprint('out')\nprint('err', file=sys.stderr)\n")
        result = exec_python(self.workspace, "ok.py", timeout_seconds=10)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "out\n")
        self.assertEqual(result.stderr, "err\n")
        self.assertFalse(result.timed_out)
        self.assertEqual(result.path, "ok.py")
        self.assertGreater(result.duration_seconds, 0)

    def test_imports_sibling_modules(self) -> None:
        self.write("helper.py", "VALUE = 42\n")
        self.write("main.py", "from helper import VALUE\nprint(VALUE)\n")
        result = exec_python(self.workspace, "main.py", timeout_seconds=10)
        self.assertEqual((result.exit_code, result.stdout), (0, "42\n"))

    def test_non_zero_exit_is_not_a_tool_error(self) -> None:
        self.write("fail.py", "raise SystemExit(3)\n")
        text, is_error, result = dispatch_tool(
            "exec_python", {"path": "fail.py"}, self.workspace, 10
        )
        self.assertFalse(is_error)
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(json.loads(text)["exit_code"], 3)

    def test_timeout_kills_the_process(self) -> None:
        self.write("loop.py", "while True:\n    pass\n")
        result = exec_python(self.workspace, "loop.py", timeout_seconds=1)
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)
        self.assertLess(result.duration_seconds, 10)

    def test_truncates_each_stream(self) -> None:
        self.write(
            "loud.py",
            "import sys\nprint('o' * 20000)\nprint('e' * 20000, file=sys.stderr)\n",
        )
        result = exec_python(self.workspace, "loud.py", timeout_seconds=10)
        for stream in (result.stdout, result.stderr):
            self.assertLessEqual(len(stream), 10_000 + len("\n[truncated]"))
            self.assertTrue(stream.endswith("[truncated]"))

    def test_rejects_missing_non_python_and_outside_files(self) -> None:
        self.write("notes.txt", "hello")
        for path in ("missing.py", "notes.txt", "../x.py"):
            with self.subTest(path=path):
                text, is_error, result = dispatch_tool(
                    "exec_python", {"path": path}, self.workspace, 10
                )
                self.assertTrue(is_error)
                self.assertIsNone(result)
                self.assertTrue(text)

    def test_files_written_by_the_program_land_in_the_workspace(self) -> None:
        self.write("save.py", "open('out.txt', 'w').write('saved')\n")
        result = exec_python(self.workspace, "save.py", timeout_seconds=10)
        self.assertEqual(result.exit_code, 0, result.stderr)
        self.assertEqual((self.workspace / "out.txt").read_text(encoding="utf-8"), "saved")


@requires_docker
class SandboxIsolationTests(WorkspaceTestCase):
    """Generated code must not reach secrets, the network, or files outside its workspace."""

    @classmethod
    def setUpClass(cls) -> None:
        ensure_sandbox()

    def run_script(self, source: str) -> str:
        self.write("probe.py", source)
        result = exec_python(self.workspace, "probe.py", timeout_seconds=20)
        self.assertEqual(result.exit_code, 0, result.stderr)
        return result.stdout

    def test_secrets_are_not_visible_in_env_or_proc(self) -> None:
        source = (
            "import os, pathlib\n"
            "print(os.environ.get('ANTHROPIC_API_KEY'))\n"
            "found = [p for p in pathlib.Path('/proc').glob('[0-9]*/environ')\n"
            "         if b'sk-ant-secret' in p.read_bytes()]\n"
            "print(found)\n"
        )
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-secret"}):
            self.assertEqual(self.run_script(source), "None\n[]\n")

    def test_has_no_network(self) -> None:
        source = (
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
            "    print('connected')\n"
            "except OSError:\n"
            "    print('blocked')\n"
        )
        self.assertEqual(self.run_script(source), "blocked\n")

    def test_sees_only_its_own_workspace(self) -> None:
        (self.workspace.parent / "secret.txt").write_text("s", encoding="utf-8")
        source = (
            "import os\n"
            f"print(os.path.exists({str(self.workspace.parent / 'secret.txt')!r}))\n"
            "print(os.path.exists('../secret.txt'))\n"
            f"print(os.path.exists({os.path.expanduser('~')!r} + '/.ssh'))\n"
        )
        self.assertEqual(self.run_script(source), "False\nFalse\nFalse\n")

    def test_root_filesystem_is_read_only_and_user_is_not_root(self) -> None:
        source = (
            "import os\n"
            "try:\n"
            "    open('/etc/evil', 'w')\n"
            "    print('writable')\n"
            "except OSError:\n"
            "    print('read-only')\n"
            "print(os.getuid() != 0)\n"
        )
        self.assertEqual(self.run_script(source), "read-only\nTrue\n")


class SandboxCommandTests(unittest.TestCase):
    def test_command_locks_the_container_down(self) -> None:
        workspace = Path(tempfile.gettempdir()) / "ws"
        command = sandbox_command(workspace, workspace / "pkg" / "main.py", "n", SANDBOX_IMAGE)
        joined = " ".join(command)
        for flag in (
            "--rm",
            "--network none",
            "--read-only",
            "--cap-drop ALL",
            "--security-opt no-new-privileges",
            "--pids-limit 128",
            "--memory 512m",
            f"--user {os.getuid()}:{os.getgid()}",
            f"--volume {workspace.resolve()}:/work:rw",
        ):
            self.assertIn(flag, joined)
        self.assertEqual(command[-3:], ["python", "-s", "pkg/main.py"])
        self.assertNotIn("ANTHROPIC", joined)
        self.assertIn("@sha256:", SANDBOX_IMAGE)


class ToolDefinitionTests(unittest.TestCase):
    def test_every_tool_requires_a_purpose_under_a_strict_schema(self) -> None:
        for tool in TOOL_DEFINITIONS:
            with self.subTest(tool=tool["name"]):
                schema = tool["input_schema"]
                self.assertTrue(tool["strict"])
                self.assertIn("purpose", schema["properties"])
                self.assertEqual(sorted(schema["required"]), sorted(schema["properties"]))
                self.assertFalse(schema["additionalProperties"])


@requires_docker
class DispatchToolTests(WorkspaceTestCase):
    def test_unknown_tool_is_an_error(self) -> None:
        text, is_error, result = dispatch_tool("rm_rf", {}, self.workspace, 10)
        self.assertTrue(is_error)
        self.assertIn("Unknown tool", text)
        self.assertIsNone(result)

    def test_routes_file_tools(self) -> None:
        dispatch_tool("write_file", {"path": "a.txt", "content": "A"}, self.workspace, 10)
        text, is_error, result = dispatch_tool(
            "read_file", {"path": "a.txt"}, self.workspace, 10
        )
        self.assertEqual((text, is_error, result), ("A", False, None))


if __name__ == "__main__":
    unittest.main()
