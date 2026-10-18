# Autonomous Multi-Agent Development Team

**Course:** Build Autonomous Multi-Agent Systems

The goal is a team of three LLM-powered agents that turn a plain-English software
requirement into tested, working Python code, with no human input after the first prompt:

1. **Product Manager (PM) Agent:** interprets the requirement, writes a technical
   specification, and breaks it into engineering tasks.
2. **Coder Agent:** implements the tasks as clean, modular Python and revises the code
   until every test passes.
3. **QA & Debugger Agent:** writes the automated tests, reviews the code for bugs and
   style problems, and sends findings back to the Coder.

The full specification, plan and task list are in `specs/001-autonomous-dev-team/`.

## Status: Increment 1, the Coder Agent

Only the **Coder Agent** exists so far. Given a coding task, it plans, writes files, runs
them, fixes what fails, and returns a structured result. The PM and QA agents, the
Coder ⇄ QA loop, and run limits on time and spend come in later increments.

## Safety: where generated code runs

The Coder Agent runs the code it writes, and that code is untrusted. **It never runs on
your machine directly.** Every `exec_python` call starts a fresh, throwaway Docker
container that:

- has **no network** (`--network none`), so nothing can be downloaded or sent out;
- has **no secrets**: the only environment variables are two Python settings, and the
  agent's own process (which holds your API key) is not visible inside the container;
- sees **only that run's workspace** (`workspace/<run_id>/`, mounted at `/work`); the rest
  of the filesystem is the read-only sandbox image;
- runs as your user ID, never root, with all Linux capabilities dropped and
  `no-new-privileges`;
- is limited to 512 MB of memory, 1 CPU, and 128 processes, and is killed at the time
  limit (default 30 s);
- uses a Python 3.12 image pinned by digest, so it can't change underneath you.

If Docker is not available, the agent refuses to start rather than run code unsandboxed.
No container is given the Docker socket.

## Setup

You need Python 3.12, Docker (your user must be able to run `docker`), and an Anthropic
API key.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
```

The first run pulls the sandbox image (`python:3.12-slim`, pinned by digest) and Chroma's
embedding model (about 80 MB, cached in `~/.cache/chroma`).

Chroma needs SQLite 3.35 or newer. On hosts with an older system SQLite (RHEL 8 ships
3.26), `agents/memory.py` switches to the bundled `pysqlite3-binary` build automatically.

## Usage

### Command line

```bash
python -m agents.coder_agent "Write fizzbuzz.py that prints FizzBuzz for 1 to 15"
```

| Option | Meaning |
|--------|---------|
| `--max-steps N` | Maximum model calls for the task, 1 to 10 (default 10) |
| `--timeout SECONDS` | Seconds before a run of generated code is killed (default 30) |

Progress goes to stderr, one line per step (`[step 2] exec_python fizzbuzz.py`). The
result is printed to stdout as JSON. The exit code is 0 if the task completed, 1 for any
other outcome (`step_limit`, `refused`, `error`), and 2 for usage errors (including Docker
not being available).

### Python

```python
from agents.coder_agent import CoderAgent

agent = CoderAgent()               # keeps short-term memory across run() calls
result = agent.run("Write primes.py that prints the primes below 50")
print(result.status, result.execution_result.stdout)
```

### What a result contains

| Field | Meaning |
|-------|---------|
| `code` | Contents of the main file, read from disk |
| `explanation` | What the agent built |
| `plan` | The steps it actually took |
| `execution_result` | A real run of the main file: exit code, stdout, stderr, timeout flag |
| `status` | `completed`, `step_limit`, `refused`, or `error` |
| `steps` | Thought → Action → Observation trace, one entry per model call |
| `usage`, `duration_seconds` | Tokens used and wall-clock time for the run |

The full contract is in `specs/001-autonomous-dev-team/contracts/coder-agent-api.md`.

### Configuration

All settings are optional environment variables (or `.env` entries):

| Variable | Default | Meaning |
|----------|---------|---------|
| `CODER_MODEL` | `claude-sonnet-5-5` | Model ID |
| `CODER_EFFORT` | `medium` | `low`, `medium`, `high`, `xhigh`, or `max` |
| `CODER_MAX_STEPS` | `10` | Maximum model calls per task (1 to 10) |
| `CODER_EXEC_TIMEOUT` | `30` | Seconds before generated code is killed |
| `CODER_WORKSPACE` | `workspace` | Folder for per-run workspaces |
| `CODER_CHROMA_PATH` | `.chroma` | Folder for long-term memory |
| `CODER_USE_FALLBACKS` | `true` | Retry declined requests on a fallback model |
| `CODER_SANDBOX_IMAGE` | `python:3.12-slim@sha256:a6e3…` | Image that runs generated code; keep it pinned by digest |

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

The suite runs offline. Tests that execute code need Docker and are skipped without it;
the live smoke test runs only when `ANTHROPIC_API_KEY` is set.

## Design decisions

- **Claude Sonnet 5.5 through the official `anthropic` SDK**, with adaptive thinking
  (summarized, so each step's reasoning appears in the trace), `medium` effort, and
  server-side refusal fallback.
- **LangGraph for the loop:** a four-node `StateGraph` (`recall → think → act`, then
  `finalize`). Only the core graph API is used; the project constitution allows nothing
  more. The same graph will carry the PM → Coder ⇄ QA flow later.
- **Append-only history.** Sonnet 5.5 rejects requests whose earlier messages were edited,
  so the system prompt and tools never change within a run, and memories are added as new
  system messages instead of being written into the system prompt.
- **Two-tier memory.** A 5-task sliding window is shown at the start of each run, and a
  persistent Chroma store (with Chroma's built-in embeddings, so no extra API key) is
  searched before every model call in the loop.
- **Ground truth over claims.** `code` and `execution_result` come from the files on disk
  and a real run, never from what the model says it did.
- **A sandbox container per code run** rather than a subprocess on the host: a subprocess
  shares your user, files and network, and can read the agent's environment (including
  the API key) through `/proc`.
- **Minimal dependencies:** `anthropic`, `langgraph`, `chromadb`, `python-dotenv`, plus
  `pysqlite3-binary` only for hosts with an old SQLite; everything else is the standard
  library, and tests use `unittest`.

## Project layout

```text
agents/coder_agent.py   Coder Agent: config, result types, LangGraph loop, CLI
agents/memory.py        Sliding-window and Chroma memory
tools/                  read_file, write_file, exec_python (sandboxed), and the dispatcher
tests/                  unittest suite (offline tests plus one live smoke test)
specs/                  Specification, plan, research, contracts, and tasks
```

The Lab 1.1 starter files (`agent.py`, `memory.py`, `test_memory.py`) and the `openai`
dependency were retired in this increment; they can be recovered from the initial commit.
