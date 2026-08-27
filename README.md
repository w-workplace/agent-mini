# coding-agent

A minimal, **dependency-free** coding agent: give it a software-engineering
task and it autonomously explores the codebase, edits files, runs commands to
verify its work, and iterates until done — in the spirit of Claude Code, Codex,
OpenCode, or DeepSeek Harness.

It talks to **any OpenAI-compatible chat-completions endpoint** using the
model's native **tool calling**, and executes every tool **locally in this
process**. It depends on **nothing but the Python standard library**: no agent
framework/SDK, no hosted code-execution or file APIs.

```console
$ python -m coding_agent "Add a --verbose flag to the CLI and add a unit test for it"
```

```
[assistant] step 1: calling list_files
[tool] list_files(...) -> {...}
[assistant] step 2: calling read_file
...
Done: added --verbose; verified with `python -m unittest tests`.
```

---

## Requirements checklist

| Requirement | Where it is satisfied |
| --- | --- |
| Interact with an LLM autonomously | `coding_agent/llm.py` + `coding_agent/agent.py` |
| Read/write files, run commands | `coding_agent/tools.py` (`read_file`, `write_file`, `edit_file`, `list_files`, `grep`, `run_command`) |
| Not a wrapper around an existing agent product | Everything is implemented from scratch |
| No agent framework/SDK | stdlib only (`urllib`, `json`, `subprocess`, …) — see `pyproject.toml` (`dependencies = []`) |
| Model vendor API client / OpenAI-compatible gateway allowed | `LLMClient` speaks the OpenAI chat-completions wire format over HTTPS |
| No hosted code-execution or file tools (Code Interpreter / Files API) | All tools execute locally in-process; nothing is delegated |
| Conversation history & context management | `Agent.messages` + `Agent._trim_context` (token budget, turn-aware trimming) |
| Tool definition & local execution | `TOOL_SCHEMAS` (JSON schemas) + `ToolRunner` |
| Model output parsing | `LLMClient._parse` (final text vs. `tool_calls`, malformed-argument handling) |
| Loop termination conditions | final answer (no tool calls), `max_iterations` cap, repeat-detection guard (`_is_stuck`) |
| Error handling | per-tool error dicts fed back to the model, HTTP retry with backoff, `AgentError`/`MaxIterationsExceeded` |
| Credentials via env / untracked config | `coding_agent/config.py`; `.gitignore` excludes `config.json` / `.coding-agent.json` / `.env` |

---

## Installation

No dependencies to install. Requires Python ≥ 3.9 (developed/tested on 3.12).

```console
# Run straight from the source tree
python -m coding_agent --help

# Optional: install a `coding-agent` console script
pip install -e .
```

## Configuration

Precedence (later wins):

1. built-in defaults
2. config file — project `.coding-agent.json`, then `~/.config/coding-agent/config.json`
3. environment variables
4. CLI flags

### API key

**Never** put your key in a committed file. Use an environment variable or an
untracked config file:

```console
export LLM_API_KEY=sk-...            # or OPENAI_API_KEY
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
```

`LLM_BASE_URL` / `OPENAI_BASE_URL` and `LLM_MODEL` / `OPENAI_MODEL` are also
recognized. Numeric options are exposed as `LLM_MAX_ITERATIONS`,
`LLM_TEMPERATURE`, `LLM_CONTEXT_LIMIT_TOKENS`, etc. (see
`coding_agent/config.py`).

Any OpenAI-compatible gateway works — e.g. OpenAI, DeepSeek
(`LLM_BASE_URL=https://api.deepseek.com`, `LLM_MODEL=deepseek-chat`), or a
local server such as vLLM / llama.cpp.

### Config file

Copy `config.example.json` to `.coding-agent.json` (project) or
`~/.config/coding-agent/config.json` (user), and set `api_key` there if you
prefer. Both are git-ignored.

## Usage

```console
# One-shot task
python -m coding_agent "Fix the failing unit test in tests/test_foo.py"

# Read the task from stdin
echo "refactor config.py to use dataclasses" | python -m coding_agent -

# Interactive REPL (keeps context across turns)
python -m coding_agent --interactive

# Useful flags
python -m coding_agent --workdir /path/to/project \
    --max-iterations 50 --context-limit-tokens 128000 --verbose "..."

python -m coding_agent --list-tools   # show the tool schemas
python -m coding_agent --version
```

### Safety flags

- `--workdir DIR` — the agent operates here; file tools are confined to it by
  default.
- `--allow-outside-workdir` — let file tools touch paths outside `--workdir`.
- `--allow-dangerous-commands` — allow commands matching the blocked
  destructive-command patterns (`rm -rf /`, `mkfs`, fork bombs, …).

> `run_command` executes with your user privileges, so run the agent in a
> sandbox/container for untrusted tasks. The danger-pattern blocklist is a
> best-effort guardrail, **not** a security boundary.

## Architecture

```
coding_agent/
├── cli.py      # argparse CLI + REPL
├── config.py   # defaults ← file ← env ← CLI precedence
├── llm.py      # OpenAI-compatible HTTP client, retry, response parsing
├── tools.py    # tool JSON schemas + local executors + safety guards
├── agent.py    # the loop: context mgmt, termination, error handling
└── __main__.py # python -m coding_agent
```

### The agent loop (`Agent.run`)

1. Append the user's task to `self.messages`.
2. **Trim** history to the token budget, turn-aware (never splits a tool call
   from its result).
3. Call the model with the conversation + `TOOL_SCHEMAS`.
4. **Parse** the reply. If there are no `tool_calls`, it's the final answer —
   stop.
5. Otherwise **execute** every requested tool locally and append each result as
   a `tool` message (keyed by `tool_call_id`).
6. Repeat until termination: a final answer, `max_iterations`, or the
   repeat-detection guard (the model made the same tool call ≥3 times in a row
   with no progress).

Tool failures are returned to the model as `{"ok": false, "error": ...}` so it
can diagnose and retry. Transient HTTP errors (429/5xx) are retried with
exponential backoff in `LLMClient.chat`.

## Testing

No third-party test runner needed — the suite uses `unittest` and a
scriptable local mock of the OpenAI endpoint.

```console
python -m unittest discover -s tests -v
```

The tests cover: every tool executor (including path-escape and
dangerous-command guards), config precedence, LLM parsing and retry, and the
full agent loop end-to-end against the fake server (tool call → local
execution → result → final answer), plus iteration caps and the stuck guard.

## Limitations / ideas

- No vision, no MCP, no sub-agents, no streaming output.
- `run_command` has no OS-level sandbox; pair it with a container for untrusted
  work.
- Token estimation is a rough ~4 chars/token heuristic.
- `max_tokens` is omitted by default (0) for maximum endpoint compatibility;
  reasoning models may ignore `temperature`/`max_tokens`.
