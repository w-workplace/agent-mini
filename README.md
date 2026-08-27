# coding-agent

A minimal, **dependency-free** coding agent with **git-like session
management**: give it a software-engineering task and it autonomously explores
the codebase, edits files, runs commands to verify, and iterates until done —
in the spirit of Claude Code, Codex, OpenCode, or DeepSeek Harness.

It talks to **any OpenAI-compatible chat-completions endpoint** using the
model's native **tool calling**, and executes every tool **locally in this
process**. It depends on **nothing but the Python standard library**: no agent
framework/SDK, no hosted code-execution or file APIs.

```console
$ python -m coding_agent "Add a --verbose flag to the CLI and a unit test for it"
```

```
created ...
[session 6f5a3c21]
```

Every run is recorded as an immutable **session** (a git-like "commit"), stored
in a dedicated workspace folder, with branches and rollback.

---

## Requirements checklist

| Requirement | Where it is satisfied |
| --- | --- |
| Interact with an LLM autonomously | `coding_agent/llm.py` + `coding_agent/agent.py` |
| Read/write files, run commands | `coding_agent/tools.py` |
| Not a wrapper around an existing agent product | Everything is implemented from scratch |
| No agent framework/SDK | stdlib only (`urllib`, `json`, `subprocess`, …) — `dependencies = []` |
| Model vendor API / OpenAI-compatible gateway | `LLMClient` speaks the OpenAI chat-completions wire format |
| No hosted code-execution / file tools | All tools execute locally in-process |
| Conversation history & context management | `Agent.messages`, turn-aware compaction/trimming |
| Tool definition & local execution | `TOOL_SCHEMAS` + `ToolRunner` |
| Model output parsing | `LLMClient._parse` |
| Loop termination | final answer / `max_iterations` / repeat-detection guard |
| Error handling | tool error dicts fed back, HTTP retry, `AgentError`/`MaxIterationsExceeded` |
| Credentials via env / untracked config | `config.py` + workspace `config.json` (git-ignored) |
| **Session/workspace management** | `coding_agent/store.py` (sessions, branches, HEAD) |
| **Cache-friendly context** | constant system prompt + append-only history + compaction |

---

## Installation

No dependencies. Python ≥ 3.9 (tested on 3.12).

```console
python -m coding_agent --help          # run straight from the source tree
pip install -e .                       # optional: `coding-agent` console script
```

## Quick start

```console
export LLM_API_KEY=sk-...             # or OPENAI_API_KEY
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini

python -m coding_agent "create a hello-world CLI and test it"
python -m coding_agent status          # workspace / branch / HEAD
python -m coding_agent log             # session history
```

## Session management (git-like)

The agent keeps a **workspace** (default `~/.coding-agent`, override with
`--workspace` / `LLM_WORKSPACE`) that holds everything:

```
~/.coding-agent/
    config.json            # workspace config (untracked; may hold api_key)
    work/                  # the agent's working directory (stable path)
    sessions/<id>/
        meta.json          # id, parent, task, message, model, timestamps
        conversation.jsonl # the session log (one JSON message per line)
        artifacts/         # snapshot of work/ when the session was sealed
    refs/heads/<branch>    # named pointer -> session id
    refs/HEAD              # -> "refs/heads/<branch>" or a bare session id
```

Sessions form a **tree** via `parent` (like commits); branches are named
pointers to sessions; `HEAD` is the current session. Commands are short and
memorable:

```console
coding-agent "task"              # run a task (new session)          [= run]
coding-agent run "task" [-M msg] # explicit
coding-agent status              # workspace / branch / HEAD
coding-agent log [--all] [--oneline]   # session history (current branch)
coding-agent show <ref>          # view a session's conversation
coding-agent switch <ref>        # roll back HEAD (no file changes)
coding-agent checkout <ref>      # roll back HEAD + restore work/ files
coding-agent branch [<name>]     # list / create a branch
coding-agent branch -d <name>    # delete a branch
coding-agent rm <ref>            # delete a session
coding-agent repl                # interactive (each turn = a session)
coding-agent init                # initialize the workspace
```

- `<ref>` is a branch name or a session id (full or unique prefix).
- `switch`/`checkout` roll HEAD back (like `git switch`/`git checkout`).
- Guards keep the DAG safe: you can't delete the current session, a branch
  tip, or a session that other sessions descend from.
- `checkout <ref>` also restores `work/` from that session's artifact snapshot
  (it **replaces** `work/` — like git checking out files). Use
  `checkout <ref> --no-restore` to only move HEAD.

## Configuration

Precedence (later wins): defaults → project `.coding-agent.json` → workspace
`config.json` → environment variables → CLI flags.

```console
export LLM_API_KEY=sk-...            # or OPENAI_API_KEY
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
export LLM_WORKSPACE=~/.coding-agent
```

Any OpenAI-compatible gateway works — OpenAI, DeepSeek
(`LLM_BASE_URL=https://api.deepseek.com`, `LLM_MODEL=deepseek-chat`), or a
local server (vLLM / llama.cpp). Credentials go in an environment variable or
the untracked workspace `config.json`; never commit them.

Key options (env `LLM_<UPPER>` / CLI): `max_iterations`, `context_limit_tokens`,
`compact`, `max_tokens`, `temperature`, `command_timeout`,
`allow_outside_workdir`, `allow_dangerous_commands`, `verbose`.

### Safety flags

- `--workdir DIR` — override the working directory (default `<workspace>/work`).
- `--allow-outside-workdir` — let file tools touch paths outside `workdir`.
- `--allow-dangerous-commands` — allow the blocked destructive-command patterns
  (`rm -rf /`, `mkfs`, fork bombs, …).

> `run_command` executes with your user privileges. The danger-pattern
> blocklist is a best-effort guardrail, **not** a security boundary — run the
> agent in a container for untrusted work.

## Cache-friendly context management

Prompt caches (Anthropic prompt caching, OpenAI/DeepSeek automatic prefix
caching) only hit when a request's **prefix** is byte-identical. The agent
follows three rules to keep hit rates high:

1. **Constant system prompt** — no session id, branch, timestamp, or absolute
   working-directory interpolation (the model is told it works in a dedicated
   directory and should use relative paths; `ToolRunner` resolves them).
2. **Append-only history** — continuing a session re-sends the parent session's
   messages verbatim as the prefix and only appends new turns; the parent
   chain never rewrites history.
3. **Compaction, not front-drop** — when the token budget is exceeded,
   `--compact` summarizes the oldest turns into one stable block placed right
   after the system prompt (`[Prior conversation summary] …`). That keeps the
   prefix stable instead of evicting the cache by dropping the front. Without
   `--compact` it falls back to dropping the oldest turns.

## Architecture

```
coding_agent/
├── cli.py      # argparse CLI + git-like subcommands + REPL
├── config.py   # defaults ← project config ← workspace config ← env ← CLI
├── llm.py      # OpenAI-compatible HTTP client, retry, response parsing
├── tools.py    # tool JSON schemas + local executors + safety guards
├── store.py    # workspace/session/branch/HEAD store (git-like DAG)
├── agent.py    # the loop + context compaction/trimming + termination
└── __main__.py # python -m coding_agent
```

### The agent loop (`Agent.run`)

1. Append the user's task to the (parent-session-seeded) message history.
2. Manage context: compact or trim if over the token budget.
3. Call the model with the conversation + `TOOL_SCHEMAS`.
4. Parse the reply; no `tool_calls` → final answer, stop.
5. Otherwise execute every requested tool locally and feed results back.
6. Repeat until a final answer, `max_iterations`, or the repeat-detection guard.

## Testing

No third-party runner — `unittest` plus a scriptable local mock of the OpenAI
endpoint:

```console
python -m unittest discover -s tests -v
```

63 tests cover every tool executor, config precedence, LLM parsing/retry, the
full agent loop end-to-end, context compaction, the session/branch store (DAG,
guards, artifacts), and the CLI run flow.

## Limitations / ideas

- No vision, MCP, sub-agents, or streaming output; no OS-level sandbox for
  `run_command`.
- Token estimation is a rough ~4 chars/token heuristic.
- Artifact snapshots copy the working directory (skipping VCS/build dirs,
  capped at 100 MiB) rather than diffing; for large external projects prefer
  `--workdir` with care, since `checkout` only restores `work/`.
