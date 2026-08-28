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
[session 6f5a3c21a9b04d7e]
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
| Tool definition & local execution | `TOOL_SCHEMAS` + `ToolRunner` (explicit allowlist + schema validation) |
| Model output parsing | `LLMClient._parse` |
| Loop termination | final answer / `max_iterations` / repeat-detection guard |
| Error handling | tool error dicts fed back, HTTP retry, `AgentError`/`MaxIterationsExceeded` |
| Credentials via env / untracked config | `config.py` + workspace `config.json` (git-ignored) |
| **Session/workspace management** | `coding_agent/store.py` (sessions, branches, HEAD) |
| **Cache-friendly context** | constant system prompt + append-only history + compaction |
| **Security** | `coding_agent/security.py` (sandbox, env scrubbing, redaction) |
| **Subagents** | `coding_agent/subagent.py` (read-only, parallel exploration) |
| **Streaming** | `LLMClient.chat_stream` + live command output |
| **Agent Skills** | `coding_agent/skills.py` + `default_skills/` (preset skills) |

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
coding-agent run "task" [-M msg] # explicit; `run` with no args reads stdin
coding-agent status              # workspace / branch / HEAD
coding-agent log [--all] [--graph] [--oneline]   # session history (DAG)
coding-agent show <ref> [--all]  # view a session's conversation (last 20 msgs)
coding-agent switch <ref>        # roll back HEAD (no file changes)
coding-agent checkout <ref>      # roll back HEAD + restore work/ files
coding-agent branch [<name>]     # list / create a branch
coding-agent branch -d <name>    # delete a branch
coding-agent rm <ref>            # delete a session
coding-agent repl                # interactive (turns merge into ONE session)
coding-agent init                # initialize the workspace
```

```console
printf 'add a README section about testing' | coding-agent run
```

- `<ref>` is a branch name or a session id (full or unique prefix).
- `switch`/`checkout` roll HEAD back (like `git switch`/`git checkout`).
- `log --graph` renders the session DAG like `git log --graph` (`*` commits,
  `|` lines, `/` `\` forks/merges), with branch/HEAD markers.
- `repl` merges every turn into **one** session (sealed when you exit), rather
  than recording a session per turn. `/save` inside the REPL seals a
  checkpoint session and continues from it.
- A failed run (LLM error, `max_iterations`, repeat guard) is still sealed as
  a session with `status: "failed"` and the error recorded in `meta.json`, so
  the partial workdir/conversation is never lost.
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
`compact`, `max_tokens`, `temperature`, `timeout`, `command_timeout`, `sandbox`,
`env_allow`, `subagents`, `subagent_parallel`, `stream`, `skills`, `max_skills`,
`snapshot`, `allow_outside_workdir`, `allow_dangerous_commands`, `verbose`,
`quiet`, `minimal`. All boolean flags accept both forms, e.g. `--sandbox` /
`--no-sandbox`, `--stream` / `--no-stream`. Additional model-API HTTP headers
can be set with repeated `--header 'Name: value'` flags, the
`LLM_EXTRA_HEADERS` environment variable, or the `extra_headers` JSON object
in a config file.

### Progress & branch display

While a task runs, a concise progress stream is printed to stderr (the final
answer stays on stdout for scripting):

```
[run] branch main · model gpt-4o-mini · ~/.coding-agent/work
[step 1/40] list_files({"pattern": "*"})
  ok (count=0)
[step 2/40] write_file({"path": "greeting.txt", ...})
  ok (bytes_written=6)
[step 3/40] run_command("cat greeting.txt")
  ok (exit_code=0)
[session 530576fc9b04d7e2] on branch main
```

The current branch is shown in the run header/footer, in `status`, and in the
REPL prompt (`main>` / `(detached)>`). Use `--verbose` for full tool results and
interim assistant text, or `--quiet` to suppress the progress stream entirely.

### Security

- `--sandbox` — run every command inside a **network-less, read-only-root**
  sandbox via bwrap (bubblewrap) or firejail: no outbound network, root and
  `$HOME` are read-only, only the working directory is writable, and `/tmp` is
  ephemeral. **Fails closed** (with a clear error) if neither backend is
  installed.
- **Environment scrubbing** — child commands only receive a curated allowlist
  of environment variables, so `LLM_API_KEY` / `AWS_*` / tokens never leak via
  `env`. Add more with `--env-allow VAR1,VAR2`.
- **Secret redaction** — user task text, command output and file content are
  redacted before reaching the model or the session log (PEM private keys,
  `sk-…` keys, AWS keys, `Bearer …`, quoted `api_key="…"` assignments).
  Conservative, so ordinary source code is not mangled. Tool results report
  paths relative to the working directory, keeping local layout out of logs.
- **Tool allowlist + validation** — the model's tool calls dispatch through an
  explicit name→function map (never `getattr` on model input) and arguments are
  validated against each tool's JSON schema (unknown/missing/wrong-typed
  arguments are rejected).
- `--allow-outside-workdir` — let file tools touch paths outside `workdir`.
- `--allow-dangerous-commands` — allow the blocked destructive-command patterns
  (`rm -rf /`, `mkfs`, fork bombs, …).

`--workdir DIR` overrides the working directory (default `<workspace>/work`).
Artifact snapshots are only taken for the default workspace `work/` directory;
use `--no-snapshot` to disable them entirely.

> These are best-effort, defense-in-depth measures. The only real security
> boundary is a dedicated OS-level sandbox (container / VM) — run the agent
> inside one for untrusted work.

## Subagents, streaming & skills

**Subagents (parallel search)** — the model can call `parallel_search` to
delegate several independent exploration subtasks to isolated **read-only**
subagents (list/read/grep only, no writes or commands) that run in parallel and
return concise summaries. This keeps the main context clean and speeds up broad
investigation. Disable with `--no-subagents`, tune with `--subagent-parallel N`.

**Streaming** — assistant text streams to stdout live, and `run_command`
output streams to stderr in real time (instead of only appearing when the
command finishes). Disable with `--no-stream`.

**Agent Skills** — keyword-matched instruction bundles loaded on demand. A
skill is a markdown file with a small frontmatter header; at the start of a
session, the task is matched against each skill's keywords and matching skills'
instructions are injected (after the system prompt, keeping it constant for
cache). Built-in presets ship in `coding_agent/default_skills/`:

- `git-commit` — Conventional Commits and tidy git history
- `testing` — write/run tests, treat failures as truth
- `code-review` — review diffs for bugs/edge cases
- `security-review` — audit for injection, secrets, unsafe subprocess use
- `python-style` — PEP 8, type hints, linting
- `documentation` — README, docstrings, usage guides

Add your own in `<workspace>/skills/<name>.md` (a `---` frontmatter block with
`name` / `description` / `keywords` and a markdown body). Disable with
`--no-skills`, cap with `--max-skills N`.

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
├── llm.py      # OpenAI-compatible HTTP client, retry, streaming, parsing
├── tools.py    # tool JSON schemas + local executors + safety guards
├── security.py # sandbox, env scrubbing, secret redaction
├── subagent.py # read-only, parallel exploration subagents
├── skills.py   # agent skills: discovery + injection
├── default_skills/  # preset skills (git-commit, testing, code-review, …)
├── store.py    # workspace/session/branch/HEAD store (git-like DAG)
├── graph.py    # git-log --graph style DAG rendering
├── agent.py    # the loop + context compaction/trimming + termination
└── __main__.py # python -m coding_agent
```

### The agent loop (`Agent.run`)

1. Append the user's task to the (parent-session-seeded) message history.
2. Manage context: compact or trim if over the token budget.
3. Call the model with the conversation + `TOOL_SCHEMAS`.
4. Parse the reply; no `tool_calls` → final answer, stop (a
   `finish_reason="tool_calls"` with no calls is treated as an error).
5. Otherwise execute every requested tool locally and feed results back.
6. Repeat until a final answer, `max_iterations`, or the repeat-detection guard.

## Testing

No third-party runner — `unittest` plus a scriptable local mock of the OpenAI
endpoint:

```console
python -m unittest discover -s tests -v
```

124 tests cover every tool executor, config precedence, LLM parsing/streaming/retry, the
full agent loop end-to-end, context compaction, the session/branch store (DAG,
guards, artifacts), DAG graph rendering, and the CLI run/REPL flow (including failed-run recovery, /save checkpoints, stdin tasks, large-file paging, and bounded command capture).

## Limitations / ideas

- No vision, MCP, or multi-agent orchestration beyond the built-in read-only
  `parallel_search` subagents.
- Token estimation is a rough ~4 chars/token heuristic (tool schemas and a
  framing allowance are included in the budget).
- Artifact snapshots copy the workspace `work/` directory (skipping VCS/build
  dirs, symlinks, capped at 5000 files / 100 MiB) rather than diffing; custom
  external `--workdir` projects are not snapshotted.
- Context trimming/compaction uses heuristic turn boundaries, not a real
  tokenizer; very long single messages may still exceed a provider's limit.
- The session store uses atomic file replacement and a workspace file lock,
  but is not a multi-user/concurrent-transaction database.
