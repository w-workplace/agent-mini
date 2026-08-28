"""The agent loop: context management, tool execution, parsing and termination.

This is the heart of the agent. It maintains the conversation history, calls
the model, parses its reply (final text vs. tool calls), executes any tool
calls locally, feeds results back, and decides when to stop.

Cache-friendly context management
---------------------------------
Prompt caches (Anthropic prompt caching, OpenAI/DeepSeek automatic prefix
caching) hit only when a request's *prefix* is unchanged. To keep cache hit
rates high across sessions and turns we follow three rules:

1. The system prompt is a **constant** — no per-session id, branch, timestamp
   or working-directory interpolation.
2. History is **append-only**: continuing a session re-sends the parent
   session's messages verbatim as the prefix and only appends new turns.
3. When the token budget is exceeded we **compact** (summarize the oldest
   turns into one stable block right after the system prompt) rather than
   dropping from the front — dropping the front would rewrite the prefix and
   evict the cache.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .llm import LLMClient, LLMError
from .rules import load_rules
from .security import redact
from .skills import discover_skills, load_skills, skill_prompt
from .tools import ToolRunner, format_tool_result

_READ_ONLY_TOOLS = frozenset({"list_files", "read_file", "grep"})

DEFAULT_SYSTEM_PROMPT = """\
You are a coding agent: an autonomous software-engineering assistant.

You work in a loop: explore the codebase, make changes, run commands to verify
them, and keep going until the user's task is done. You have tools for listing
files, reading/writing/editing files, searching with grep, and running shell
commands.

You operate inside a dedicated working directory. File tools and shell commands
resolve relative paths against that directory, so use relative paths (for
example `src/main.py`, `tests/`) rather than absolute paths.

Guidelines:
- Explore first: use list_files, grep and read_file to understand the code
  before changing it.
- Batch independent reads: when you need several unrelated files or searches,
  request all of them in ONE response as multiple read-only tool calls; the
  agent executes those in parallel, which is much faster than one per turn.
- Prefer edit_file for small, targeted changes (it replaces one exact string);
  use write_file to create a new file or rewrite one entirely.
- Verify your work: after changing code, run the project's tests/build (or
  otherwise check correctness) with run_command, and fix any failures.
- If a command or tool returns an error, read it, diagnose the cause, and fix
  it yourself rather than guessing.
- When the task is complete, stop calling tools and reply with a concise
  summary of what you changed and how you verified it.
"""

SUMMARY_MARKER = "[Prior conversation summary]"
SUMMARIZER_SYSTEM = (
    "You summarize conversations for a coding agent. Produce one concise but "
    "information-dense summary that preserves all important facts, decisions, "
    "file paths, function/class names, commands run, error messages, and open "
    "questions, so that future turns can continue seamlessly. Output only the "
    "summary."
)


class AgentError(Exception):
    """A fatal, unrecoverable agent failure."""


class MaxIterationsExceeded(AgentError):
    """Raised when the loop hits the iteration cap without a final answer."""


class Agent:
    def __init__(
        self,
        config: Any,
        llm: LLMClient | None = None,
        tools: ToolRunner | None = None,
        history: list[dict[str, Any]] | None = None,
    ):
        self.config = config
        self.llm = llm or LLMClient(
            config.base_url,
            config.api_key,
            config.model,
            timeout=config.timeout,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            retries=config.request_retries,
            verbose=config.verbose,
        )
        self.tools = tools or ToolRunner(
            workdir=config.workdir or ".",
            allow_outside_workdir=config.allow_outside_workdir,
            allow_dangerous_commands=config.allow_dangerous_commands,
            command_timeout=config.command_timeout,
            sandbox=config.sandbox,
            env_allow=config.env_allow,
            stream=config.stream,
            quiet=config.quiet,
        )
        # Wire subagent support: adds the `parallel_search` tool when enabled.
        if config.subagents and not getattr(self.tools, "read_only", False):
            self.tools.set_subagent_executor(self._run_subagents)
        self.tool_schemas = self.tools.tool_schemas
        self._tool_token_estimate: int | None = None
        self.usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        self.changed_files: set[str] = set()
        self._duration_ms = 0.0
        self.messages: list[dict[str, Any]] = []
        self._init_system_prompt()
        # Skills load on the first turn of a fresh session (not when continuing
        # a parent session, whose history already contains any injected skills).
        self.skills = load_skills(config.workspace) if config.skills else []
        self._skills_pending = bool(config.skills) and not history
        if history:
            # `history` is the parent session's messages *without* the system
            # prompt, so the reconstructed prefix is [system] + history and the
            # system prompt stays byte-identical across sessions (cache hit).
            self.messages.extend(history)

    def _run_subagents(self, queries: list[str]) -> list[dict[str, Any]]:
        from .subagent import run_subagents
        return run_subagents(
            self.llm, self.config, str(self.tools.workdir), queries,
            max_parallel=self.config.subagent_parallel,
        )

    def _stream_delta(self, text: str) -> None:
        """Write streamed assistant text to stdout (live final answer)."""
        sys.stdout.write(text)
        sys.stdout.flush()

    def _init_system_prompt(self) -> None:
        base = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT
        rules = load_rules(str(self.tools.workdir), getattr(self.config, "rules", ""))
        if rules:
            base = base.rstrip() + "\n\n" + rules
        self.messages.append({"role": "system", "content": base})

    # -- public API ----------------------------------------------------------
    def run(self, task: str) -> str:
        """Run the agent on one task; returns the model's final answer."""
        started = time.monotonic()
        self._duration_ms = 0.0
        safe_task = redact(task)
        if safe_task != task:
            self._progress("[security] redacted potential secrets from the task")
            task = safe_task
        if self._skills_pending:
            self._skills_pending = False
            for skill in discover_skills(task, self.skills, self.config.max_skills):
                self.messages.append({"role": "user", "content": skill_prompt(skill)})
                self._progress(f"[skill] loaded {skill.name}")
        self.messages.append({"role": "user", "content": task})
        signatures: list[tuple[tuple[str, str, str], ...]] = []
        final_answer = ""

        for step in range(1, self.config.max_iterations + 1):
            self.messages = self._manage_context(self.messages)
            llm_started = time.monotonic()
            try:
                reply = self._call_llm()
            except LLMError as exc:
                self._duration_ms = (time.monotonic() - started) * 1000
                raise AgentError(f"LLM call failed: {exc.message}") from exc
            llm_ms = (time.monotonic() - llm_started) * 1000
            self._record_usage(getattr(reply, "usage", {}) or {})

            self.messages.append(reply.to_api_dict())
            self._show_step(step, reply)

            if not reply.tool_calls:
                if getattr(reply, "finish_reason", "") == "tool_calls":
                    self._duration_ms = (time.monotonic() - started) * 1000
                    raise AgentError(
                        "the model returned finish_reason='tool_calls' without "
                        "any tool calls; refusing to treat that as a final answer"
                    )
                final_answer = reply.content or "(the model returned no text)"
                self._progress(
                    f"[answer] llm {llm_ms / 1000:.2f}s · "
                    f"total {self._duration_ms / 1000:.2f}s"
                )
                break

            signature: list[tuple[str, str, str]] = []
            calls = reply.tool_calls
            parallel_results: dict[int, dict[str, Any]] = {}
            tool_started = time.monotonic()
            if len(calls) > 1 and all(tc.name in _READ_ONLY_TOOLS for tc in calls):
                parallel_results = self._run_read_only_parallel(calls)
            for idx, tc in enumerate(calls):
                self._show_tool_start(step, tc)
                if idx in parallel_results:
                    result = parallel_results[idx]
                else:
                    result = self.tools.execute(tc.name, tc.arguments)
                fingerprint = self._tool_result_fingerprint(result)
                signature.append(
                    (
                        tc.name,
                        json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False),
                        fingerprint,
                    )
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": format_tool_result(result),
                    }
                )
                self._record_changed_file(tc.name, result)
                self._show_tool_end(result, (time.monotonic() - tool_started) * 1000, llm_ms)

            signatures.append(tuple(signature))
            if self._is_stuck(signatures):
                self._duration_ms = (time.monotonic() - started) * 1000
                raise AgentError(
                    "the model repeated the same tool call with no observable "
                    "progress; aborting to avoid an infinite loop"
                )
        else:
            self._duration_ms = (time.monotonic() - started) * 1000
            raise MaxIterationsExceeded(
                f"reached max_iterations ({self.config.max_iterations}) "
                "without a final answer"
            )

        self._duration_ms = (time.monotonic() - started) * 1000
        return final_answer

    def _run_read_only_parallel(
        self, calls: list[Any]
    ) -> dict[int, dict[str, Any]]:
        def run_one(item: tuple[int, Any]) -> tuple[int, dict[str, Any]]:
            idx, tc = item
            return idx, self.tools.execute(tc.name, tc.arguments)

        with ThreadPoolExecutor(max_workers=min(8, len(calls))) as ex:
            return dict(ex.map(run_one, enumerate(calls)))

    # -- stats ---------------------------------------------------------------
    @property
    def duration_ms(self) -> float:
        return self._duration_ms

    @property
    def total_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0) + self.usage.get("completion_tokens", 0)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "duration_ms": round(self._duration_ms, 1),
            "prompt_tokens": self.usage.get("prompt_tokens", 0),
            "completion_tokens": self.usage.get("completion_tokens", 0),
            "total_tokens": self.total_tokens,
            "changed_files": sorted(self.changed_files),
        }

    def _record_usage(self, usage: dict[str, Any]) -> None:
        try:
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        except (TypeError, ValueError):
            pass

    def _record_changed_file(self, tool_name: str, result: dict[str, Any]) -> None:
        if (
            tool_name in ("write_file", "edit_file")
            and result.get("ok")
            and result.get("diff")
        ):
            path = result.get("path")
            if path:
                self.changed_files.add(path)

    def compact_now(self) -> tuple[int, int]:
        """Force a compaction pass now (used by the REPL ``/compact``)."""
        before = len(self.messages)
        if before <= 1:
            return before, before
        self.messages = self._compact(self.messages)
        return before, len(self.messages)

    def _call_llm(self) -> Any:
        if getattr(self.config, "stream", False):
            return self.llm.chat_stream(self.messages, self.tool_schemas, on_text=self._stream_delta)
        return self.llm.chat(self.messages, self.tool_schemas)

    # -- progress output ------------------------------------------------------
    def _progress(self, message: str) -> None:
        """Concise per-step progress, shown unless ``quiet``."""
        if not getattr(self.config, "quiet", False):
            print(message, file=sys.stderr, flush=True)

    def _detail(self, message: str) -> None:
        """Verbose detail, shown only with ``--verbose``."""
        if not getattr(self.config, "quiet", False) and self.config.verbose:
            print(message, file=sys.stderr, flush=True)

    @staticmethod
    def _fmt_args(arguments: dict[str, Any]) -> str:
        s = json.dumps(arguments, ensure_ascii=False)
        return s if len(s) <= 120 else s[:120] + "…"

    @staticmethod
    def _result_summary(result: dict[str, Any]) -> str:
        if result.get("ok"):
            for key in ("exit_code", "bytes_written", "replacements", "count", "lines_returned"):
                if key in result:
                    return f"ok ({key}={result[key]})"
            return "ok"
        err = str(result.get("error", "error")).replace("\n", " ")[:120]
        return f"error: {err}"

    def _show_step(self, step: int, reply: Any) -> None:
        if reply.content and reply.tool_calls:
            self._detail(
                f"[assistant] {reply.content.strip().replace(chr(10), ' ')[:200]}"
            )

    def _show_tool_start(self, step: int, tc: Any) -> None:
        self._progress(
            f"[step {step}/{self.config.max_iterations}] "
            f"{tc.name}({self._fmt_args(tc.arguments)})"
        )

    def _show_tool_end(
        self,
        result: dict[str, Any],
        tool_ms: float | None = None,
        llm_ms: float | None = None,
    ) -> None:
        summary = self._result_summary(result)
        timing = ""
        if tool_ms is not None:
            timing += f" · tool {tool_ms / 1000:.2f}s"
        if llm_ms is not None:
            timing += f" · llm {llm_ms / 1000:.2f}s"
        self._progress(f"  {summary}{timing}")
        diff = result.get("diff") or ""
        if diff:
            path = result.get("path") or ""
            added = result.get("added_lines", 0)
            removed = result.get("removed_lines", 0)
            self._progress(f"  ~ {path} +{added} -{removed}")
            if self.config.verbose:
                for line in diff.splitlines()[:40]:
                    self._progress(f"    {line}")

    # -- context management ---------------------------------------------------
    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Rough token estimate: ~4 chars/token, counting tool arguments too."""
        total = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                total += max(1, len(content) // 4)
            for tc in msg.get("tool_calls") or []:
                total += max(1, len(json.dumps(tc, ensure_ascii=False)) // 4)
        return total

    def _estimate_tool_tokens(self) -> int:
        """Estimate tokens consumed by the tool schemas sent with every call.

        Cached: the tool set is fixed after ``Agent.__init__``, so serializing
        the schemas on every iteration would only waste CPU.
        """
        if self._tool_token_estimate is None:
            schemas = json.dumps(self.tool_schemas, ensure_ascii=False)
            # Add a modest per-request framing allowance so the estimate does
            # not undercount the actual wire payload.
            self._tool_token_estimate = max(0, len(schemas) // 4) + 32
        return self._tool_token_estimate

    def _manage_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        limit = self.config.context_limit_tokens
        if limit <= 0:
            return messages
        effective_limit = max(1, limit - self._estimate_tool_tokens())
        if self._estimate_tokens(messages) <= effective_limit:
            return messages
        if getattr(self.config, "compact", False):
            try:
                return self._compact(messages, limit=effective_limit)
            except Exception:  # noqa: BLE001 — fall back to dropping oldest
                return self._trim_context(messages, limit=effective_limit)
        return self._trim_context(messages, limit=effective_limit)

    @staticmethod
    def _split_turns(rest: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Split messages (after the system prompt) into conversation turns.

        A turn is a user message, or an assistant message together with the
        tool messages that follow it — so we never split a tool call from its
        result.
        """
        turns: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for msg in rest:
            if msg["role"] in ("user", "assistant"):
                if current:
                    turns.append(current)
                current = [msg]
            else:
                current.append(msg)
        if current:
            turns.append(current)
        return turns

    @staticmethod
    def _flatten_turns(turns: list[list[dict[str, Any]]]) -> str:
        parts: list[str] = []
        for turn in turns:
            for msg in turn:
                role = msg["role"]
                content = msg.get("content") or ""
                tcs = msg.get("tool_calls") or []
                if role == "tool":
                    parts.append(f"[tool result] {content}")
                elif tcs:
                    calls = ", ".join(
                        f"{t['function']['name']}({t['function']['arguments']})" for t in tcs
                    )
                    parts.append(f"[assistant tool calls] {calls}")
                    if content:
                        parts.append(content)
                else:
                    parts.append(f"[{role}] {content}")
        return "\n".join(parts)

    def _summarize_turns(self, prior: str, turns: list[list[dict[str, Any]]]) -> str:
        text = self._flatten_turns(turns)
        if prior:
            prompt = (
                f"{prior}\n\n"
                f"Additional conversation to fold into the summary:\n\n{text}\n\n"
                "Return a single consolidated summary with the same level of detail."
            )
        else:
            prompt = (
                "Summarize the following conversation so that a future turn can "
                f"continue seamlessly:\n\n{text}\n\nReturn the summary."
            )
        reply = self.llm.chat(
            [{"role": "system", "content": SUMMARIZER_SYSTEM},
             {"role": "user", "content": prompt}]
        )
        return SUMMARY_MARKER + "\n" + (reply.content or "")

    def _compact(
        self,
        messages: list[dict[str, Any]],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fold the oldest turns into one stable summary block.

        The latest turn is always preserved verbatim; older turns are folded
        into the summary.  The summary sits immediately after the system
        prompt, so the prefix ``[system][summary]`` stays stable across turns.
        """
        if limit is None:
            limit = self.config.context_limit_tokens
        system = messages[0] if messages and messages[0]["role"] == "system" else None
        rest = messages[1:] if system else messages
        turns = self._split_turns(rest)

        summary_text = ""
        if turns and (turns[0][0].get("content") or "").startswith(SUMMARY_MARKER):
            summary_text = turns[0][0]["content"]
            turns = turns[1:]

        if not turns:
            out = [system] if system else []
            return out

        system_tokens = self._estimate_tokens([system]) if system else 0
        reserve = max(1, (limit - system_tokens) // 2)

        latest = turns[-1]
        keep: list[list[dict[str, Any]]] = [latest]
        used = self._estimate_tokens(latest)
        fold: list[list[dict[str, Any]]] = []
        for turn in reversed(turns[:-1]):
            t = self._estimate_tokens(turn)
            if used + t <= reserve:
                keep.insert(0, turn)
                used += t
            else:
                fold.insert(0, turn)

        if fold:
            summary_text = self._summarize_turns(summary_text, fold)

        out: list[dict[str, Any]] = []
        if system:
            out.append(system)
        if summary_text:
            out.append({"role": "user", "content": summary_text})
        for turn in keep:
            out.extend(turn)

        if self._estimate_tokens(out) > limit:
            return self._trim_context(messages, limit=limit)
        if len(out) < len(messages):
            self._progress(
                f"[context] compacted {len(messages) - len(out)} messages into "
                "a summary to fit the token budget"
            )
        return out

    def _trim_context(
        self,
        messages: list[dict[str, Any]],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fallback: drop the oldest turns (not cache-optimal, but always works).

        The newest turn is always retained, even if it alone exceeds the
        budget; dropping it would send a request without the user's task.
        """
        if limit is None:
            limit = self.config.context_limit_tokens
        system = messages[0] if messages and messages[0]["role"] == "system" else None
        rest = messages[1:] if system else messages
        turns = self._split_turns(rest)
        if not turns:
            return messages

        budget = limit - (self._estimate_tokens([system]) if system else 0)
        latest = turns[-1]
        keep: list[list[dict[str, Any]]] = [latest]
        used = self._estimate_tokens(latest)
        for turn in reversed(turns[:-1]):
            t = self._estimate_tokens(turn)
            if used + t > budget:
                break
            keep.insert(0, turn)
            used += t

        out: list[dict[str, Any]] = []
        if system:
            out.append(system)
        for turn in keep:
            out.extend(turn)
        if len(out) < len(messages):
            self._progress(
                f"[context] trimmed {len(messages) - len(out)} messages to fit "
                f"the {limit}-token budget"
            )
        if self._estimate_tokens(out) > limit:
            self._progress(
                "[context] warning: newest turn alone exceeds the token budget"
            )
        return out

    # -- loop guard -----------------------------------------------------------
    @staticmethod
    def _is_stuck(
        signatures: list[tuple[tuple[str, str, str], ...]],
        window: int = 3,
    ) -> bool:
        """True when the last `window` tool calls are identical *and* produced
        identical results.

        ``signatures`` entries are ``(tool_name, arguments_json, result_fingerprint)``.
        Older callers may pass 2-tuples; in that case only name+arguments are
        compared (backward-compatible with the previous behaviour).
        """
        if len(signatures) < window:
            return False
        recent = signatures[-window:]
        first = recent[0]
        for sig in recent[1:]:
            if sig[:2] != first[:2]:
                return False
            if len(sig) > 2 and len(first) > 2 and sig[2] != first[2]:
                return False
        return True

    @staticmethod
    def _tool_result_fingerprint(result: dict[str, Any]) -> str:
        """Stable-ish fingerprint of a tool result for the loop guard."""
        try:
            compact = json.dumps(result, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            compact = repr(result)
        return compact[:512]
