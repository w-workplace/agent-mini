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
from typing import Any

from .llm import LLMClient, LLMError
from .tools import TOOL_SCHEMAS, ToolRunner, format_tool_result

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
        )
        self.messages: list[dict[str, Any]] = []
        self._init_system_prompt()
        if history:
            # `history` is the parent session's messages *without* the system
            # prompt, so the reconstructed prefix is [system] + history and the
            # system prompt stays byte-identical across sessions (cache hit).
            self.messages.extend(history)

    def _init_system_prompt(self) -> None:
        base = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT
        self.messages.append({"role": "system", "content": base})

    # -- public API ----------------------------------------------------------
    def run(self, task: str) -> str:
        """Run the agent on one task; returns the model's final answer."""
        self.messages.append({"role": "user", "content": task})
        signatures: list[tuple[tuple[str, str], ...]] = []
        final_answer = ""

        for step in range(1, self.config.max_iterations + 1):
            self.messages = self._manage_context(self.messages)
            try:
                reply = self.llm.chat(self.messages, TOOL_SCHEMAS)
            except LLMError as exc:
                raise AgentError(f"LLM call failed: {exc.message}") from exc

            self.messages.append(reply.to_api_dict())
            if self.config.verbose:
                self._log_step(step, reply)

            if not reply.tool_calls:
                final_answer = reply.content or "(the model returned no text)"
                break

            signature: list[tuple[str, str]] = []
            for tc in reply.tool_calls:
                signature.append((tc.name, json.dumps(tc.arguments, sort_keys=True)))
                result = self.tools.execute(tc.name, tc.arguments)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": format_tool_result(result),
                    }
                )
                if self.config.verbose:
                    print(
                        f"[tool] {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)}) "
                        f"-> {format_tool_result(result)[:600]}",
                        file=sys.stderr,
                    )

            signatures.append(tuple(signature))
            if self._is_stuck(signatures):
                raise AgentError(
                    "the model repeated the same tool call without making "
                    "progress; aborting to avoid an infinite loop"
                )
        else:
            raise MaxIterationsExceeded(
                f"reached max_iterations ({self.config.max_iterations}) "
                "without a final answer"
            )

        return final_answer

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

    def _manage_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        limit = self.config.context_limit_tokens
        if limit <= 0 or self._estimate_tokens(messages) <= limit:
            return messages
        if getattr(self.config, "compact", False):
            try:
                return self._compact(messages)
            except Exception:  # noqa: BLE001 — fall back to dropping oldest
                return self._trim_context(messages)
        return self._trim_context(messages)

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

    def _compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fold the oldest turns into one stable summary block.

        The summary sits immediately after the system prompt, so the prefix
        ``[system][summary]`` stays stable across subsequent turns (and across
        sessions that continue this one) until the recent window itself
        overflows and the summary is regenerated.
        """
        limit = self.config.context_limit_tokens
        system = messages[0] if messages and messages[0]["role"] == "system" else None
        rest = messages[1:] if system else messages
        turns = self._split_turns(rest)

        summary_text = ""
        if turns and (turns[0][0].get("content") or "").startswith(SUMMARY_MARKER):
            summary_text = turns[0][0]["content"]
            turns = turns[1:]

        system_tokens = self._estimate_tokens([system]) if system else 0
        reserve = max(1, (limit - system_tokens) // 2)

        keep: list[list[dict[str, Any]]] = []
        fold: list[list[dict[str, Any]]] = []
        used = 0
        overflow = False
        for turn in reversed(turns):
            t = self._estimate_tokens(turn)
            if not overflow and used + t <= reserve:
                keep.insert(0, turn)
                used += t
            else:
                overflow = True
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
            return self._trim_context(messages)
        if self.config.verbose and len(out) < len(messages):
            print(
                f"[context] compacted {len(messages) - len(out)} messages into "
                "a summary to fit the token budget",
                file=sys.stderr,
            )
        return out

    def _trim_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fallback: drop the oldest turns (not cache-optimal, but always works)."""
        limit = self.config.context_limit_tokens
        system = messages[0] if messages and messages[0]["role"] == "system" else None
        rest = messages[1:] if system else messages
        turns = self._split_turns(rest)

        budget = limit - (self._estimate_tokens([system]) if system else 0)
        keep: list[list[dict[str, Any]]] = []
        used = 0
        for turn in reversed(turns):
            t = self._estimate_tokens(turn)
            if used + t > budget:
                break
            keep.append(turn)
            used += t
        keep.reverse()

        out: list[dict[str, Any]] = []
        if system:
            out.append(system)
        for turn in keep:
            out.extend(turn)
        if self.config.verbose and len(out) < len(messages):
            print(
                f"[context] trimmed {len(messages) - len(out)} messages to fit "
                f"the {limit}-token budget",
                file=sys.stderr,
            )
        return out

    # -- loop guard -----------------------------------------------------------
    @staticmethod
    def _is_stuck(signatures: list[tuple[tuple[str, str], ...]], window: int = 3) -> bool:
        """True when the last `window` tool-call signatures are identical."""
        if len(signatures) < window:
            return False
        recent = signatures[-window:]
        return all(s == recent[0] for s in recent)

    # -- logging --------------------------------------------------------------
    @staticmethod
    def _log_step(step: int, reply: Any) -> None:
        if reply.content:
            print(f"[assistant] {reply.content}", file=sys.stderr)
        if reply.tool_calls:
            names = ", ".join(tc.name for tc in reply.tool_calls)
            print(f"[assistant] step {step}: calling {names}", file=sys.stderr)
