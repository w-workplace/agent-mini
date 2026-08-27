"""The agent loop: context management, tool execution, parsing and termination.

This is the heart of the agent. It maintains the conversation history, calls
the model, parses its reply (final text vs. tool calls), executes any tool
calls locally, feeds results back, and decides when to stop — either because
the model produced a final answer, or because a safety guard (iteration cap or
repeat-detection) fired.
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


class AgentError(Exception):
    """A fatal, unrecoverable agent failure."""


class MaxIterationsExceeded(AgentError):
    """Raised when the loop hits the iteration cap without a final answer."""


class Agent:
    def __init__(self, config: Any, llm: LLMClient | None = None, tools: ToolRunner | None = None):
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
            workdir=config.workdir,
            allow_outside_workdir=config.allow_outside_workdir,
            allow_dangerous_commands=config.allow_dangerous_commands,
            command_timeout=config.command_timeout,
        )
        self.messages: list[dict[str, Any]] = []
        self._init_system_prompt()

    def _init_system_prompt(self) -> None:
        base = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT
        base += f"\n\nWorking directory: {self.config.workdir}"
        self.messages.append({"role": "system", "content": base})

    # -- public API ----------------------------------------------------------
    def run(self, task: str) -> str:
        """Run the agent on one task; returns the model's final answer.

        History accumulates across calls, so the same instance can be reused
        for a multi-turn REPL session.
        """
        self.messages.append({"role": "user", "content": task})
        signatures: list[tuple[tuple[str, str], ...]] = []
        final_answer = ""

        for step in range(1, self.config.max_iterations + 1):
            self.messages = self._trim_context(self.messages)
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

            # Execute every tool call in this assistant message (supports
            # parallel tool calls), and record the results.
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

    def _trim_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop the oldest conversation *turns* once the token budget is exceeded.

        A turn is a user message, or an assistant message together with the
        tool messages that follow it — so we never split a tool call from its
        result. The system message is always kept.
        """
        limit = self.config.context_limit_tokens
        if limit <= 0 or self._estimate_tokens(messages) <= limit:
            return messages

        system = messages[0] if messages and messages[0]["role"] == "system" else None
        rest = messages[1:] if system else messages

        turns: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for msg in rest:
            if msg["role"] in ("user", "assistant"):
                if current:
                    turns.append(current)
                current = [msg]
            else:
                current.append(msg)  # tool messages attach to the assistant above
        if current:
            turns.append(current)

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
