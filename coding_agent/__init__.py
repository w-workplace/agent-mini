"""A minimal, dependency-free coding agent.

``coding_agent`` is an autonomous software-engineering assistant that talks to
any OpenAI-compatible chat-completions API using the model's native tool
calling, and executes a small set of local tools (read / write / edit files,
grep, list files, run shell commands) entirely in-process.

It deliberately depends on nothing beyond the Python standard library: the HTTP
client, JSON handling, retry logic, streaming, context management, tool
execution, subagents, skills, model output parsing and loop termination are all
implemented here, not delegated to an agent framework or a hosted
code-execution service.
"""

from .agent import Agent, AgentError, MaxIterationsExceeded
from .config import Config, load_config
from .llm import LLMClient, LLMError
from .skills import Skill, discover_skills, load_skills
from .store import SessionStore, StoreError
from .subagent import run_subagents
from .tools import TOOL_SCHEMAS, ToolRunner

__version__ = "0.6.0"

__all__ = [
    "Agent",
    "AgentError",
    "Config",
    "LLMClient",
    "LLMError",
    "MaxIterationsExceeded",
    "SessionStore",
    "Skill",
    "StoreError",
    "TOOL_SCHEMAS",
    "ToolRunner",
    "discover_skills",
    "load_config",
    "load_skills",
    "run_subagents",
    "__version__",
]
