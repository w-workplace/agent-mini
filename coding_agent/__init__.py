"""A minimal, dependency-free coding agent.

``coding_agent`` is an autonomous software-engineering assistant that talks to
any OpenAI-compatible chat-completions API using the model's native tool
calling, and executes a small set of local tools (read / write / edit files,
grep, list files, run shell commands) entirely in-process.

It deliberately depends on nothing beyond the Python standard library: the HTTP
client, JSON handling, retry logic, context management, tool execution, model
output parsing and loop termination are all implemented here, not delegated to
an agent framework or a hosted code-execution service.
"""

from .agent import Agent, AgentError, MaxIterationsExceeded
from .config import Config, load_config
from .llm import LLMClient, LLMError
from .tools import TOOL_SCHEMAS, ToolRunner

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentError",
    "Config",
    "LLMClient",
    "LLMError",
    "MaxIterationsExceeded",
    "TOOL_SCHEMAS",
    "ToolRunner",
    "load_config",
    "__version__",
]
