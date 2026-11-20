"""Agent abstraction."""

from agent.core.agent import Agent, AgentResult
from agent.core.context import prepare_context, trim_messages
from agent.core.definition import AgentDefinition, ExecutionSettings

__all__ = [
    "Agent",
    "AgentDefinition",
    "AgentResult",
    "ExecutionSettings",
    "prepare_context",
    "trim_messages",
]
