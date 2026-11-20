"""Structured LangGraph state for the agent workflow."""

from __future__ import annotations

from typing import Any, TypedDict

from agent.llm.models import LLMResponse, Message, TokenUsage, ToolCall, ToolSpec
from agent.tools.bridge import ToolOutcome


class AgentState(TypedDict, total=False):
    request_id: str
    user_input: str
    # Working conversation (without the system prompt).
    messages: list[Message]
    # Messages created during this run (user turn, assistant turns, tool results).
    new_messages: list[Message]
    prepared: list[Message]
    tool_specs: list[ToolSpec]
    response: LLMResponse | None
    pending_calls: list[ToolCall]
    outcomes: list[ToolOutcome]
    turn_outcomes: list[ToolOutcome]
    iterations: int
    usage: TokenUsage
    final_response: str
    stop_reason: str  # completed | max_iterations | error
    error: dict[str, Any] | None
