"""Conversation context preparation."""

from __future__ import annotations

from agent.llm.models import Message, Role


def trim_messages(history: list[Message], max_messages: int) -> list[Message]:
    """Bound the conversation without breaking tool-call structure.

    Keeps the most recent ``max_messages`` but always starts on a USER
    message, so no TOOL result is ever orphaned from its assistant call and
    providers that require a leading user turn (Gemini) stay valid. The
    latest user request is always retained.
    """
    if len(history) <= max_messages:
        return list(history)

    window = history[-max_messages:]
    first_user = next((i for i, m in enumerate(window) if m.role is Role.USER), None)

    if first_user is not None:
        return window[first_user:]

    last_user = max((i for i, m in enumerate(history) if m.role is Role.USER), default=0)
    return history[last_user:]


def prepare_context(
    system_instructions: str,
    messages: list[Message],
    max_messages: int,
) -> list[Message]:
    system = Message(role=Role.SYSTEM, content=system_instructions)
    return [system, *trim_messages(messages, max_messages)]
