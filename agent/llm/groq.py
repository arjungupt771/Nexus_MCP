"""Groq adapter (OpenAI-compatible endpoint)."""

from __future__ import annotations

from agent.llm.openai_compat import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    default_base_url = "https://api.groq.com/openai/v1"
