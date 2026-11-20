"""Helpers for LLM provider tests (mocked HTTP only)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from agent.llm import ModelConfig, RetryConfig, default_registry

SECRET = "sk-test-SECRET-key-123456"


def build(provider: str, handler: Callable[[httpx.Request], httpx.Response], **overrides: Any):
    """Create a provider wired to a mock transport; returns (provider, sleeps)."""
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    retry = overrides.pop("retry", RetryConfig(jitter=False, initial_delay_seconds=0.5))
    config = ModelConfig(provider=provider, model="test-model", retry=retry, **overrides)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    instance = default_registry().create(
        config,
        env={"GEMINI_API_KEY": SECRET, "GROQ_API_KEY": SECRET},
        client=client,
        sleep=fake_sleep,
    )
    return instance, sleeps


def text_body(provider: str, text: str = "hello") -> dict[str, Any]:
    if provider == "gemini":
        return {
            "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 7,
                "candidatesTokenCount": 3,
                "totalTokenCount": 10,
            },
        }

    if provider == "groq":
        return {
            "model": "test-model",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }

    return {
        "message": {"content": text},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 7,
        "eval_count": 3,
    }


def tool_body(provider: str, name: str = "notes__create_note", args: dict | None = None):
    args = args or {"title": "T"}

    if provider == "gemini":
        return {
            "candidates": [
                {
                    "content": {"parts": [{"functionCall": {"name": name, "args": args}}]},
                    "finishReason": "STOP",
                }
            ]
        }

    if provider == "groq":
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "function": {"name": name, "arguments": json.dumps(args)},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    return {
        "message": {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]},
        "done": True,
    }


def ok(body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=body)


class BrokenStream(httpx.AsyncByteStream):
    """Yields some bytes then fails, simulating a dropped connection."""

    def __init__(self, first: bytes) -> None:
        self._first = first

    async def __aiter__(self):
        yield self._first
        raise httpx.ReadError("connection dropped")
