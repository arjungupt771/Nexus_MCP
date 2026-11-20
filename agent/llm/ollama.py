"""Ollama adapter for local models (no API key required)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from agent.llm.errors import LLMResponseError
from agent.llm.http_base import HTTPProvider, parse_arguments, schema_instruction
from agent.llm.models import (
    FinishReason,
    LLMResponse,
    Message,
    Role,
    StreamChunk,
    TokenUsage,
    ToolCall,
    ToolSpec,
)


def _finish(reason: Any) -> FinishReason:
    return {"stop": FinishReason.STOP, "length": FinishReason.LENGTH}.get(
        str(reason), FinishReason.OTHER if reason else FinishReason.STOP
    )


def _calls(raw_calls: Any, offset: int = 0) -> list[ToolCall]:
    calls = []

    for index, raw in enumerate(raw_calls or [], start=offset):
        function = raw.get("function") or {}
        name = str(function.get("name", ""))
        calls.append(
            ToolCall(
                id=str(raw.get("id") or f"call_{index}_{name}"),
                name=name,
                arguments=parse_arguments(function.get("arguments")),
            )
        )

    return calls


class OllamaProvider(HTTPProvider):
    name = "ollama"
    default_base_url = "http://localhost:11434"
    requires_api_key = False

    def _endpoint(self, *, stream: bool) -> str:
        return f"{self._base_url}/api/chat"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _health_url(self) -> str:
        return f"{self._base_url}/api/tags"

    @staticmethod
    def _wire_message(message: Message) -> dict[str, Any]:
        if message.role is Role.TOOL:
            return {"role": "tool", "content": message.content, "tool_name": message.name or ""}

        wire: dict[str, Any] = {"role": message.role.value, "content": message.content}

        if message.role is Role.ASSISTANT and message.tool_calls:
            wire["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in message.tool_calls
            ]

        return wire

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        json_schema: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        wire = [self._wire_message(m) for m in messages]

        if json_schema is not None:
            wire.insert(0, {"role": "system", "content": schema_instruction(json_schema)})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": wire,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_output_tokens,
            },
        }

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        if json_schema is not None:
            payload["format"] = "json"

        return payload

    def _parse_response(self, body: dict[str, Any]) -> LLMResponse:
        message = body.get("message")

        if not isinstance(message, dict):
            raise LLMResponseError("ollama returned no message.", provider=self.name)

        calls = _calls(message.get("tool_calls"))
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=calls,
            usage=TokenUsage.of(body.get("prompt_eval_count"), body.get("eval_count")),
            finish_reason=FinishReason.TOOL_CALLS if calls else _finish(body.get("done_reason")),
            model=str(body.get("model") or ""),
        )

    async def _parse_stream(self, lines: AsyncIterator[str]) -> AsyncIterator[StreamChunk]:
        calls: list[ToolCall] = []
        usage = TokenUsage()
        reason: Any = None

        async for line in lines:
            if not line.strip():
                continue

            try:
                event = json.loads(line)
            except ValueError as exc:
                raise LLMResponseError(
                    "ollama sent a malformed stream line.", provider=self.name
                ) from exc

            if isinstance(event.get("error"), str):
                raise LLMResponseError(
                    f"ollama stream error: {event['error'][:200]}", provider=self.name
                )

            message = event.get("message") or {}
            calls.extend(_calls(message.get("tool_calls"), offset=len(calls)))

            if message.get("content"):
                yield StreamChunk(delta=message["content"])

            if event.get("done"):
                reason = event.get("done_reason")
                usage = TokenUsage.of(event.get("prompt_eval_count"), event.get("eval_count"))
                break

        yield StreamChunk(
            tool_calls=calls,
            usage=usage,
            finish_reason=FinishReason.TOOL_CALLS if calls else _finish(reason),
        )
