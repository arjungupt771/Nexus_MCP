"""Base adapter for OpenAI-compatible chat-completions APIs (used by Groq)."""

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

_FINISH = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}


def _usage(raw: Any) -> TokenUsage | None:
    if not isinstance(raw, dict):
        return None

    return TokenUsage.of(
        raw.get("prompt_tokens"), raw.get("completion_tokens"), raw.get("total_tokens")
    )


class OpenAICompatibleProvider(HTTPProvider):
    def _endpoint(self, *, stream: bool) -> str:
        return f"{self._base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _health_url(self) -> str:
        return f"{self._base_url}/models"

    @staticmethod
    def _wire_message(message: Message) -> dict[str, Any]:
        if message.role is Role.TOOL:
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id or "",
                "content": message.content,
            }

        wire: dict[str, Any] = {"role": message.role.value, "content": message.content}

        if message.role is Role.ASSISTANT and message.tool_calls:
            wire["content"] = message.content or None
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in message.tool_calls
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
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "stream": stream,
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
            payload["response_format"] = {"type": "json_object"}

        if stream:
            payload["stream_options"] = {"include_usage": True}

        return payload

    def _parse_response(self, body: dict[str, Any]) -> LLMResponse:
        choices = body.get("choices")

        if not isinstance(choices, list) or not choices:
            raise LLMResponseError(f"{self.name} returned no choices.", provider=self.name)

        choice = choices[0]
        message = choice.get("message") or {}
        calls = [
            ToolCall(
                id=str(raw.get("id") or f"call_{index}"),
                name=str(raw["function"]["name"]),
                arguments=parse_arguments(raw["function"].get("arguments")),
            )
            for index, raw in enumerate(message.get("tool_calls") or [])
        ]
        finish = _FINISH.get(str(choice.get("finish_reason")), FinishReason.OTHER)

        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=calls,
            usage=_usage(body.get("usage")) or TokenUsage(),
            finish_reason=FinishReason.TOOL_CALLS if calls else finish,
            model=str(body.get("model") or ""),
        )

    async def _parse_stream(self, lines: AsyncIterator[str]) -> AsyncIterator[StreamChunk]:
        partial: dict[int, dict[str, str]] = {}
        finish: FinishReason | None = None
        usage: TokenUsage | None = None

        async for line in lines:
            line = line.strip()

            if not line.startswith("data:"):
                continue

            data = line[5:].strip()

            if data == "[DONE]":
                break

            try:
                event = json.loads(data)
            except ValueError as exc:
                raise LLMResponseError(
                    f"{self.name} sent a malformed stream event.", provider=self.name
                ) from exc

            usage = (
                _usage(event.get("usage"))
                or _usage((event.get("x_groq") or {}).get("usage"))
                or usage
            )

            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}

                for raw in delta.get("tool_calls") or []:
                    slot = partial.setdefault(
                        int(raw.get("index", 0)), {"id": "", "name": "", "args": ""}
                    )
                    slot["id"] = raw.get("id") or slot["id"]
                    function = raw.get("function") or {}
                    slot["name"] += function.get("name") or ""
                    slot["args"] += function.get("arguments") or ""

                if delta.get("content"):
                    yield StreamChunk(delta=delta["content"])

                if choice.get("finish_reason"):
                    finish = _FINISH.get(choice["finish_reason"], FinishReason.OTHER)

        calls = [
            ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments=parse_arguments(slot["args"]),
            )
            for index, slot in sorted(partial.items())
        ]
        yield StreamChunk(
            tool_calls=calls,
            usage=usage,
            finish_reason=FinishReason.TOOL_CALLS if calls else (finish or FinishReason.STOP),
        )
