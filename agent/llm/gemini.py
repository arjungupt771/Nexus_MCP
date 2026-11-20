"""Google Gemini adapter (Generative Language REST API)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from agent.llm.errors import LLMResponseError
from agent.llm.http_base import HTTPProvider, schema_instruction
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

# JSON Schema keywords Gemini's OpenAPI-subset parameter schema rejects.
_UNSUPPORTED = {
    "$schema",
    "additionalProperties",
    "$defs",
    "definitions",
    "title",
    "default",
    "examples",
}
_FINISH = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
    "BLOCKLIST": FinishReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
}


def sanitize_schema(schema: Any) -> Any:
    """Convert a JSON Schema into the subset Gemini accepts."""
    if isinstance(schema, list):
        return [sanitize_schema(item) for item in schema]

    if not isinstance(schema, dict):
        return schema

    cleaned: dict[str, Any] = {}

    for key, value in schema.items():
        if key in _UNSUPPORTED:
            continue

        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {name: sanitize_schema(sub) for name, sub in value.items()}
        elif key == "type" and isinstance(value, list):
            non_null = [item for item in value if item != "null"]
            cleaned["type"] = non_null[0] if non_null else "string"

            if "null" in value:
                cleaned["nullable"] = True
        else:
            cleaned[key] = sanitize_schema(value)

    return cleaned


def _usage(raw: Any) -> TokenUsage | None:
    if not isinstance(raw, dict):
        return None

    return TokenUsage.of(
        raw.get("promptTokenCount"), raw.get("candidatesTokenCount"), raw.get("totalTokenCount")
    )


def _tool_result(content: str) -> Any:
    try:
        parsed = json.loads(content)
    except ValueError:
        return {"result": content}

    return parsed if isinstance(parsed, dict) else {"result": parsed}


class GeminiProvider(HTTPProvider):
    name = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _endpoint(self, *, stream: bool) -> str:
        model = f"{self._base_url}/models/{self.config.model}"
        return f"{model}:streamGenerateContent?alt=sse" if stream else f"{model}:generateContent"

    def _headers(self) -> dict[str, str]:
        # Header (not query string) so keys never end up in URLs or logs.
        return {"x-goog-api-key": self._api_key or "", "Content-Type": "application/json"}

    def _health_url(self) -> str:
        return f"{self._base_url}/models/{self.config.model}"

    def _contents(self, messages: list[Message]) -> tuple[list[str], list[dict[str, Any]]]:
        system: list[str] = []
        contents: list[dict[str, Any]] = []

        for message in messages:
            if message.role is Role.SYSTEM:
                system.append(message.content)
            elif message.role is Role.TOOL:
                part = {
                    "functionResponse": {
                        "name": message.name or "",
                        "response": _tool_result(message.content),
                    }
                }

                # All responses to one model turn travel in a single content.
                if contents and contents[-1].get("_tool"):
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part], "_tool": True})
            elif message.role is Role.ASSISTANT:
                parts: list[dict[str, Any]] = []

                if message.content:
                    parts.append({"text": message.content})

                parts.extend(
                    {"functionCall": {"name": c.name, "args": c.arguments}}
                    for c in message.tool_calls
                )
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            else:
                contents.append({"role": "user", "parts": [{"text": message.content}]})

        for content in contents:
            content.pop("_tool", None)

        return system, contents

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        json_schema: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        system, contents = self._contents(messages)

        if json_schema is not None:
            system.append(schema_instruction(json_schema))

        generation: dict[str, Any] = {
            "temperature": self.config.temperature,
            "maxOutputTokens": self.config.max_output_tokens,
        }

        if json_schema is not None:
            generation["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation}

        if system:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}

        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": sanitize_schema(t.parameters),
                        }
                        for t in tools
                    ]
                }
            ]

        return payload

    @staticmethod
    def _extract(candidate: dict[str, Any], offset: int = 0) -> tuple[str, list[ToolCall]]:
        text: list[str] = []
        calls: list[ToolCall] = []

        for part in (candidate.get("content") or {}).get("parts") or []:
            if "text" in part:
                text.append(part["text"])
            elif "functionCall" in part:
                function = part["functionCall"]
                name = str(function.get("name", ""))
                calls.append(
                    ToolCall(
                        id=f"call_{offset + len(calls)}_{name}",
                        name=name,
                        arguments=function.get("args") or {},
                    )
                )

        return "".join(text), calls

    def _parse_response(self, body: dict[str, Any]) -> LLMResponse:
        candidates = body.get("candidates")

        if not candidates:
            blocked = (body.get("promptFeedback") or {}).get("blockReason")

            if blocked:
                return LLMResponse(finish_reason=FinishReason.CONTENT_FILTER)

            raise LLMResponseError("gemini returned no candidates.", provider=self.name)

        text, calls = self._extract(candidates[0])
        return LLMResponse(
            content=text,
            tool_calls=calls,
            usage=_usage(body.get("usageMetadata")) or TokenUsage(),
            finish_reason=FinishReason.TOOL_CALLS
            if calls
            else _FINISH.get(str(candidates[0].get("finishReason")), FinishReason.OTHER),
            model=str(body.get("modelVersion") or ""),
        )

    async def _parse_stream(self, lines: AsyncIterator[str]) -> AsyncIterator[StreamChunk]:
        calls: list[ToolCall] = []
        usage: TokenUsage | None = None
        finish: FinishReason | None = None

        async for line in lines:
            line = line.strip()

            if not line.startswith("data:"):
                continue

            try:
                event = json.loads(line[5:].strip())
            except ValueError as exc:
                raise LLMResponseError(
                    "gemini sent a malformed stream event.", provider=self.name
                ) from exc

            usage = _usage(event.get("usageMetadata")) or usage

            for candidate in event.get("candidates") or []:
                text, new_calls = self._extract(candidate, offset=len(calls))
                calls.extend(new_calls)

                if text:
                    yield StreamChunk(delta=text)

                if candidate.get("finishReason"):
                    finish = _FINISH.get(candidate["finishReason"], FinishReason.OTHER)

        yield StreamChunk(
            tool_calls=calls,
            usage=usage,
            finish_reason=FinishReason.TOOL_CALLS if calls else (finish or FinishReason.STOP),
        )
