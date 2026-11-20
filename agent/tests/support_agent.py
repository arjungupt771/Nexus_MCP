"""Fakes for agent / bridge tests: scripted LLM + scriptable MCP transport."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from agent.llm import (
    FinishReason,
    LLMError,
    LLMResponse,
    ModelConfig,
    ModelInfo,
    TokenUsage,
    ToolCall,
)
from core.container import NexusContainer


def sdk_tool(name: str, properties: dict | None = None, required: list[str] | None = None):
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}}

    if required:
        schema["required"] = required

    return SimpleNamespace(name=name, description=f"{name} description", input_schema=schema)


class ToolTransport:
    """Fake MCP transport that records calls and returns SDK-like results."""

    def __init__(self, tools: list[Any], handlers: dict[str, Any] | None = None) -> None:
        self.tools = tools
        self.handlers = handlers or {}
        self.calls: list[tuple[str, dict]] = []
        self.delay = 0.0
        self.connected = False
        self.server_capabilities = {"tools": {}}

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def list_tools(self, *, cursor: str | None = None):
        return SimpleNamespace(tools=list(self.tools), next_cursor=None)

    async def call_tool(self, name, arguments=None, *, read_timeout_seconds=None):
        self.calls.append((name, dict(arguments or {})))

        if self.delay:
            await asyncio.sleep(self.delay)

        handler = self.handlers.get(name)

        if handler is None:
            return SimpleNamespace(
                is_error=False, structured_content={"ok": True, "tool": name}, content=[]
            )

        return handler(arguments or {})


def tool_error(message: str):
    return SimpleNamespace(
        is_error=True, structured_content=None, content=[SimpleNamespace(text=message)]
    )


class Connector:
    def __init__(self, transports: dict[str, ToolTransport]) -> None:
        self.transports = transports

    async def connect(self, server):
        transport = self.transports[server.server_id]
        await transport.connect()
        return transport


async def make_world(transports: dict[str, ToolTransport]) -> NexusContainer:
    container = NexusContainer(Connector(transports))

    for server_id in transports:
        await container.servers.register(
            {
                "server_id": server_id,
                "name": f"{server_id.title()} Server",
                "config": {"transport": "stdio", "command": "python"},
            }
        )
        await container.servers.connect(server_id)

    return container


class ScriptedProvider:
    """LLMProvider stand-in replaying scripted responses (or raising errors)."""

    name = "scripted"

    def __init__(self, script: list[LLMResponse | LLMError]) -> None:
        self.config = ModelConfig(provider="scripted", model="scripted-1")
        self.script = list(script)
        self.requests: list[tuple[list, list | None]] = []
        self.closed = False

    async def generate(self, messages, *, tools=None):
        self.requests.append((list(messages), tools))
        step = self.script.pop(0)

        if isinstance(step, Exception):
            raise step

        return step

    async def stream(self, messages, *, tools=None):
        yield  # pragma: no cover

    async def generate_structured(self, messages, schema):
        raise NotImplementedError

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider="scripted", model="scripted-1")

    async def health_check(self) -> bool:
        return True

    async def aclose(self) -> None:
        self.closed = True


def say(text: str, tokens: int = 10) -> LLMResponse:
    return LLMResponse(content=text, usage=TokenUsage.of(tokens, 1))


def call(*calls: tuple[str, dict], tokens: int = 10) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)],
        finish_reason=FinishReason.TOOL_CALLS,
        usage=TokenUsage.of(tokens, 1),
    )
