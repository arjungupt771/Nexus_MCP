"""Shared fakes and builders for Phase 5 tests (no real MCP, no network)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from core.container import NexusContainer
from core.models import ServerDescriptor
from core.models.server_config import ServerRegistration


def sdk_tool(name: str, description: str = "", properties: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=description or f"{name} tool",
        input_schema={"type": "object", "properties": properties or {}},
    )


def stdio_payload(server_id: str = "notes", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "server_id": server_id,
        "name": f"{server_id.title()} Server",
        "description": "Test server",
        "tags": ["test"],
        "capabilities": ["notes"],
        "config": {"transport": "stdio", "command": "python", "args": ["-m", "x"]},
    }
    payload.update(overrides)
    return payload


def http_payload(server_id: str = "remote", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "server_id": server_id,
        "name": "Remote",
        "config": {
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer ${REMOTE_TOKEN}"},
        },
    }
    payload.update(overrides)
    return payload


def registration(server_id: str = "notes", **overrides: Any) -> ServerRegistration:
    return ServerRegistration.from_payload(stdio_payload(server_id, **overrides))


class FakeTransport:
    """Scriptable MCP transport."""

    def __init__(self, tools: list[SimpleNamespace] | None = None) -> None:
        self.tools = tools if tools is not None else [sdk_tool("create_note")]
        self.connected = False
        self.disconnect_calls = 0
        self.list_calls = 0
        self.fail_list = False
        self.list_delay = 0.0
        self.server_capabilities: Any = {"tools": {"listChanged": True}}

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    async def list_tools(self, *, cursor: str | None = None) -> Any:
        self.list_calls += 1

        if self.list_delay:
            await asyncio.sleep(self.list_delay)

        if self.fail_list:
            raise RuntimeError("discovery boom")

        return SimpleNamespace(tools=list(self.tools), next_cursor=None)

    async def call_tool(self, name: str, arguments: Any = None, **_: Any) -> Any:
        raise NotImplementedError


class FakeConnector:
    """Connector producing one FakeTransport per server id."""

    def __init__(self) -> None:
        self.transports: dict[str, FakeTransport] = {}
        self.connects: list[str] = []
        self.fail_for: set[str] = set()
        self.connect_delay = 0.0

    def transport_for(self, server_id: str) -> FakeTransport:
        return self.transports.setdefault(server_id, FakeTransport())

    async def connect(self, server: ServerDescriptor) -> FakeTransport:
        self.connects.append(server.server_id)

        if self.connect_delay:
            await asyncio.sleep(self.connect_delay)

        if server.server_id in self.fail_for:
            raise RuntimeError("connect boom")

        transport = FakeTransport(list(self.transport_for(server.server_id).tools))
        previous = self.transports.get(server.server_id)

        if previous is not None:
            transport.fail_list = previous.fail_list
            transport.server_capabilities = previous.server_capabilities

        self.transports[server.server_id] = transport
        await transport.connect()
        return transport


def make_container(connector: FakeConnector | None = None, **kwargs: Any) -> NexusContainer:
    return NexusContainer(connector or FakeConnector(), **kwargs)


def event_types(container: NexusContainer) -> list[str]:
    return [event.type.value for event in container.events.recent()]
