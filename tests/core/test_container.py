
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.container import NexusContainer
from core.interfaces import ServerRegistry, ToolCatalog
from core.models import ServerDescriptor, ServerStatus, ToolDefinition


class FakeTransport:
    def __init__(self) -> None:
        self.connected = False
        self.disconnect_calls = 0

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False
        self.disconnect_calls += 1

    async def list_tools(self, *, cursor: str | None = None):
        return SimpleNamespace(
            tools=[
                ToolDefinition(
                    server_id="notes",
                    server_name="Notes MCP Server",
                    name="create_note",
                    description="Create a note",
                    input_schema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                )
            ],
            next_cursor=None,
        )

    async def call_tool(self, request):
        raise NotImplementedError


class FakeConnector:
    def __init__(self) -> None:
        self.transport = FakeTransport()

    async def connect(self, server: ServerDescriptor) -> FakeTransport:
        await self.transport.connect()
        return self.transport


def make_server() -> ServerDescriptor:
    return ServerDescriptor(
        server_id="notes",
        name="Notes MCP Server",
        transport="stdio",
    )


def test_container_creates_default_dependencies() -> None:
    container = NexusContainer(FakeConnector())

    assert isinstance(container.registry, ServerRegistry)
    assert isinstance(container.catalog, ToolCatalog)
    assert container.lifecycle is not None


def test_container_accepts_custom_dependencies() -> None:
    from core.registry import (
        InMemoryServerRegistry,
        InMemoryToolCatalog,
    )

    registry = InMemoryServerRegistry()
    catalog = InMemoryToolCatalog()

    container = NexusContainer(
        connector=FakeConnector(),
        registry=registry,
        catalog=catalog,
    )

    assert container.registry is registry
    assert container.catalog is catalog


@pytest.mark.asyncio
async def test_container_manages_complete_server_lifecycle() -> None:
    connector = FakeConnector()
    container = NexusContainer(connector)

    await container.lifecycle.register(make_server())
    await container.lifecycle.connect("notes")

    assert connector.transport.connected
    assert await container.catalog.count() == 1

    await container.close()

    assert not connector.transport.connected
    assert connector.transport.disconnect_calls == 1
    assert await container.catalog.count() == 0

    server = await container.registry.get("notes")

    assert server is not None
    assert server.status == ServerStatus.DISCONNECTED


@pytest.mark.asyncio
async def test_container_close_is_idempotent() -> None:
    connector = FakeConnector()
    container = NexusContainer(connector)

    await container.lifecycle.register(make_server())
    await container.lifecycle.connect("notes")

    await container.close()
    await container.close()

    assert connector.transport.disconnect_calls == 1


@pytest.mark.asyncio
async def test_container_rejects_reentry_after_close() -> None:
    container = NexusContainer(FakeConnector())

    await container.close()

    with pytest.raises(RuntimeError, match="already closed"):
        await container.__aenter__()


@pytest.mark.asyncio
async def test_container_supports_async_context_manager() -> None:
    connector = FakeConnector()

    async with NexusContainer(connector) as container:
        await container.lifecycle.register(make_server())
        await container.lifecycle.connect("notes")

        assert connector.transport.connected

    assert not connector.transport.connected
    assert connector.transport.disconnect_calls == 1
