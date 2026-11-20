
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.exceptions import (
    DiscoveryError,
    ServerConnectionError,
    ServerNotFoundError,
)
from core.models import (
    ServerDescriptor,
    ServerStatus,
    ToolDefinition,
)
from core.registry import (
    InMemoryServerRegistry,
    InMemoryToolCatalog,
)
from core.services import ServerLifecycleManager


def make_server() -> ServerDescriptor:
    return ServerDescriptor(
        server_id="github",
        name="GitHub MCP Server",
        transport="stdio",
    )


def make_tool(name: str = "search_repositories") -> ToolDefinition:
    return ToolDefinition(
        server_id="github",
        server_name="GitHub MCP Server",
        name=name,
        description="Test tool",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )


class FakeTransport:
    def __init__(self) -> None:
        self.connected = False
        self.disconnect_calls = 0
        self.tools = [make_tool()]
        self.fail_discovery = False
        self.fail_disconnect = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnect_calls += 1

        if self.fail_disconnect:
            raise RuntimeError("Disconnect failed")

        self.connected = False

    async def list_tools(self, *, cursor: str | None = None):
        if self.fail_discovery:
            raise RuntimeError("Discovery failed")

        return SimpleNamespace(tools=self.tools, next_cursor=None)

    async def call_tool(self, request):
        raise NotImplementedError


class FakeConnector:
    def __init__(self, transport: FakeTransport | None = None) -> None:
        self.transport = transport or FakeTransport()
        self.calls = 0
        self.fail_connect = False

    async def connect(self, server: ServerDescriptor) -> FakeTransport:
        self.calls += 1

        if self.fail_connect:
            raise RuntimeError("Connection failed")

        await self.transport.connect()
        return self.transport


@pytest.fixture
def components():
    registry = InMemoryServerRegistry()
    catalog = InMemoryToolCatalog()
    transport = FakeTransport()
    connector = FakeConnector(transport)

    manager = ServerLifecycleManager(
        registry=registry,
        connector=connector,
        catalog=catalog,
    )

    return manager, registry, catalog, connector, transport


@pytest.mark.asyncio
async def test_register_does_not_connect(components) -> None:
    manager, _, _, connector, _ = components

    server = await manager.register(make_server())

    assert server.status == ServerStatus.DISCONNECTED
    assert connector.calls == 0


@pytest.mark.asyncio
async def test_connect_discovers_tools_and_updates_status(components) -> None:
    manager, _, catalog, _, _ = components

    await manager.register(make_server())
    connected = await manager.connect("github")

    assert connected.status == ServerStatus.CONNECTED
    assert await manager.is_connected("github")
    assert await catalog.count() == 1


@pytest.mark.asyncio
async def test_connect_is_idempotent(components) -> None:
    manager, _, _, connector, _ = components

    await manager.register(make_server())

    await manager.connect("github")
    await manager.connect("github")

    assert connector.calls == 1


@pytest.mark.asyncio
async def test_connect_unknown_server_raises(components) -> None:
    manager, _, _, _, _ = components

    with pytest.raises(ServerNotFoundError):
        await manager.connect("unknown")


@pytest.mark.asyncio
async def test_connect_failure_sets_error_status(components) -> None:
    manager, registry, catalog, connector, _ = components

    connector.fail_connect = True
    await manager.register(make_server())

    with pytest.raises(ServerConnectionError):
        await manager.connect("github")

    server = await registry.get("github")

    assert server is not None
    assert server.status == ServerStatus.ERROR
    assert await catalog.count() == 0
    assert not await manager.is_connected("github")


@pytest.mark.asyncio
async def test_connected_status_failure_does_not_publish_transport() -> None:
    class FailingConnectedRegistry(InMemoryServerRegistry):
        async def update_status(
            self,
            server_id: str,
            status: ServerStatus,
        ) -> ServerDescriptor:
            if status == ServerStatus.CONNECTED:
                raise RuntimeError("Status update failed")

            return await super().update_status(server_id, status)

    registry = FailingConnectedRegistry()
    catalog = InMemoryToolCatalog()
    transport = FakeTransport()
    manager = ServerLifecycleManager(
        registry=registry,
        connector=FakeConnector(transport),
        catalog=catalog,
    )
    await manager.register(make_server())

    with pytest.raises(ServerConnectionError):
        await manager.connect("github")

    stored = await registry.get("github")
    assert stored is not None
    assert stored.status == ServerStatus.ERROR
    assert transport.disconnect_calls == 1
    assert not await manager.is_connected("github")


@pytest.mark.asyncio
async def test_discovery_failure_cleans_up_transport(components) -> None:
    manager, registry, catalog, _, transport = components

    transport.fail_discovery = True
    await manager.register(make_server())

    with pytest.raises(DiscoveryError):
        await manager.connect("github")

    server = await registry.get("github")

    assert server is not None
    assert server.status == ServerStatus.ERROR
    assert transport.disconnect_calls == 1
    assert not await manager.is_connected("github")
    assert await catalog.count() == 0


@pytest.mark.asyncio
async def test_connect_discovery_failure_preserves_existing_snapshot(
    components,
) -> None:
    manager, _, catalog, _, transport = components

    await catalog.upsert_tools("github", [make_tool("existing_tool")])
    transport.fail_discovery = True
    await manager.register(make_server())

    with pytest.raises(DiscoveryError):
        await manager.connect("github")

    assert await catalog.get_tool("github.existing_tool") is not None


@pytest.mark.asyncio
async def test_refresh_tools_replaces_snapshot(components) -> None:
    manager, _, catalog, _, transport = components

    await manager.register(make_server())
    await manager.connect("github")

    transport.tools = [make_tool("list_issues")]

    refreshed = await manager.refresh_tools("github")

    assert len(refreshed) == 1
    assert refreshed[0].name == "list_issues"
    assert await catalog.get_tool(
        "github.search_repositories"
    ) is None


@pytest.mark.asyncio
async def test_refresh_requires_connection(components) -> None:
    manager, _, _, _, _ = components

    await manager.register(make_server())

    with pytest.raises(ServerConnectionError):
        await manager.refresh_tools("github")


@pytest.mark.asyncio
async def test_get_transport_returns_active_transport(components) -> None:
    manager, _, _, _, transport = components

    await manager.register(make_server())
    await manager.connect("github")

    assert await manager.get_transport("github") is transport


@pytest.mark.asyncio
async def test_transport_leases_allow_concurrent_calls(components) -> None:
    manager, _, _, _, transport = components

    await manager.register(make_server())
    await manager.connect("github")

    entered_calls = 0
    both_entered = asyncio.Event()
    release_calls = asyncio.Event()

    async def use_transport() -> None:
        nonlocal entered_calls

        async with manager.use_transport("github") as active_transport:
            assert active_transport is transport
            entered_calls += 1
            if entered_calls == 2:
                both_entered.set()
            await release_calls.wait()

    calls = [
        asyncio.create_task(use_transport()),
        asyncio.create_task(use_transport()),
    ]
    await asyncio.wait_for(both_entered.wait(), timeout=1)
    release_calls.set()
    await asyncio.gather(*calls)


@pytest.mark.asyncio
async def test_disconnect_waits_for_leases_and_blocks_new_ones(
    components,
) -> None:
    manager, _, _, _, _ = components

    await manager.register(make_server())
    await manager.connect("github")

    lease_entered = asyncio.Event()
    release_lease = asyncio.Event()

    async def hold_lease() -> None:
        async with manager.use_transport("github"):
            lease_entered.set()
            await release_lease.wait()

    active_call = asyncio.create_task(hold_lease())
    await lease_entered.wait()
    disconnect_task = asyncio.create_task(manager.disconnect("github"))
    await asyncio.sleep(0)

    new_lease_entered = asyncio.Event()

    async def acquire_new_lease() -> None:
        async with manager.use_transport("github"):
            new_lease_entered.set()

    new_call = asyncio.create_task(acquire_new_lease())
    await asyncio.sleep(0.01)

    assert not disconnect_task.done()
    assert not new_lease_entered.is_set()

    release_lease.set()
    await active_call
    await disconnect_task

    with pytest.raises(ServerConnectionError):
        await new_call


@pytest.mark.asyncio
async def test_cancelled_transport_lease_is_released(components) -> None:
    manager, _, _, _, _ = components

    await manager.register(make_server())
    await manager.connect("github")

    lease_entered = asyncio.Event()

    async def hold_lease() -> None:
        async with manager.use_transport("github"):
            lease_entered.set()
            await asyncio.Event().wait()

    active_call = asyncio.create_task(hold_lease())
    await lease_entered.wait()
    active_call.cancel()

    with pytest.raises(asyncio.CancelledError):
        await active_call

    await manager.disconnect("github")
    assert not await manager.is_connected("github")


@pytest.mark.asyncio
async def test_refresh_failure_preserves_connected_transport(components) -> None:
    manager, _, _, _, transport = components

    await manager.register(make_server())
    await manager.connect("github")

    transport.fail_discovery = True

    with pytest.raises(DiscoveryError):
        await manager.refresh_tools("github")

    assert await manager.is_connected("github")


@pytest.mark.asyncio
async def test_disconnect_removes_tools_and_updates_status(components) -> None:
    manager, registry, catalog, _, _ = components

    await manager.register(make_server())
    await manager.connect("github")

    disconnected = await manager.disconnect("github")

    assert disconnected.status == ServerStatus.DISCONNECTED
    assert not await manager.is_connected("github")
    assert await catalog.count() == 0

    stored = await registry.get("github")
    assert stored is not None
    assert stored.status == ServerStatus.DISCONNECTED


@pytest.mark.asyncio
async def test_disconnect_is_idempotent(components) -> None:
    manager, _, catalog, _, transport = components

    await manager.register(make_server())

    await manager.disconnect("github")
    await manager.disconnect("github")

    assert transport.disconnect_calls == 0
    assert await catalog.count() == 0


@pytest.mark.asyncio
async def test_disconnect_failure_sets_error_status(components) -> None:
    manager, registry, catalog, _, transport = components

    await manager.register(make_server())
    await manager.connect("github")

    transport.fail_disconnect = True

    with pytest.raises(ServerConnectionError):
        await manager.disconnect("github")

    stored = await registry.get("github")

    assert stored is not None
    assert stored.status == ServerStatus.ERROR
    assert await catalog.count() == 0


@pytest.mark.asyncio
async def test_unregister_disconnects_and_removes_server(components) -> None:
    manager, registry, catalog, _, _ = components

    await manager.register(make_server())
    await manager.connect("github")

    removed = await manager.unregister("github")

    assert removed.server_id == "github"
    assert await registry.get("github") is None
    assert await catalog.count() == 0
    assert not await manager.is_connected("github")


@pytest.mark.asyncio
async def test_unregister_unknown_server_raises(components) -> None:
    manager, _, _, _, _ = components

    with pytest.raises(ServerNotFoundError):
        await manager.unregister("unknown")



@pytest.mark.asyncio
async def test_close_all_disconnects_every_server() -> None:
    registry = InMemoryServerRegistry()
    catalog = InMemoryToolCatalog()

    first_transport = FakeTransport()
    first_transport.tools = [make_tool("search_repositories")]

    second_transport = FakeTransport()
    second_transport.tools = [
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
    ]

    class MultiConnector:
        async def connect(self, server: ServerDescriptor):
            transport = (
                first_transport
                if server.server_id == "github"
                else second_transport
            )
            await transport.connect()
            return transport

    manager = ServerLifecycleManager(
        registry,
        MultiConnector(),
        catalog,
    )

    await manager.register(make_server())

    await manager.register(
        ServerDescriptor(
            server_id="notes",
            name="Notes MCP Server",
            transport="stdio",
        )
    )

    await manager.connect("github")
    await manager.connect("notes")

    assert await catalog.count() == 2

    await manager.close_all()

    assert not await manager.is_connected("github")
    assert not await manager.is_connected("notes")
    assert await catalog.count() == 0


@pytest.mark.asyncio
async def test_concurrent_connect_only_opens_one_transport(components) -> None:
    manager, _, _, connector, _ = components

    await manager.register(make_server())

    results = await asyncio.gather(
        manager.connect("github"),
        manager.connect("github"),
        manager.connect("github"),
    )

    assert len(results) == 3
    assert connector.calls == 1


@pytest.mark.asyncio
async def test_list_servers_returns_registered_servers(components) -> None:
    manager, _, _, _, _ = components

    await manager.register(make_server())

    servers = await manager.list_servers()

    assert len(servers) == 1
    assert servers[0].server_id == "github"
