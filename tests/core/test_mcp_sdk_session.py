
from __future__ import annotations

import pytest
from mcp import Client

from core.transports.sdk_session import MCPSDKSession
from mcp_servers.notes.server import create_notes_server


@pytest.fixture
def notes_server():
    return create_notes_server()


@pytest.mark.asyncio
async def test_connect_and_discover_tools(notes_server):
    session = MCPSDKSession(
        lambda: Client(
            notes_server,
            raise_exceptions=True,
            mode="auto",
        )
    )

    assert session.connected is False

    await session.connect()

    assert session.connected is True

    result = await session.list_tools()

    assert result.tools
    assert all(tool.name for tool in result.tools)

    await session.disconnect()

    assert session.connected is False


@pytest.mark.asyncio
async def test_disconnect_is_idempotent(notes_server):
    session = MCPSDKSession(
        lambda: Client(notes_server, mode="auto")
    )

    await session.connect()
    await session.disconnect()
    await session.disconnect()

    assert session.connected is False


@pytest.mark.asyncio
async def test_reconnect_creates_a_fresh_client(notes_server):
    created_clients: list[Client] = []

    def client_factory() -> Client:
        client = Client(notes_server, mode="auto")
        created_clients.append(client)
        return client

    session = MCPSDKSession(client_factory)

    await session.connect()
    first_client = session.client

    await session.disconnect()
    await session.connect()

    second_client = session.client

    assert first_client is not second_client
    assert len(created_clients) == 2

    result = await session.list_tools()
    assert result.tools

    await session.disconnect()


@pytest.mark.asyncio
async def test_client_access_before_connect_fails(notes_server):
    session = MCPSDKSession(
        lambda: Client(notes_server, mode="auto")
    )

    with pytest.raises(RuntimeError, match="not connected"):
        _ = session.client

    with pytest.raises(RuntimeError, match="not connected"):
        await session.list_tools()


@pytest.mark.asyncio
async def test_factory_failure_does_not_publish_connected_state():
    def failing_factory() -> Client:
        raise RuntimeError("Simulated client construction failure")

    session = MCPSDKSession(failing_factory)

    with pytest.raises(
        RuntimeError,
        match="Simulated client construction failure",
    ):
        await session.connect()

    assert session.connected is False
