"""Regression: SDK sessions must survive being driven from different tasks."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from mcp import Client, types

from core.transports.sdk_session import MCPSDKSession, build_tools_changed_handler
from mcp_servers.notes.server import create_notes_server


def make_session() -> MCPSDKSession:
    server = create_notes_server()
    return MCPSDKSession(lambda: Client(server, mode="auto"))


async def test_connect_and_disconnect_may_happen_in_different_tasks():
    session = make_session()

    await asyncio.create_task(session.connect())
    assert session.connected

    assert (await asyncio.create_task(session.list_tools())).tools

    await asyncio.create_task(session.disconnect())
    assert not session.connected


async def test_reconnect_across_tasks_creates_a_working_session():
    session = make_session()

    for _ in range(2):
        await asyncio.create_task(session.connect())
        assert (await session.list_tools()).tools
        await asyncio.create_task(session.disconnect())

    assert not session.connected


async def test_cancelled_connect_leaves_no_session_behind():
    started = asyncio.Event()

    class SlowClient:
        async def __aenter__(self):
            started.set()
            await asyncio.sleep(30)

        async def __aexit__(self, *exc):
            return None

    session = MCPSDKSession(lambda: SlowClient())
    task = asyncio.create_task(session.connect())
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert not session.connected
    await session.disconnect()  # still safe


async def test_server_capabilities_are_exposed_when_connected():
    session = make_session()
    assert session.server_capabilities is None

    await session.connect()
    assert session.server_capabilities.tools is not None

    await session.disconnect()
    assert session.server_capabilities is None


async def test_tools_changed_handler_only_fires_for_tool_list_notifications():
    calls: list[int] = []
    handler = build_tools_changed_handler(lambda: calls.append(1))

    await handler(types.ToolListChangedNotification())
    await handler(SimpleNamespace(root=types.ToolListChangedNotification()))
    await handler(types.PromptListChangedNotification())
    await handler(RuntimeError("transport error"))

    assert calls == [1, 1]


async def test_tools_changed_handler_swallows_callback_errors():
    def broken() -> None:
        raise RuntimeError("boom")

    await build_tools_changed_handler(broken)(types.ToolListChangedNotification())


async def test_cancelling_the_owner_task_closes_the_session_cleanly():
    """Regression: cancellation (e.g. loop shutdown) must not hang SDK cleanup."""
    session = make_session()
    await session.connect()
    owner = session._owner
    assert owner is not None

    owner.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(owner, timeout=5)

    assert not session.connected
