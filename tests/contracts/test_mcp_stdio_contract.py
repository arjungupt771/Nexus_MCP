from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import pytest

from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SERVER_CASES = [
    pytest.param(
        "mcp_servers.notes.main",
        10,
        id="notes",
    ),
    pytest.param(
        "mcp_servers.github.main",
        10,
        id="github",
    ),
    pytest.param(
        "mcp_servers.calendar.main",
        8,
        id="calendar",
    ),
]


def build_server_parameters(module: str) -> StdioServerParameters:
    """Build an isolated stdio subprocess configuration."""

    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")

    env["PYTHONPATH"] = os.pathsep.join(
        part for part in [str(PROJECT_ROOT), existing_pythonpath] if part
    )

    # Do not pass real service credentials into contract subprocesses.
    env.pop("GITHUB_TOKEN", None)
    env.pop("GOOGLE_ACCESS_TOKEN", None)

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", module],
        env=env,
    )


def extract_payload(result) -> dict:
    payload = result.structured_content

    assert isinstance(payload, dict)
    return payload


# =========================================================
# 1. Independent stdio server discovery
# =========================================================


@pytest.mark.parametrize("module, expected_tool_count", SERVER_CASES)
@pytest.mark.asyncio
async def test_server_discovers_tools_over_stdio(
    module: str,
    expected_tool_count: int,
):
    params = build_server_parameters(module)

    async with Client(params, raise_exceptions=True) as client:
        result = await client.list_tools()

    assert len(result.tools) == expected_tool_count

    names = [tool.name for tool in result.tools]

    assert len(names) == len(set(names))
    assert all(name.strip() for name in names)


# =========================================================
# 2. Simultaneous connections to all three servers
# =========================================================


@pytest.mark.asyncio
async def test_all_servers_can_run_simultaneously_over_stdio():
    expected_counts = {
        "notes": 10,
        "github": 10,
        "calendar": 8,
    }

    modules = {
        "notes": "mcp_servers.notes.main",
        "github": "mcp_servers.github.main",
        "calendar": "mcp_servers.calendar.main",
    }

    async with AsyncExitStack() as stack:
        clients = {}

        # Start all three independent server processes.
        for name, module in modules.items():
            params = build_server_parameters(module)

            client = await stack.enter_async_context(Client(params, raise_exceptions=True))

            clients[name] = client

        # Discover tools from every connected server.
        for name, client in clients.items():
            result = await client.list_tools()

            assert len(result.tools) == expected_counts[name]

    # AsyncExitStack closes all three client sessions here.


# =========================================================
# 3. Actual Notes invocation over stdio
# =========================================================


@pytest.mark.asyncio
async def test_notes_tool_invocation_over_stdio():
    params = build_server_parameters("mcp_servers.notes.main")

    async with Client(params, raise_exceptions=True) as client:
        result = await client.call_tool(
            "list_notes",
            {},
        )

    payload = extract_payload(result)

    assert result.is_error is not True
    assert payload.get("success") is True
    assert "data" in payload


# =========================================================
# 4. Lifecycle: repeated independent startup and shutdown
# =========================================================


@pytest.mark.asyncio
async def test_notes_server_can_restart_cleanly():
    params = build_server_parameters("mcp_servers.notes.main")

    # First process lifecycle.
    async with Client(params, raise_exceptions=True) as client:
        first_result = await client.list_tools()

    assert len(first_result.tools) == 10

    # A new client must be able to start a fresh process.
    async with Client(params, raise_exceptions=True) as client:
        second_result = await client.list_tools()

    assert len(second_result.tools) == 10
