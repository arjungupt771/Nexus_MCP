from __future__ import annotations

import pytest

from mcp import Client

from mcp_servers.notes.server import create_notes_server
from mcp_servers.github.server import create_github_server
from mcp_servers.calendar.server import create_calendar_server


SERVER_CASES = [
    pytest.param(create_notes_server, 10, id="notes"),
    pytest.param(create_github_server, 10, id="github"),
    pytest.param(create_calendar_server, 8, id="calendar"),
]


@pytest.mark.parametrize("server_factory, expected_tool_count", SERVER_CASES)
@pytest.mark.asyncio
async def test_server_discovery_contract(
    server_factory,
    expected_tool_count: int,
) -> None:
    """Every Nexus MCP server must expose its expected tools."""
    server = server_factory()

    async with Client(server, raise_exceptions=True) as client:
        result = await client.list_tools()

    tools = result.tools

    assert len(tools) == expected_tool_count

    tool_names = [tool.name for tool in tools]

    # Tool names must be present and unique.
    assert all(isinstance(name, str) and name.strip() for name in tool_names)
    assert len(tool_names) == len(set(tool_names))


@pytest.mark.parametrize("server_factory, expected_tool_count", SERVER_CASES)
@pytest.mark.asyncio
async def test_every_tool_has_valid_metadata(
    server_factory,
    expected_tool_count: int,
) -> None:
    """Every discovered tool must provide usable metadata and an input schema."""
    server = server_factory()

    async with Client(server, raise_exceptions=True) as client:
        result = await client.list_tools()

    assert len(result.tools) == expected_tool_count

    for tool in result.tools:
        assert tool.name.strip()
        assert isinstance(tool.description, str)
        assert tool.description.strip()

        schema = tool.input_schema

        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert isinstance(schema.get("properties"), dict)

        required = schema.get("required", [])
        assert isinstance(required, list)

        # Every required input must exist in the properties definition.
        for field in required:
            assert field in schema["properties"]
