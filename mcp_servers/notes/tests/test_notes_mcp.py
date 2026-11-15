import pytest
from mcp import Client

from mcp_servers.notes.server import create_notes_server


@pytest.mark.asyncio
async def test_notes_tool_discovery(tmp_path):
    server = create_notes_server(tmp_path / "discovery.db")

    async with Client(server, raise_exceptions=True) as client:
        result = await client.list_tools()

        names = {tool.name for tool in result.tools}

        expected = {
            "create_note",
            "get_note",
            "search_notes",
            "update_note",
            "delete_note",
            "list_notes",
            "add_note_tags",
            "retrieve_notes_by_tag",
            "archive_note",
            "restore_note",
        }

        assert expected.issubset(names)


@pytest.mark.asyncio
async def test_create_and_retrieve_note_over_mcp(tmp_path):
    server = create_notes_server(tmp_path / "create_get.db")

    async with Client(server, raise_exceptions=True) as client:
        create_result = await client.call_tool(
            "create_note",
            {
                "title": "MCP integration test",
                "content": "This note was created through the MCP protocol.",
                "tags": ["integration", "mcp"],
            },
        )

        assert not create_result.is_error
        assert create_result.structured_content is not None

        created = create_result.structured_content

        assert created["success"] is True
        note_id = created["data"]["id"]

        get_result = await client.call_tool(
            "get_note",
            {"note_id": note_id},
        )

        assert not get_result.is_error
        assert get_result.structured_content is not None

        retrieved = get_result.structured_content

        assert retrieved["success"] is True
        assert retrieved["data"]["title"] == "MCP integration test"
        assert retrieved["data"]["id"] == note_id


@pytest.mark.asyncio
async def test_validation_error_is_returned_safely(tmp_path):
    server = create_notes_server(tmp_path / "validation.db")

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "create_note",
            {
                "title": "   ",
                "content": "This should fail.",
            },
        )

        assert not result.is_error
        assert result.structured_content is not None

        payload = result.structured_content

        assert payload["success"] is False
        assert payload["error"]["code"] == "VALIDATION_ERROR"
        assert "Title cannot be empty" in payload["error"]["message"]
