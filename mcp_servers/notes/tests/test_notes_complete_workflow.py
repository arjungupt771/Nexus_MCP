import pytest
from mcp import Client

from mcp_servers.notes.server import create_notes_server


@pytest.mark.asyncio
async def test_all_notes_tools_through_mcp(tmp_path):
    server = create_notes_server(tmp_path / "complete_workflow.db")

    async with Client(server, raise_exceptions=True) as client:
        # 1. Discover tools and inspect their schemas.
        discovery = await client.list_tools()
        tools = {tool.name: tool for tool in discovery.tools}

        expected_tools = {
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

        assert expected_tools.issubset(tools)

        for name in expected_tools:
            assert tools[name].input_schema["type"] == "object"
            assert tools[name].description

        assert "title" in tools["create_note"].input_schema["properties"]
        assert "content" in tools["create_note"].input_schema["properties"]

        # 2. Create a note.
        result = await client.call_tool(
            "create_note",
            {
                "title": "Nexus MCP Architecture",
                "content": "Build a dynamic multi-server agent platform.",
                "tags": ["architecture", "agent"],
            },
        )

        assert result.structured_content["success"] is True
        note_id = result.structured_content["data"]["id"]

        # 3. Retrieve the note.
        result = await client.call_tool(
            "get_note",
            {"note_id": note_id},
        )

        assert result.structured_content["data"]["title"] == ("Nexus MCP Architecture")

        # 4. Search notes.
        result = await client.call_tool(
            "search_notes",
            {"query": "multi-server"},
        )

        assert len(result.structured_content["data"]) == 1

        # 5. List notes.
        result = await client.call_tool(
            "list_notes",
            {"include_archived": False, "limit": 10, "offset": 0},
        )

        assert len(result.structured_content["data"]) == 1

        # 6. Update note.
        result = await client.call_tool(
            "update_note",
            {
                "note_id": note_id,
                "title": "Nexus MCP Architecture - Updated",
            },
        )

        assert result.structured_content["data"]["title"].endswith("Updated")

        # 7. Add tags without replacing existing tags.
        result = await client.call_tool(
            "add_note_tags",
            {
                "note_id": note_id,
                "tags": ["python", "mcp"],
            },
        )

        assert {"architecture", "agent", "python", "mcp"}.issubset(
            set(result.structured_content["data"]["tags"])
        )

        # 8. Retrieve notes by tag.
        result = await client.call_tool(
            "retrieve_notes_by_tag",
            {"tag": "python"},
        )

        assert len(result.structured_content["data"]) == 1

        # 9. Archive the note.
        result = await client.call_tool(
            "archive_note",
            {"note_id": note_id},
        )

        assert result.structured_content["data"]["archived"] is True

        # Archived notes should be hidden by default.
        result = await client.call_tool(
            "list_notes",
            {"include_archived": False},
        )

        assert result.structured_content["data"] == []

        # 10. Restore the note.
        result = await client.call_tool(
            "restore_note",
            {"note_id": note_id},
        )

        assert result.structured_content["data"]["archived"] is False

        # Finally, delete the note.
        result = await client.call_tool(
            "delete_note",
            {"note_id": note_id},
        )

        assert result.structured_content["success"] is True

        # Verify that it no longer exists.
        result = await client.call_tool(
            "get_note",
            {"note_id": note_id},
        )

        assert result.structured_content["success"] is False
        assert result.structured_content["error"]["code"] == "NOT_FOUND"
