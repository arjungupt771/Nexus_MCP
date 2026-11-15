import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_server_parameters(db_path: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_servers.notes.main"],
        env={
            "PYTHONPATH": str(PROJECT_ROOT),
            "NEXUS_NOTES_DB_PATH": str(db_path),
        },
    )


@pytest.mark.asyncio
async def test_notes_server_over_stdio(tmp_path):
    db_path = tmp_path / "stdio_notes.db"
    server_parameters = get_server_parameters(db_path)

    # First process: create a note.
    async with Client(server_parameters) as client:
        discovery = await client.list_tools()
        names = {tool.name for tool in discovery.tools}

        assert "create_note" in names
        assert "get_note" in names

        result = await client.call_tool(
            "create_note",
            {
                "title": "Stdio Integration",
                "content": "This note was created through a separate process.",
                "tags": ["stdio", "integration"],
            },
        )

        assert result.is_error is False
        assert result.structured_content["success"] is True

        note_id = result.structured_content["data"]["id"]

    # The first server process has now shut down.

    # Second process: retrieve the note from the same database.
    async with Client(server_parameters) as client:
        result = await client.call_tool(
            "get_note",
            {"note_id": note_id},
        )

        assert result.is_error is False
        assert result.structured_content["success"] is True
        assert result.structured_content["data"]["title"] == ("Stdio Integration")
