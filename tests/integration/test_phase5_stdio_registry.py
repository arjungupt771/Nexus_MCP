"""Phase 5 end-to-end: real stdio MCP server, SQLite registry, restart restore."""

from __future__ import annotations

import sys
from pathlib import Path

from core.container import NexusContainer
from core.models import ServerStatus, ToolCallRequest
from core.registry import SQLiteServerConfigStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def notes_payload(db_path: Path) -> dict:
    return {
        "server_id": "notes",
        "name": "Notes MCP Server",
        "description": "Local notes storage",
        "tags": ["notes", "local"],
        "capabilities": ["note-storage"],
        "config": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "mcp_servers.notes.main"],
            "env": {
                "PYTHONPATH": str(PROJECT_ROOT),
                "NEXUS_NOTES_DB_PATH": str(db_path),
            },
        },
    }


async def test_register_connect_call_and_restore_after_restart(tmp_path):
    registry_url = f"sqlite:///{tmp_path / 'nexus.db'}"
    notes_db = tmp_path / "notes.db"

    # --- first run: register at runtime, connect, use a tool ------------------
    first = NexusContainer(config_store=SQLiteServerConfigStore(registry_url))
    await first.initialize()
    report = await first.servers.register(notes_payload(notes_db))
    assert report.status is ServerStatus.DISCONNECTED

    connected = await first.servers.connect("notes")
    assert connected.connected and connected.protocol_capabilities.tools
    assert {"create_note", "get_note"} <= set(connected.tools)
    assert connected.last_discovery_at is not None

    result = await first.executor.execute(
        ToolCallRequest(
            server_id="notes",
            tool_name="create_note",
            arguments={"title": "Phase 5", "content": "registry works"},
        )
    )
    assert result.success is True

    await first.close()

    # --- second run: nothing registered in memory, state comes from SQLite ----
    second = NexusContainer(config_store=SQLiteServerConfigStore(registry_url))
    restore = await second.initialize()

    assert restore.restored == ["notes"] and restore.connected == ["notes"]
    assert restore.failed == {}
    status = await second.servers.status("notes")
    assert status.connected and status.description == "Local notes storage"
    assert "create_note" in status.tools

    # Restarting through the manager works against the real process too.
    restarted = await second.servers.restart("notes")
    assert restarted.connected and "create_note" in restarted.tools

    # Disabling stops the real subprocess and removes its tools.
    disabled = await second.servers.disable("notes")
    assert not disabled.connected and disabled.tools == []

    await second.close()
