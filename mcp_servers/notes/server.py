import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from mcp_servers.notes.service import (
    NoteNotFoundError,
    NotesService,
    NoteValidationError,
)
from mcp_servers.notes.storage import DEFAULT_DB_PATH, NotesStore

logger = logging.getLogger("nexus.notes")


def create_notes_server(
    db_path: str | Path | None = None,
) -> MCPServer:
    """Create an independently runnable Notes MCP server."""

    configured_path = db_path or os.getenv("NEXUS_NOTES_DB_PATH") or DEFAULT_DB_PATH

    store = NotesStore(configured_path)
    service = NotesService(store)

    mcp = MCPServer("Nexus Notes Server")

    def execute(operation: Callable[[], Any]) -> dict[str, Any]:
        try:
            return {
                "success": True,
                "data": operation(),
            }
        except NoteNotFoundError as exc:
            return {
                "success": False,
                "error": {
                    "code": "NOT_FOUND",
                    "message": str(exc),
                },
            }
        except NoteValidationError as exc:
            return {
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(exc),
                },
            }
        except Exception:
            logger.exception("Unexpected Notes MCP tool failure")
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "The Notes server encountered an internal error.",
                },
            }

    @mcp.tool()
    def create_note(
        title: str,
        content: str,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a note with a title, content, and optional tags."""
        return execute(lambda: service.create_note(title, content, tags))

    @mcp.tool()
    def get_note(note_id: str) -> dict[str, Any]:
        """Retrieve one note using its unique note ID."""
        return execute(lambda: service.get_note(note_id))

    @mcp.tool()
    def search_notes(
        query: str,
        include_archived: bool = False,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search note titles, contents, and tags."""
        return execute(lambda: service.search_notes(query, include_archived, limit))

    @mcp.tool()
    def update_note(
        note_id: str,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update a note's title, content, or complete tag list."""
        return execute(lambda: service.update_note(note_id, title, content, tags))

    @mcp.tool()
    def delete_note(note_id: str) -> dict[str, Any]:
        """Permanently delete a note and its associated tags."""
        return execute(lambda: service.delete_note(note_id))

    @mcp.tool()
    def list_notes(
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List notes with pagination; archived notes are excluded by default."""
        return execute(lambda: service.list_notes(include_archived, limit, offset))

    @mcp.tool()
    def add_note_tags(
        note_id: str,
        tags: list[str],
    ) -> dict[str, Any]:
        """Add tags to an existing note without replacing its current tags."""
        return execute(lambda: service.add_note_tags(note_id, tags))

    @mcp.tool()
    def retrieve_notes_by_tag(
        tag: str,
        include_archived: bool = False,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Retrieve notes matching a specific tag."""
        return execute(lambda: service.retrieve_notes_by_tag(tag, include_archived, limit))

    @mcp.tool()
    def archive_note(note_id: str) -> dict[str, Any]:
        """Archive a note without permanently deleting it."""
        return execute(lambda: service.archive_note(note_id))

    @mcp.tool()
    def restore_note(note_id: str) -> dict[str, Any]:
        """Restore an archived note."""
        return execute(lambda: service.restore_note(note_id))

    return mcp


# mcp = create_notes_server()
