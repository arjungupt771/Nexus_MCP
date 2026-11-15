import pytest

from mcp_servers.notes.service import (
    NoteNotFoundError,
    NotesService,
    NoteValidationError,
)
from mcp_servers.notes.storage import NotesStore


@pytest.fixture
def service(tmp_path):
    store = NotesStore(tmp_path / "test_notes.db")
    return NotesService(store)


def test_create_and_get_note(service):
    created = service.create_note(
        "Nexus MCP",
        "Build the Notes MCP server.",
        ["AI", "MCP"],
    )

    fetched = service.get_note(created["id"])

    assert fetched["id"] == created["id"]
    assert fetched["title"] == "Nexus MCP"
    assert fetched["tags"] == ["ai", "mcp"]
    assert fetched["archived"] is False


def test_update_note(service):
    note = service.create_note("Old title", "Old content")

    updated = service.update_note(
        note["id"],
        title="New title",
        content="New content",
        tags=["updated"],
    )

    assert updated["title"] == "New title"
    assert updated["content"] == "New content"
    assert updated["tags"] == ["updated"]


def test_search_notes(service):
    service.create_note(
        "LangGraph architecture",
        "Design the orchestration graph.",
        ["agent"],
    )
    service.create_note(
        "Meeting notes",
        "Discuss project milestones.",
        ["planning"],
    )

    results = service.search_notes("LangGraph")

    assert len(results) == 1
    assert results[0]["title"] == "LangGraph architecture"


def test_add_tags_without_replacing_existing_tags(service):
    note = service.create_note(
        "Tagged note",
        "Testing tags.",
        ["python"],
    )

    updated = service.add_note_tags(
        note["id"],
        ["testing", "python"],
    )

    assert updated["tags"] == ["python", "testing"]


def test_retrieve_notes_by_tag(service):
    service.create_note("First", "Content one", ["backend"])
    service.create_note("Second", "Content two", ["frontend"])

    results = service.retrieve_notes_by_tag("backend")

    assert len(results) == 1
    assert results[0]["title"] == "First"


def test_archive_and_restore_note(service):
    note = service.create_note("Archive test", "Content")

    archived = service.archive_note(note["id"])

    assert archived["archived"] is True
    assert service.list_notes() == []

    restored = service.restore_note(note["id"])

    assert restored["archived"] is False
    assert len(service.list_notes()) == 1


def test_delete_note(service):
    note = service.create_note("Delete test", "Content")

    service.delete_note(note["id"])

    with pytest.raises(NoteNotFoundError):
        service.get_note(note["id"])


def test_reject_empty_title(service):
    with pytest.raises(NoteValidationError):
        service.create_note("   ", "Some content")


def test_reject_empty_content(service):
    with pytest.raises(NoteValidationError):
        service.create_note("A title", "   ")


def test_reject_invalid_limit(service):
    with pytest.raises(NoteValidationError):
        service.list_notes(limit=500)


def test_reject_update_without_fields(service):
    note = service.create_note("Title", "Content")

    with pytest.raises(NoteValidationError):
        service.update_note(note["id"])


def test_unknown_note_returns_not_found(service):
    with pytest.raises(NoteNotFoundError):
        service.get_note("does-not-exist")
