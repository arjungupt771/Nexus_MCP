from typing import Any

from mcp_servers.notes.storage import NotesStore


class NoteNotFoundError(Exception):
    pass


class NoteValidationError(ValueError):
    pass


class NotesService:
    MAX_TITLE_LENGTH = 200
    MAX_CONTENT_LENGTH = 50_000
    MAX_TAG_LENGTH = 40
    MAX_TAGS = 20
    MAX_PAGE_SIZE = 100

    def __init__(self, store: NotesStore) -> None:
        self.store = store

    @classmethod
    def _validate_title(cls, title: str) -> str:
        title = title.strip()

        if not title:
            raise NoteValidationError("Title cannot be empty.")

        if len(title) > cls.MAX_TITLE_LENGTH:
            raise NoteValidationError(f"Title cannot exceed {cls.MAX_TITLE_LENGTH} characters.")

        return title

    @classmethod
    def _validate_content(cls, content: str) -> str:
        content = content.strip()

        if not content:
            raise NoteValidationError("Content cannot be empty.")

        if len(content) > cls.MAX_CONTENT_LENGTH:
            raise NoteValidationError(f"Content cannot exceed {cls.MAX_CONTENT_LENGTH} characters.")

        return content

    @classmethod
    def _validate_tags(cls, tags: list[str] | None) -> list[str]:
        if not tags:
            return []

        normalized: list[str] = []

        for raw_tag in tags:
            tag = raw_tag.strip().lower()

            if not tag:
                continue

            if len(tag) > cls.MAX_TAG_LENGTH:
                raise NoteValidationError(f"Tags cannot exceed {cls.MAX_TAG_LENGTH} characters.")

            if tag not in normalized:
                normalized.append(tag)

        if len(normalized) > cls.MAX_TAGS:
            raise NoteValidationError(f"A note can have at most {cls.MAX_TAGS} unique tags.")

        return normalized

    @classmethod
    def _validate_limit(cls, limit: int) -> int:
        if not 1 <= limit <= cls.MAX_PAGE_SIZE:
            raise NoteValidationError(f"Limit must be between 1 and {cls.MAX_PAGE_SIZE}.")

        return limit

    @staticmethod
    def _require_note(note: dict[str, Any] | None) -> dict[str, Any]:
        if note is None:
            raise NoteNotFoundError("The requested note does not exist.")

        return note

    def create_note(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.store.create(
            title=self._validate_title(title),
            content=self._validate_content(content),
            tags=self._validate_tags(tags),
        )

    def get_note(self, note_id: str) -> dict[str, Any]:
        return self._require_note(self.store.get(note_id))

    def list_notes(
        self,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self._validate_limit(limit)

        if offset < 0:
            raise NoteValidationError("Offset cannot be negative.")

        return self.store.list_notes(
            include_archived=include_archived,
            limit=limit,
            offset=offset,
        )

    def search_notes(
        self,
        query: str,
        include_archived: bool = False,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query = query.strip()

        if not query:
            raise NoteValidationError("Search query cannot be empty.")

        if len(query) > 200:
            raise NoteValidationError("Search query cannot exceed 200 characters.")

        self._validate_limit(limit)

        return self.store.search(query, include_archived, limit)

    def update_note(
        self,
        note_id: str,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        if title is None and content is None and tags is None:
            raise NoteValidationError("Provide at least one field to update.")

        validated_title = self._validate_title(title) if title is not None else None
        validated_content = self._validate_content(content) if content is not None else None
        validated_tags = self._validate_tags(tags) if tags is not None else None

        return self._require_note(
            self.store.update(
                note_id,
                validated_title,
                validated_content,
                validated_tags,
            )
        )

    def delete_note(self, note_id: str) -> dict[str, str]:
        if not self.store.delete(note_id):
            raise NoteNotFoundError("The requested note does not exist.")

        return {"note_id": note_id, "message": "Note permanently deleted."}

    def add_note_tags(
        self,
        note_id: str,
        tags: list[str],
    ) -> dict[str, Any]:
        validated_tags = self._validate_tags(tags)

        if not validated_tags:
            raise NoteValidationError("Provide at least one non-empty tag.")

        return self._require_note(self.store.add_tags(note_id, validated_tags))

    def retrieve_notes_by_tag(
        self,
        tag: str,
        include_archived: bool = False,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        tag = tag.strip().lower()

        if not tag:
            raise NoteValidationError("Tag cannot be empty.")

        if len(tag) > self.MAX_TAG_LENGTH:
            raise NoteValidationError(f"Tag cannot exceed {self.MAX_TAG_LENGTH} characters.")

        self._validate_limit(limit)

        return self.store.get_by_tag(tag, include_archived, limit)

    def archive_note(self, note_id: str) -> dict[str, Any]:
        return self._require_note(self.store.set_archived(note_id, True))

    def restore_note(self, note_id: str) -> dict[str, Any]:
        return self._require_note(self.store.set_archived(note_id, False))
