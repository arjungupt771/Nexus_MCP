"""Structured memory errors (names avoid shadowing the builtin ``MemoryError``)."""

from __future__ import annotations

from typing import Any, ClassVar


class MemorySystemError(Exception):
    code: ClassVar[str] = "MEMORY_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class MemoryValidationError(MemorySystemError):
    code = "MEMORY_INVALID"


class MemoryNotFoundError(MemorySystemError):
    code = "MEMORY_NOT_FOUND"


class MemoryConflictError(MemorySystemError):
    """The change would duplicate another memory."""

    code = "MEMORY_CONFLICT"


class MemoryStorageError(MemorySystemError):
    """The backing store failed (disk, locked database, ...)."""

    code = "MEMORY_STORAGE_FAILED"


class ConversationNotFoundError(MemorySystemError):
    code = "CONVERSATION_NOT_FOUND"
