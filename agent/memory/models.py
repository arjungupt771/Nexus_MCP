"""Typed memory domain models.

Four memory categories (8.1):

* short-term - in-process working memory (see ``short_term.py``), never persisted
* long-term  - ``MemoryType.LONG_TERM``: preferences, facts, summaries, context
* episodic   - ``MemoryType.EPISODIC``: tasks, outcomes, decisions, workflow events
* semantic   - ``MemoryType.SEMANTIC``: knowledge, entities and relationships
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_CONTENT_CHARS = 2000


def utcnow() -> datetime:
    return datetime.now(UTC)


class MemoryType(StrEnum):
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryKind(StrEnum):
    # long-term
    PREFERENCE = "preference"
    FACT = "fact"
    SUMMARY = "summary"
    CONTEXT = "context"
    # episodic
    TASK = "task"
    OUTCOME = "outcome"
    DECISION = "decision"
    EVENT = "event"
    # semantic
    KNOWLEDGE = "knowledge"
    RELATIONSHIP = "relationship"


class MemorySource(StrEnum):
    USER_MESSAGE = "user_message"
    SUMMARY = "summary"
    TOOL_RESULT = "tool_result"
    WORKFLOW = "workflow"
    EXPLICIT = "explicit"  # the user (or API caller) asked for it directly
    SYSTEM = "system"


DEFAULT_KIND_FOR_TYPE = {
    MemoryType.LONG_TERM: MemoryKind.FACT,
    MemoryType.EPISODIC: MemoryKind.EVENT,
    MemoryType.SEMANTIC: MemoryKind.KNOWLEDGE,
}
KINDS_BY_TYPE = {
    MemoryType.LONG_TERM: {
        MemoryKind.PREFERENCE,
        MemoryKind.FACT,
        MemoryKind.SUMMARY,
        MemoryKind.CONTEXT,
    },
    MemoryType.EPISODIC: {
        MemoryKind.TASK,
        MemoryKind.OUTCOME,
        MemoryKind.DECISION,
        MemoryKind.EVENT,
    },
    MemoryType.SEMANTIC: {MemoryKind.KNOWLEDGE, MemoryKind.RELATIONSHIP},
}


class MemoryRecord(BaseModel):
    """One persisted memory."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str = Field(min_length=1, max_length=100)
    type: MemoryType
    kind: MemoryKind
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    source: MemorySource = MemorySource.EXPLICIT
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    conversation_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    content_hash: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    last_accessed_at: datetime | None = None
    access_count: int = Field(default=0, ge=0)
    expires_at: datetime | None = None
    pinned: bool = False

    @field_validator("kind")
    @classmethod
    def _kind_matches_type(cls, value: MemoryKind, info: Any) -> MemoryKind:
        memory_type = info.data.get("type")

        if memory_type is not None and value not in KINDS_BY_TYPE[memory_type]:
            raise ValueError(f"kind '{value.value}' does not belong to type '{memory_type.value}'")

        return value


class MemoryCreate(BaseModel):
    """Request to add a memory. Missing type/kind/importance are inferred."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    user_id: str = "default"
    type: MemoryType | None = None
    kind: MemoryKind | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    source: MemorySource = MemorySource.EXPLICIT
    conversation_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    ttl_days: float | None = Field(default=None, gt=0)
    pinned: bool = False


class MemoryUpdate(BaseModel):
    """Partial update; unset fields are unchanged."""

    model_config = ConfigDict(extra="forbid")

    content: str | None = Field(default=None, min_length=1)
    kind: MemoryKind | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, str] | None = None
    pinned: bool | None = None
    ttl_days: float | None = Field(default=None, gt=0)
    clear_expiry: bool = False


class MemoryFilter(BaseModel):
    """Metadata filter applied by every store and by retrieval."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    types: list[MemoryType] | None = None
    kinds: list[MemoryKind] | None = None
    sources: list[MemorySource] | None = None
    conversation_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    min_importance: float | None = None
    max_importance: float | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    pinned: bool | None = None
    include_expired: bool = False


class ScoredMemory(BaseModel):
    """A retrieval hit with a transparent score breakdown."""

    model_config = ConfigDict(extra="forbid")

    record: MemoryRecord
    score: float
    relevance: float = 0.0
    keyword_score: float = 0.0
    semantic_score: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    standing: bool = False  # included as a standing preference, not by query match


class WriteResult(BaseModel):
    """Outcome of adding a memory: created, merged into an existing one, or rejected."""

    model_config = ConfigDict(extra="forbid")

    record: MemoryRecord | None = None
    created: bool = False
    deduplicated: bool = False
    rejected_reason: str | None = None

    @property
    def stored(self) -> bool:
        return self.record is not None


# ---------------------------------------------------------------- conversations


class ConversationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str
    title: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    archived_at: datetime | None = None
    summary: str = ""
    # Messages with seq <= summarized_through are represented by ``summary``.
    summarized_through: int = 0
    message_count: int = 0
    # False = no long-term memory is read or written for this conversation.
    memory_enabled: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def archived(self) -> bool:
        return self.archived_at is not None


class StoredMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    seq: int
    role: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
