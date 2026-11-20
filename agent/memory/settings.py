"""Configuration for the memory system (retention, retrieval, conversations)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.memory.models import MemoryType
from agent.memory.privacy import PrivacyPolicy


class RetentionPolicy(BaseModel):
    """How long and how many memories are kept (8.3, 8.7).

    ``None`` means unlimited. Pinned memories are never removed by retention.
    """

    model_config = ConfigDict(extra="forbid")

    max_age_days: dict[MemoryType, float | None] = Field(
        default_factory=lambda: {
            MemoryType.LONG_TERM: None,
            MemoryType.EPISODIC: 90.0,
            MemoryType.SEMANTIC: None,
        }
    )
    max_records_per_user: dict[MemoryType, int | None] = Field(
        default_factory=lambda: {
            MemoryType.LONG_TERM: 1000,
            MemoryType.EPISODIC: 500,
            MemoryType.SEMANTIC: 1000,
        }
    )
    # Records older than this AND below the importance floor are dropped.
    low_importance_floor: float = Field(default=0.3, ge=0, le=1)
    low_importance_max_age_days: float | None = 30.0
    max_messages_per_conversation: int = Field(default=500, ge=10)
    archive_conversations_after_days: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> RetentionPolicy:
        for value in self.max_age_days.values():
            if value is not None and value <= 0:
                raise ValueError("max_age_days values must be positive")

        for value in self.max_records_per_user.values():
            if value is not None and value < 1:
                raise ValueError("max_records_per_user values must be at least 1")

        return self


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=8, ge=1, le=50)
    candidate_limit: int = Field(default=60, ge=5, le=500)
    keyword_weight: float = Field(default=0.5, ge=0)
    semantic_weight: float = Field(default=0.5, ge=0)
    # final = relevance*w_rel + recency*w_rec + importance*w_imp
    relevance_weight: float = Field(default=0.6, ge=0)
    recency_weight: float = Field(default=0.2, ge=0)
    importance_weight: float = Field(default=0.2, ge=0)
    min_relevance: float = Field(default=0.15, ge=0, le=1)
    recency_half_life_days: dict[MemoryType, float] = Field(
        default_factory=lambda: {
            MemoryType.LONG_TERM: 180.0,
            MemoryType.EPISODIC: 21.0,
            MemoryType.SEMANTIC: 365.0,
        }
    )
    standing_preferences: int = Field(default=3, ge=0, le=10)
    standing_preference_min_importance: float = Field(default=0.5, ge=0, le=1)
    touch_on_retrieve: bool = True


class MemorySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    min_importance_to_store: float = Field(default=0.5, ge=0, le=1)
    near_duplicate_similarity: float = Field(default=0.93, ge=0.5, le=1)
    max_content_chars: int = Field(default=1000, ge=50, le=2000)
    max_candidates_per_turn: int = Field(default=8, ge=1, le=50)
    retention: RetentionPolicy = Field(default_factory=RetentionPolicy)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    privacy: PrivacyPolicy = Field(default_factory=PrivacyPolicy)
    # Conversation / context window management (8.6)
    context_token_budget: int = Field(default=1200, ge=100)
    history_token_budget: int = Field(default=3000, ge=200)
    summarize_after_messages: int = Field(default=16, ge=4)
    keep_recent_messages: int = Field(default=8, ge=2)
    max_message_chars: int = Field(default=8000, ge=100)
    persist_tool_messages: bool = False
    # Short-term memory bounds
    short_term_messages: int = Field(default=20, ge=2)
    short_term_tool_results: int = Field(default=20, ge=1)
    short_term_conversations: int = Field(default=200, ge=1)

    @model_validator(mode="after")
    def _validate(self) -> MemorySettings:
        if self.keep_recent_messages >= self.summarize_after_messages:
            raise ValueError("keep_recent_messages must be smaller than summarize_after_messages")

        return self
