"""MemoryManager: wires the memory subsystem together and exposes user controls (8.7)."""

from __future__ import annotations

from dataclasses import dataclass

from agent.memory.conversation import ConversationManager, Summarizer
from agent.memory.embeddings import Embedder, HashingEmbedder
from agent.memory.extraction import CandidateEvaluator, ImportanceScorer, RuleBasedExtractor
from agent.memory.privacy import SensitiveDataFilter
from agent.memory.service import MemoryService, RetentionReport
from agent.memory.settings import MemorySettings, RetentionPolicy
from agent.memory.short_term import ShortTermMemoryManager
from agent.memory.sqlite import (
    SQLiteBackend,
    SQLiteConversationStore,
    SQLiteMemoryStore,
    SQLiteVectorIndex,
)
from agent.memory.store import ConversationStore, MemoryStore, VectorIndex


@dataclass
class ClearReport:
    memories: int = 0
    messages: int = 0


class MemoryManager:
    def __init__(
        self,
        memory: MemoryService,
        conversations: ConversationManager,
        short_term: ShortTermMemoryManager,
        extractor: RuleBasedExtractor,
        evaluator: CandidateEvaluator,
        settings: MemorySettings,
        backend: SQLiteBackend | None = None,
    ) -> None:
        self.memory = memory
        self.conversations = conversations
        self.short_term = short_term
        self.extractor = extractor
        self.evaluator = evaluator
        self.settings = settings
        self._backend = backend

    @classmethod
    def sqlite(
        cls,
        url: str = "sqlite:///./nexus_memory.db",
        settings: MemorySettings | None = None,
        *,
        embedder: Embedder | None = None,
        vector_index: VectorIndex | None = None,
        summarizer: Summarizer | None = None,
        extractor: RuleBasedExtractor | None = None,
    ) -> MemoryManager:
        """Default local setup: SQLite for everything, hashing embeddings, no external services."""
        return cls.build(
            SQLiteBackend(url),
            settings,
            embedder=embedder,
            vector_index=vector_index,
            summarizer=summarizer,
            extractor=extractor,
        )

    @classmethod
    def in_memory(cls, settings: MemorySettings | None = None, **kwargs: object) -> MemoryManager:
        return cls.build(SQLiteBackend.in_memory(), settings, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def build(
        cls,
        backend: SQLiteBackend,
        settings: MemorySettings | None = None,
        *,
        embedder: Embedder | None = None,
        vector_index: VectorIndex | None = None,
        summarizer: Summarizer | None = None,
        extractor: RuleBasedExtractor | None = None,
        memory_store: MemoryStore | None = None,
        conversation_store: ConversationStore | None = None,
    ) -> MemoryManager:
        settings = settings or MemorySettings()
        privacy = SensitiveDataFilter(settings.privacy)
        embedder = embedder or HashingEmbedder()
        memory = MemoryService(
            memory_store or SQLiteMemoryStore(backend),
            vector_index if vector_index is not None else SQLiteVectorIndex(backend),
            embedder,
            settings,
            privacy=privacy,
        )
        conversations = ConversationManager(
            conversation_store or SQLiteConversationStore(backend),
            settings,
            privacy=privacy,
            summarizer=summarizer,
        )
        short_term = ShortTermMemoryManager(
            max_conversations=settings.short_term_conversations,
            max_messages=settings.short_term_messages,
            max_tool_results=settings.short_term_tool_results,
        )
        evaluator = CandidateEvaluator(settings, privacy, ImportanceScorer())
        return cls(
            memory,
            conversations,
            short_term,
            extractor or RuleBasedExtractor(),
            evaluator,
            settings,
            backend,
        )

    async def initialize(self) -> None:
        if self._backend is not None:
            await self._backend.engine()

    async def close(self) -> None:
        if self._backend is not None:
            await self._backend.close()

    # ------------------------------------------------------------------ controls

    async def clear_conversation_memory(
        self, user_id: str, conversation_id: str, *, include_messages: bool = False
    ) -> ClearReport:
        """Forget what was learned from a conversation (optionally its messages too)."""
        await self.conversations.get(user_id, conversation_id)  # ownership check
        report = ClearReport(
            memories=await self.memory.clear_conversation_memory(user_id, conversation_id)
        )
        self.short_term.drop(conversation_id)

        if include_messages:
            report.messages = await self.conversations.clear(user_id, conversation_id)

        return report

    def configure_retention(self, policy: RetentionPolicy) -> None:
        self.settings = self.settings.model_copy(update={"retention": policy})
        self.memory.configure_retention(policy)
        self.conversations.settings = self.conversations.settings.model_copy(
            update={"retention": policy}
        )

    async def apply_retention(self, user_id: str | None = None) -> RetentionReport:
        report = await self.memory.apply_retention(user_id)

        if user_id:
            await self.conversations.archive_stale(user_id)

        return report
