"""Conversation management (8.6): ids, history, summaries, context windows, archival."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from agent.llm.errors import LLMError
from agent.llm.models import Message, Role
from agent.llm.provider import LLMProvider
from agent.memory.errors import ConversationNotFoundError, MemoryValidationError
from agent.memory.models import ConversationRecord, StoredMessage, utcnow
from agent.memory.privacy import SensitiveDataFilter
from agent.memory.settings import MemorySettings
from agent.memory.store import ConversationStore
from agent.memory.text import estimate_tokens

logger = logging.getLogger(__name__)
VALID_ROLES = {"user", "assistant", "system", "tool"}
_FETCH_CAP = 400  # never load unbounded history, whatever the conversation size


class Summarizer(Protocol):
    async def summarize(self, previous: str, messages: list[StoredMessage]) -> str: ...


class ExtractiveSummarizer:
    """Deterministic, offline summarizer: one clipped line per message, newest kept."""

    def __init__(self, max_chars: int = 1500, line_chars: int = 140) -> None:
        self._max = max_chars
        self._line = line_chars

    async def summarize(self, previous: str, messages: list[StoredMessage]) -> str:
        lines = [previous] if previous else []

        for message in messages:
            if message.role not in ("user", "assistant") or not message.content.strip():
                continue

            first = message.content.strip().replace("\n", " ")
            sentence = first.split(". ")[0][: self._line]
            lines.append(f"{'User' if message.role == 'user' else 'Assistant'}: {sentence}")

        text = "\n".join(lines)

        if len(text) > self._max:  # drop the oldest lines first
            kept: list[str] = []
            size = 0

            for line in reversed(text.split("\n")):
                if size + len(line) + 1 > self._max:
                    break

                kept.append(line)
                size += len(line) + 1

            text = "\n".join(reversed(kept))

        return text


class LLMSummarizer:
    """Model-written rolling summary; falls back to the extractive one on failure."""

    PROMPT = (
        "You maintain a rolling summary of a conversation. Merge the previous summary with the new "
        "messages into a concise summary (max 150 words) keeping goals, decisions, and unresolved "
        "items. Omit secrets, credentials and sensitive personal data. Output only the summary."
    )

    def __init__(self, provider: LLMProvider, fallback: Summarizer | None = None) -> None:
        self._provider = provider
        self._fallback = fallback or ExtractiveSummarizer()

    async def summarize(self, previous: str, messages: list[StoredMessage]) -> str:
        transcript = "\n".join(
            f"{m.role}: {m.content[:500]}" for m in messages if m.role in ("user", "assistant")
        )
        prompt = f"Previous summary:\n{previous or '(none)'}\n\nNew messages:\n{transcript}"

        try:
            response = await self._provider.generate(
                [
                    Message(role=Role.SYSTEM, content=self.PROMPT),
                    Message(role=Role.USER, content=prompt),
                ]
            )
        except LLMError:
            logger.warning("LLM summarization failed; using extractive summary")
            return await self._fallback.summarize(previous, messages)

        return response.content.strip() or await self._fallback.summarize(previous, messages)


class ContextWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    messages: list[StoredMessage] = Field(default_factory=list)
    tokens: int = 0
    truncated: bool = False  # older unsummarized messages did not fit the budget

    def to_llm_messages(self) -> list[Message]:
        """History for an agent: summary first (as data), then recent turns."""
        out: list[Message] = []

        if self.summary:
            out += [
                Message(
                    role=Role.USER,
                    content=(
                        "Summary of the earlier conversation (background data):\n" + self.summary
                    ),
                ),
                Message(role=Role.ASSISTANT, content="Understood."),
            ]

        for m in self.messages:
            if m.role == "user":
                out.append(Message(role=Role.USER, content=m.content))
            elif m.role == "assistant":
                out.append(Message(role=Role.ASSISTANT, content=m.content))

        return out


class ConversationManager:
    def __init__(
        self,
        store: ConversationStore,
        settings: MemorySettings | None = None,
        *,
        privacy: SensitiveDataFilter | None = None,
        summarizer: Summarizer | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.settings = settings or MemorySettings()
        self._store = store
        self._privacy = privacy or SensitiveDataFilter(self.settings.privacy)
        self.summarizer: Summarizer = summarizer or ExtractiveSummarizer()
        self._clock = clock

    # ------------------------------------------------------------------ lifecycle

    async def create(
        self,
        user_id: str,
        *,
        title: str = "",
        memory_enabled: bool = True,
        metadata: dict[str, str] | None = None,
    ) -> ConversationRecord:
        now = self._clock()
        return await self._store.create(
            ConversationRecord(
                user_id=user_id,
                title=title[:300],
                memory_enabled=memory_enabled,
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
            )
        )

    async def get(self, user_id: str, conversation_id: str) -> ConversationRecord:
        conversation = await self._store.get(conversation_id)

        # Unknown id and another user's id look identical: no existence leak.
        if conversation is None or conversation.user_id != user_id:
            raise ConversationNotFoundError(
                "Conversation not found.", details={"conversation_id": conversation_id}
            )

        return conversation

    async def get_or_create(
        self, user_id: str, conversation_id: str | None = None, **kwargs: object
    ) -> ConversationRecord:
        if conversation_id is None:
            return await self.create(user_id, **kwargs)  # type: ignore[arg-type]

        return await self.get(user_id, conversation_id)

    async def list(
        self, user_id: str, *, include_archived: bool = False, limit: int = 50, offset: int = 0
    ) -> list[ConversationRecord]:
        return await self._store.list(
            user_id, include_archived=include_archived, limit=limit, offset=offset
        )

    async def rename(self, user_id: str, conversation_id: str, title: str) -> ConversationRecord:
        conversation = await self.get(user_id, conversation_id)
        return await self._store.update(conversation.model_copy(update={"title": title[:300]}))

    async def set_memory_enabled(
        self, user_id: str, conversation_id: str, enabled: bool
    ) -> ConversationRecord:
        conversation = await self.get(user_id, conversation_id)
        return await self._store.update(conversation.model_copy(update={"memory_enabled": enabled}))

    async def archive(self, user_id: str, conversation_id: str) -> ConversationRecord:
        conversation = await self.get(user_id, conversation_id)
        return await self._store.update(
            conversation.model_copy(update={"archived_at": self._clock()})
        )

    async def unarchive(self, user_id: str, conversation_id: str) -> ConversationRecord:
        conversation = await self.get(user_id, conversation_id)
        return await self._store.update(conversation.model_copy(update={"archived_at": None}))

    async def delete(self, user_id: str, conversation_id: str) -> bool:
        await self.get(user_id, conversation_id)
        return await self._store.delete(conversation_id)

    async def clear(self, user_id: str, conversation_id: str) -> int:
        """Remove all messages and the summary, keeping the conversation itself."""
        conversation = await self.get(user_id, conversation_id)
        removed = await self._store.delete_messages(conversation_id)
        await self._store.update(
            conversation.model_copy(
                update={"summary": "", "summarized_through": 0, "message_count": 0}
            )
        )
        return removed

    async def archive_stale(self, user_id: str) -> int:
        days = self.settings.retention.archive_conversations_after_days

        if days is None:
            return 0

        cutoff = self._clock() - timedelta(days=days)
        count = 0

        for conversation in await self._store.list(user_id, include_archived=False, limit=10_000):
            if conversation.updated_at < cutoff:
                await self._store.update(
                    conversation.model_copy(update={"archived_at": self._clock()})
                )
                count += 1

        return count

    # ------------------------------------------------------------------ messages

    async def add_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, str] | None = None,
    ) -> StoredMessage | None:
        """Persist a message (redacted, size-capped); tool messages are skipped by default."""
        if role not in VALID_ROLES:
            raise MemoryValidationError(f"Unknown message role '{role}'.")

        if role == "tool" and not self.settings.persist_tool_messages:
            return None

        await self.get(user_id, conversation_id)
        text = content.strip()

        if not text:
            return None

        if self._privacy.policy.redact_messages:
            text = self._privacy.redact_secrets(text)

        return await self._store.append_message(
            conversation_id,
            role,
            text[: self.settings.max_message_chars],
            metadata,
            created_at=self._clock(),
        )

    async def history(
        self, user_id: str, conversation_id: str, *, limit: int | None = None, after_seq: int = 0
    ) -> list[StoredMessage]:
        await self.get(user_id, conversation_id)
        return await self._store.messages(conversation_id, after_seq=after_seq, limit=limit)

    # ------------------------------------------------------------------ context window

    async def context_window(
        self, user_id: str, conversation_id: str, *, max_tokens: int | None = None
    ) -> ContextWindow:
        """Summary + the newest unsummarized messages that fit the token budget."""
        conversation = await self.get(user_id, conversation_id)
        budget = max_tokens or self.settings.history_token_budget
        summary = conversation.summary
        remaining = budget - (estimate_tokens(summary) if summary else 0)
        newest = await self._store.messages(
            conversation_id,
            after_seq=conversation.summarized_through,
            limit=_FETCH_CAP,
            newest_first=True,
        )
        chosen: list[StoredMessage] = []
        used = 0

        for message in newest:
            cost = estimate_tokens(message.content) + 4

            # Always keep the latest message, even if it alone exceeds the budget.
            if chosen and used + cost > remaining:
                break

            chosen.append(message)
            used += cost

        chosen.reverse()
        return ContextWindow(
            summary=summary,
            messages=chosen,
            # Also truncated when the hard fetch cap cut off even older messages.
            truncated=len(chosen) < len(newest) or len(newest) >= _FETCH_CAP,
            tokens=used + (estimate_tokens(summary) if summary else 0),
        )

    # ------------------------------------------------------------------ summaries & pruning

    async def needs_summary(self, user_id: str, conversation_id: str) -> bool:
        conversation = await self.get(user_id, conversation_id)
        pending = await self._store.messages(
            conversation_id, after_seq=conversation.summarized_through, limit=_FETCH_CAP
        )
        return len(pending) >= self.settings.summarize_after_messages

    async def update_summary(
        self, user_id: str, conversation_id: str, *, force: bool = False
    ) -> ConversationRecord | None:
        """Fold older unsummarized messages into the summary; recent ones stay verbatim."""
        conversation = await self.get(user_id, conversation_id)
        pending = await self._store.messages(
            conversation_id, after_seq=conversation.summarized_through, limit=_FETCH_CAP
        )

        if not force and len(pending) < self.settings.summarize_after_messages:
            return None

        keep = self.settings.keep_recent_messages
        to_fold = pending[:-keep] if len(pending) > keep else ([] if not force else pending)

        if not to_fold:
            return None

        text = await self.summarizer.summarize(conversation.summary, to_fold)
        text = self._privacy.redact_secrets(text)[:2000]
        updated = await self._store.update(
            conversation.model_copy(update={"summary": text, "summarized_through": to_fold[-1].seq})
        )
        await self.prune(user_id, conversation_id)
        return updated

    async def prune(self, user_id: str, conversation_id: str) -> int:
        """Bound stored history: drop old messages that are already summarized."""
        conversation = await self.get(user_id, conversation_id)
        cap = self.settings.retention.max_messages_per_conversation

        if conversation.message_count <= cap:
            return 0

        excess = conversation.message_count - cap
        oldest = await self._store.messages(conversation_id, limit=excess)

        if not oldest:
            return 0

        cutoff = min(oldest[-1].seq, conversation.summarized_through)
        return (
            await self._store.delete_messages_through(conversation_id, cutoff) if cutoff > 0 else 0
        )
