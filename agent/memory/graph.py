"""LangGraph memory integration (8.5).

The memory graph wraps an existing workflow - the Phase 6 ``Agent`` or the
Phase 7 ``Orchestrator`` - as its *execute* node, so neither is modified:

    load_context -> retrieve_memories -> prepare_context -> run_agent
        -> extract_memories -> evaluate_importance -> persist_memories
        -> update_conversation

Routing skips memory work when it is disabled (globally or per conversation),
when the user asked not to be remembered, or when the run failed. Memory
failures are recorded in ``memory_errors`` and never prevent an answer.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from types import SimpleNamespace
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field

from agent.llm.models import Message, Role
from agent.memory.errors import MemorySystemError
from agent.memory.extraction import Evaluation, MemoryCandidate
from agent.memory.manager import MemoryManager
from agent.memory.models import ConversationRecord, ScoredMemory
from agent.memory.retrieval import MemoryContext

logger = logging.getLogger(__name__)

N_LOAD = "load_context"
N_RETRIEVE = "retrieve_memories"
N_PREPARE = "prepare_context"
N_RUN = "run_agent"
N_EXTRACT = "extract_memories"
N_EVALUATE = "evaluate_importance"
N_PERSIST = "persist_memories"
N_UPDATE = "update_conversation"


class AgentRunner(Protocol):
    """Anything with ``run(text, history=..., request_id=...)``: ``Agent`` or ``Orchestrator``."""

    async def run(
        self, request: str, /, *, history: list[Message] | None = ..., request_id: str | None = ...
    ) -> Any: ...


class MemoryGraphState(TypedDict, total=False):
    user_id: str
    message: str
    request_id: str
    conversation_id: str
    conversation: ConversationRecord | None
    persistence_ok: bool
    memory_active: bool
    history: list[Message]
    retrieved: list[ScoredMemory]
    memory_context: MemoryContext | None
    agent_history: list[Message]
    result: Any
    response: str
    agent_error: dict[str, Any] | None
    candidates: list[MemoryCandidate]
    evaluation: Evaluation | None
    persisted: list[str]
    deduplicated: int
    summary_updated: bool
    memory_errors: list[dict[str, Any]]


class MemoryRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    conversation_id: str
    request_id: str
    response: str
    stop_reason: str
    retrieved: list[ScoredMemory] = Field(default_factory=list)
    persisted: list[str] = Field(default_factory=list)
    deduplicated: int = 0
    rejected: list[dict[str, str]] = Field(default_factory=list)
    summary_updated: bool = False
    memory_errors: list[dict[str, Any]] = Field(default_factory=list)
    memory_used: bool = False
    # The wrapped workflow's own result (AgentResult / OrchestrationResult).
    inner: Any = Field(default=None, exclude=True)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _outcomes_of(result: Any) -> list[Any]:
    """Normalize tool outcomes from an AgentResult or an OrchestrationResult."""
    outcomes = getattr(result, "tool_outcomes", None)

    if outcomes:
        return list(outcomes)

    records = getattr(result, "records", None) or {}
    return [
        SimpleNamespace(
            tool_name=r.tool_name or (r.tool or ""),
            spec_name=r.tool or "",
            server_id=r.server_id,
            success=r.succeeded,
            error_code=r.error_code,
            arguments=r.resolved_arguments,
            data=r.data,
        )
        for r in records.values()
        if getattr(r, "tool", None)
    ]


class MemoryNodes:
    def __init__(self, manager: MemoryManager, runner: AgentRunner) -> None:
        self._m = manager
        self._runner = runner

    @staticmethod
    def _error(state: MemoryGraphState, stage: str, exc: Exception) -> list[dict[str, Any]]:
        info = (
            exc.to_dict()
            if isinstance(exc, MemorySystemError)
            else {"code": type(exc).__name__, "message": str(exc)[:200]}
        )
        return [*state.get("memory_errors", []), {"stage": stage, **info}]

    # 1 ------------------------------------------------------------------
    async def load_context(self, state: MemoryGraphState) -> MemoryGraphState:
        """Open (or create) the conversation; load summary + recent turns within budget."""
        user_id = state["user_id"]
        update: MemoryGraphState = {
            "memory_errors": [],
            "persisted": [],
            "deduplicated": 0,
            "summary_updated": False,
            "candidates": [],
            "history": [],
            "persistence_ok": True,
            "memory_active": self._m.settings.enabled,
        }

        try:
            conversation = await self._m.conversations.get_or_create(
                user_id, state.get("conversation_id")
            )
            window = await self._m.conversations.context_window(user_id, conversation.id)
        except MemorySystemError as exc:
            if state.get("conversation_id") and exc.code == "CONVERSATION_NOT_FOUND":
                raise  # a bad conversation id is the caller's error, not a storage fault

            # Storage trouble: still answer, just without persistence or memory.
            return {
                **update,
                "conversation_id": state.get("conversation_id") or uuid.uuid4().hex,
                "conversation": None,
                "persistence_ok": False,
                "memory_active": False,
                "memory_errors": self._error(state, "load_context", exc),
            }

        short = self._m.short_term.get(conversation.id)
        short.clear_execution()
        short.set_context(request_id=state["request_id"])

        for stored in window.messages:
            short.add_message(stored.role, stored.content)

        return {
            **update,
            "conversation_id": conversation.id,
            "conversation": conversation,
            "history": window.to_llm_messages(),
            "memory_active": self._m.settings.enabled and conversation.memory_enabled,
        }

    # 2 ------------------------------------------------------------------
    async def retrieve_memories(self, state: MemoryGraphState) -> MemoryGraphState:
        """Retrieve only the memories relevant to this task."""
        try:
            retrieved = await self._m.memory.retrieve(state["user_id"], state["message"])
            from agent.memory.retrieval import build_context

            context = build_context(retrieved, self._m.settings.context_token_budget)
        except Exception as exc:  # retrieval problems must not block the answer
            return {
                "retrieved": [],
                "memory_context": None,
                "memory_errors": self._error(state, "retrieve", exc),
            }

        included = set(context.included)
        return {
            "retrieved": [h for h in retrieved if h.record.id in included],
            "memory_context": context,
        }

    # 3 ------------------------------------------------------------------
    async def prepare_context(self, state: MemoryGraphState) -> MemoryGraphState:
        """Memory goes in as clearly-labelled background data, ahead of the conversation."""
        history = list(state.get("history", []))
        context = state.get("memory_context")

        if context and context.text:
            history = [
                Message(role=Role.USER, content=context.text),
                Message(
                    role=Role.ASSISTANT,
                    content="Understood. I will treat this as background information only.",
                ),
                *history,
            ]

        return {"agent_history": history}

    # 4 ------------------------------------------------------------------
    async def run_agent(self, state: MemoryGraphState) -> MemoryGraphState:
        try:
            result = await self._runner.run(
                state["message"],
                history=state.get("agent_history") or None,
                request_id=state["request_id"],
            )
        except Exception as exc:
            logger.exception("Wrapped agent raised")
            return {
                "result": None,
                "response": "",
                "agent_error": {"code": type(exc).__name__, "message": str(exc)[:300]},
            }

        short = self._m.short_term.get(state["conversation_id"])

        for outcome in _outcomes_of(result):
            short.add_tool_result(
                getattr(outcome, "tool_name", None) or "unknown",
                server_id=getattr(outcome, "server_id", None),
                success=bool(getattr(outcome, "success", False)),
                preview=str(getattr(outcome, "data", "") or ""),
            )

        return {
            "result": result,
            "response": getattr(result, "response", "") or "",
            "agent_error": None,
        }

    # 5 ------------------------------------------------------------------
    async def extract_memories(self, state: MemoryGraphState) -> MemoryGraphState:
        message = state["message"]

        if self._m.memory.privacy.is_restricted_request(message):
            return {"candidates": []}  # the user asked not to be remembered

        extractor = self._m.extractor
        candidates = list(
            await _maybe_await(
                extractor.extract_from_message(message, conversation_id=state["conversation_id"])
            )
        )
        result = state.get("result")

        if result is not None:
            candidates += extractor.extract_from_tool_outcomes(_outcomes_of(result))
            candidates += extractor.extract_from_workflow(message, result)

        for candidate in candidates:
            candidate.conversation_id = candidate.conversation_id or state["conversation_id"]

        return {"candidates": candidates}

    # 6 ------------------------------------------------------------------
    async def evaluate_importance(self, state: MemoryGraphState) -> MemoryGraphState:
        return {"evaluation": self._m.evaluator.evaluate(state.get("candidates", []))}

    # 7 ------------------------------------------------------------------
    async def persist_memories(self, state: MemoryGraphState) -> MemoryGraphState:
        evaluation = state.get("evaluation")

        if evaluation is None or not evaluation.approved:
            return {}

        try:
            results = await self._m.memory.add_candidates(
                state["user_id"], evaluation.approved, conversation_id=state["conversation_id"]
            )
        except Exception as exc:
            return {"memory_errors": self._error(state, "persist", exc)}

        return {
            "persisted": [r.record.id for r in results if r.created and r.record],
            "deduplicated": sum(1 for r in results if r.deduplicated),
        }

    # 8 ------------------------------------------------------------------
    async def update_conversation(self, state: MemoryGraphState) -> MemoryGraphState:
        """Persist the turn, then fold old messages into the rolling summary."""
        if not state.get("persistence_ok", True):
            return {}

        user_id, conversation_id = state["user_id"], state["conversation_id"]
        short = self._m.short_term.get(conversation_id)

        try:
            await self._m.conversations.add_message(
                user_id, conversation_id, "user", state["message"]
            )
            short.add_message("user", state["message"])

            if state.get("response"):
                await self._m.conversations.add_message(
                    user_id, conversation_id, "assistant", state["response"]
                )
                short.add_message("assistant", state["response"])

            updated = None

            if await self._m.conversations.needs_summary(user_id, conversation_id):
                updated = await self._m.conversations.update_summary(user_id, conversation_id)

            conversation = state.get("conversation")

            if (
                updated
                and state.get("memory_active")
                and conversation
                and conversation.memory_enabled
            ):
                await self._m.memory.upsert_summary(user_id, conversation_id, updated.summary)

            return {"summary_updated": updated is not None}
        except Exception as exc:
            return {"memory_errors": self._error(state, "update_conversation", exc)}


def build_memory_graph(nodes: MemoryNodes) -> CompiledStateGraph:
    graph = StateGraph(MemoryGraphState)

    for name, fn in (
        (N_LOAD, nodes.load_context),
        (N_RETRIEVE, nodes.retrieve_memories),
        (N_PREPARE, nodes.prepare_context),
        (N_RUN, nodes.run_agent),
        (N_EXTRACT, nodes.extract_memories),
        (N_EVALUATE, nodes.evaluate_importance),
        (N_PERSIST, nodes.persist_memories),
        (N_UPDATE, nodes.update_conversation),
    ):
        graph.add_node(name, fn)

    def after_load(state: MemoryGraphState) -> str:
        return N_RETRIEVE if state.get("memory_active") else N_PREPARE

    def after_run(state: MemoryGraphState) -> str:
        return (
            N_EXTRACT if state.get("memory_active") and not state.get("agent_error") else N_UPDATE
        )

    def after_evaluate(state: MemoryGraphState) -> str:
        evaluation = state.get("evaluation")
        return N_PERSIST if evaluation and evaluation.approved else N_UPDATE

    graph.add_edge(START, N_LOAD)
    graph.add_conditional_edges(N_LOAD, after_load, {N_RETRIEVE: N_RETRIEVE, N_PREPARE: N_PREPARE})
    graph.add_edge(N_RETRIEVE, N_PREPARE)
    graph.add_edge(N_PREPARE, N_RUN)
    graph.add_conditional_edges(N_RUN, after_run, {N_EXTRACT: N_EXTRACT, N_UPDATE: N_UPDATE})
    graph.add_edge(N_EXTRACT, N_EVALUATE)
    graph.add_conditional_edges(
        N_EVALUATE, after_evaluate, {N_PERSIST: N_PERSIST, N_UPDATE: N_UPDATE}
    )
    graph.add_edge(N_PERSIST, N_UPDATE)
    graph.add_edge(N_UPDATE, END)
    return graph.compile()


class MemoryAgent:
    """An agent/orchestrator with conversation history and long-term memory."""

    def __init__(self, manager: MemoryManager, runner: AgentRunner) -> None:
        self.manager = manager
        self.runner = runner
        self._graph = build_memory_graph(MemoryNodes(manager, runner))

    async def run(
        self,
        user_id: str,
        message: str,
        *,
        conversation_id: str | None = None,
        request_id: str | None = None,
    ) -> MemoryRunResult:
        if not message or not message.strip():
            raise ValueError("message cannot be empty.")

        state: MemoryGraphState = {
            "user_id": user_id,
            "message": message.strip(),
            "request_id": request_id or uuid.uuid4().hex,
        }

        if conversation_id:
            state["conversation_id"] = conversation_id

        final = await self._graph.ainvoke(state, config={"recursion_limit": 30})
        inner = final.get("result")
        evaluation = final.get("evaluation")
        stop = (
            "error" if final.get("agent_error") else str(getattr(inner, "stop_reason", "completed"))
        )
        return MemoryRunResult(
            conversation_id=final["conversation_id"],
            request_id=final["request_id"],
            response=final.get("response", ""),
            stop_reason=stop,
            retrieved=final.get("retrieved", []),
            persisted=final.get("persisted", []),
            deduplicated=final.get("deduplicated", 0),
            rejected=[
                {"preview": r.preview, "reason": r.reason}
                for r in (evaluation.rejected if evaluation else [])
            ],
            summary_updated=final.get("summary_updated", False),
            memory_errors=final.get("memory_errors", []),
            memory_used=bool(final.get("memory_active")),
            inner=inner,
        )
