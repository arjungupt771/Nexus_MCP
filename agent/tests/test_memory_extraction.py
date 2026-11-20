from __future__ import annotations

from types import SimpleNamespace

import pytest
from support_memory import outcome
from support_orch import ScriptedProvider

from agent.llm import LLMRateLimitError
from agent.memory import (
    CandidateEvaluator,
    ImportanceScorer,
    LLMMemoryExtractor,
    MemoryCandidate,
    MemoryKind,
    MemorySettings,
    MemorySource,
    MemoryType,
    RuleBasedExtractor,
    SensitiveDataFilter,
)
from agent.memory.extraction import _ExtractedItem, _ExtractedMemories, to_third_person

X = RuleBasedExtractor()


def contents(text: str) -> list[str]:
    return [c.content for c in X.extract_from_message(text)]


# ------------------------------------------------------------------ user messages


@pytest.mark.parametrize(
    ("message", "expected", "memory_type", "kind"),
    [
        (
            "I prefer dark mode",
            "User prefers dark mode",
            MemoryType.LONG_TERM,
            MemoryKind.PREFERENCE,
        ),
        (
            "I really love hiking in the mountains",
            "User loves hiking in the mountains",
            MemoryType.LONG_TERM,
            MemoryKind.PREFERENCE,
        ),
        (
            "My favorite editor is Neovim.",
            "User's favorite editor is Neovim",
            MemoryType.LONG_TERM,
            MemoryKind.PREFERENCE,
        ),
        (
            "My name is Ada Lovelace",
            "User's name is Ada Lovelace",
            MemoryType.LONG_TERM,
            MemoryKind.FACT,
        ),
        (
            "Call me Ada",
            "User prefers to be called Ada",
            MemoryType.LONG_TERM,
            MemoryKind.PREFERENCE,
        ),
        ("I work at Acme Corp", "User works at Acme Corp", MemoryType.LONG_TERM, MemoryKind.FACT),
        (
            "I'm a backend engineer",
            "User is a backend engineer",
            MemoryType.LONG_TERM,
            MemoryKind.FACT,
        ),
        ("I live in Pune", "User lives in Pune", MemoryType.LONG_TERM, MemoryKind.FACT),
        ("I'm based in Noida", "User is based in Noida", MemoryType.LONG_TERM, MemoryKind.FACT),
        ("My timezone is IST", "User's timezone is IST", MemoryType.LONG_TERM, MemoryKind.FACT),
        (
            "Sam Lee is my manager",
            "Sam Lee is the user's manager",
            MemoryType.SEMANTIC,
            MemoryKind.RELATIONSHIP,
        ),
        (
            "My teammate is Priya Rao",
            "Priya Rao is the user's teammate",
            MemoryType.SEMANTIC,
            MemoryKind.RELATIONSHIP,
        ),
        (
            "Let's go with PostgreSQL",
            "Decision: go with PostgreSQL",
            MemoryType.EPISODIC,
            MemoryKind.DECISION,
        ),
        (
            "I've decided to migrate to Rust",
            "Decision: migrate to Rust",
            MemoryType.EPISODIC,
            MemoryKind.DECISION,
        ),
        (
            "We'll use Redis for caching",
            "Decision: use Redis for caching",
            MemoryType.EPISODIC,
            MemoryKind.DECISION,
        ),
        (
            "Always answer in metric units",
            "User wants the assistant to always answer in metric units",
            MemoryType.LONG_TERM,
            MemoryKind.PREFERENCE,
        ),
    ],
)
def test_statements_become_typed_third_person_memories(message, expected, memory_type, kind):
    (candidate,) = X.extract_from_message(message, conversation_id="c1")

    assert candidate.content == expected and (candidate.type, candidate.kind) == (memory_type, kind)
    assert candidate.source is MemorySource.USER_MESSAGE and candidate.conversation_id == "c1"


def test_relationships_carry_structured_metadata():
    (c,) = X.extract_from_message("Sam Lee is my manager")

    assert c.metadata == {"subject": "Sam Lee", "predicate": "is_user's_manager", "object": "user"}


@pytest.mark.parametrize(
    "message",
    [
        "What is the weather today?",
        "Can you help me write an email",
        "If I were rich I would travel",
        "How do I deploy to production",
        "I like it",
        "Hello there",
        "ok thanks",
        "I'm a bit tired today",
        "Please summarise the attached report for me",
        "x " * 200,
    ],
)
def test_questions_chitchat_hypotheticals_and_requests_are_not_remembered(message):
    assert X.extract_from_message(message) == []


def test_compound_messages_yield_separate_memories_without_duplicates():
    assert contents(
        "I prefer vim; my favorite shell is zsh. I live in Pune, and I work at Acme. I live in Pune"
    ) == [
        "User prefers vim",
        "User's favorite shell is zsh",
        "User lives in Pune",
        "User works at Acme",
    ]
    assert contents("I like tea and coffee") == ["User likes tea and coffee"]


def test_explicit_requests_are_flagged_and_normalized():
    (a,) = X.extract_from_message("Please remember that I deploy on Fridays")
    (b,) = X.extract_from_message("Remember: my manager is Sam")

    assert (
        a.explicit and a.source is MemorySource.EXPLICIT and a.content == "User deploys on Fridays"
    )
    assert b.explicit and "User's manager is Sam" in b.content


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "I go home and I pass tests and I fix bugs",
            "User goes home and User passes tests and User fixes bugs",
        ),
        ("I'm happy and I have time and my cat", "User is happy and User has time and User's cat"),
        ("tell me about it", "tell the user about it"),
    ],
)
def test_third_person_conversion(text, expected):
    assert to_third_person(text) == expected


# ------------------------------------------------------------------ scoring


def candidate(content="User prefers dark mode and tabs", kind=MemoryKind.PREFERENCE, **kwargs):
    return MemoryCandidate(content, MemoryType.LONG_TERM, kind, MemorySource.USER_MESSAGE, **kwargs)


def test_importance_scoring_is_bounded_and_explainable():
    s = ImportanceScorer()
    base = s.score(candidate())

    assert 0 < base < 1
    assert s.score(candidate(explicit=True)) > base + 0.2 and s.score(candidate(explicit=True)) <= 1
    assert s.score(candidate("User likes tea")) < base  # very short
    assert s.score(candidate(metadata={"status": "failed"})) > base
    assert s.score(candidate(metadata={"directive": "true"})) > base
    assert s.score(
        candidate("Ran notes.create_note and github.search", metadata={"servers": "github,notes"})
    ) > s.score(candidate("Ran notes.create_note and github.search"))
    assert (
        s.score(candidate("Ada works at Acme Corp since 2019", kind=MemoryKind.FACT))
        > s.score(candidate("Ada works at Acme Corp since long", kind=MemoryKind.FACT)) - 0.001
    )
    assert s.score(candidate(kind=MemoryKind.PREFERENCE)) > s.score(
        candidate(kind=MemoryKind.EVENT)
    )


# ------------------------------------------------------------------ tool results & workflows


def test_only_actions_are_remembered_from_tool_results_never_raw_reads():
    found = X.extract_from_tool_outcomes(
        [
            outcome("search_notes", args={"query": "secret plans"}),
            outcome("list_events", server="calendar"),
            outcome("create_note", args={"title": "Weekly plan", "content": "long private body"}),
            outcome(
                "create_event",
                server="calendar",
                ok=False,
                args={"summary": "Review"},
                code="TOOL_TIMEOUT",
            ),
        ]
    )

    assert [c.content for c in found] == [
        "Ran notes.create_note ('Weekly plan') successfully.",
        "Attempt to run calendar.create_event ('Review') failed (TOOL_TIMEOUT).",
    ]
    assert all(c.type is MemoryType.EPISODIC and c.kind is MemoryKind.OUTCOME for c in found)
    assert found[1].metadata["status"] == "failed" and found[0].metadata["server_id"] == "notes"
    assert "long private body" not in " ".join(
        c.content for c in found
    )  # argument bodies are not copied


def test_workflow_episodes_are_created_only_when_tools_were_used():
    used = SimpleNamespace(
        stop_reason="partial",
        request_id="r1",
        replans=1,
        aggregated=SimpleNamespace(servers_used=["github", "notes"], failed={"s2": "x"}),
        records={"s1": 1, "s2": 2, "s3": 3},
    )
    (episode,) = X.extract_from_workflow("Find repos and save a note", used)

    assert episode.type is MemoryType.EPISODIC and episode.kind is MemoryKind.TASK
    assert (
        episode.content == "Task 'Find repos and save a note' finished as partial; "
        "used github, notes (3 step(s)), 1 failed."
    )
    assert episode.metadata == {
        "request_id": "r1",
        "status": "partial",
        "servers": "github,notes",
        "steps": "3",
        "replans": "1",
    }

    chat = SimpleNamespace(
        stop_reason="no_tools_needed",
        request_id="r2",
        aggregated=None,
        records={},
        tool_outcomes=[],
    )
    assert X.extract_from_workflow("hello", chat) == []

    agent_result = SimpleNamespace(
        stop_reason="completed", request_id="r3", tool_outcomes=[outcome("create_note")]
    )
    assert "used notes (1 step(s))." in X.extract_from_workflow("note it", agent_result)[0].content

    failed = SimpleNamespace(stop_reason="error", request_id="r4", tool_outcomes=[])
    assert X.extract_from_workflow(
        "do x", failed
    )  # errors are worth remembering even without steps


def test_summaries_and_explicit_preferences():
    summary = X.from_summary("We planned the migration.", "c1")
    assert (summary.type, summary.kind, summary.source, summary.conversation_id) == (
        MemoryType.LONG_TERM,
        MemoryKind.SUMMARY,
        MemorySource.SUMMARY,
        "c1",
    )
    assert X.from_summary("   ", "c1") is None

    prefs = X.from_preferences({"response_length": "short", "empty": ""})
    assert [p.content for p in prefs] == ["User's response length preference: short"]
    assert prefs[0].explicit and prefs[0].metadata == {"preference_key": "response_length"}


# ------------------------------------------------------------------ evaluation


def evaluator(**settings):
    s = MemorySettings(**settings)
    return CandidateEvaluator(s, SensitiveDataFilter(s.privacy))


def test_evaluator_scores_filters_and_orders():
    cands = [
        candidate("User prefers dark mode and tabs"),
        candidate("User saw it", kind=MemoryKind.EVENT),  # trivial -> below threshold
        candidate("my api key is abc123", kind=MemoryKind.FACT),  # sensitive
        candidate("User prefers dark mode and tabs"),  # in-batch duplicate
        candidate("User wants the assistant to always answer briefly", explicit=True),
    ]

    result = evaluator().evaluate(cands)

    assert [c.content for c in result.approved] == [
        "User wants the assistant to always answer briefly",
        "User prefers dark mode and tabs",
    ]
    assert all(c.importance is not None for c in result.approved)
    reasons = {r.reason.split(":")[0] for r in result.rejected}
    assert reasons == {"below_threshold", "sensitive", "duplicate_in_batch"}
    assert not any(
        "abc123" in r.preview for r in result.rejected
    )  # rejected secrets are never echoed


def test_evaluator_threshold_and_turn_limit_are_configurable():
    cands = [candidate(f"User prefers option number {i} in the app") for i in range(6)]

    assert len(evaluator(max_candidates_per_turn=2).evaluate(cands).approved) == 2
    assert len(evaluator(max_candidates_per_turn=2).evaluate(cands).rejected) == 4
    assert evaluator(min_importance_to_store=0.99).evaluate(cands).approved == []
    explicit = [candidate("User likes tea", explicit=True)]
    assert (
        evaluator(min_importance_to_store=0.99).evaluate(explicit).approved
    )  # explicit bypasses the threshold


def test_evaluator_redacts_pii_in_approved_candidates():
    result = evaluator().evaluate(
        [candidate("Reach the user at ada@example.com for invoices", kind=MemoryKind.FACT)]
    )

    assert (
        "[email]" in result.approved[0].content and "example.com" not in result.approved[0].content
    )


# ------------------------------------------------------------------ optional LLM extractor


async def test_llm_extractor_merges_model_output_with_rules():
    provider = ScriptedProvider(
        structured=[
            _ExtractedMemories(
                memories=[
                    _ExtractedItem(
                        content="User is learning Rust",
                        type=MemoryType.LONG_TERM,
                        kind=MemoryKind.FACT,
                        importance=0.7,
                    ),
                    _ExtractedItem(
                        content="Odd", type=MemoryType.SEMANTIC, kind=MemoryKind.PREFERENCE
                    ),  # kind/type mismatch
                ]
            )
        ]
    )

    found = await LLMMemoryExtractor(provider).extract_from_message(
        "I'm learning Rust. My name is Ada Lovelace", conversation_id="c1"
    )

    assert [c.content for c in found] == [
        "User is learning Rust",
        "Odd",
        "User's name is Ada Lovelace",
    ]
    assert (
        found[1].kind is MemoryKind.KNOWLEDGE
        and found[0].importance == 0.7
        and found[0].conversation_id == "c1"
    )
    assert "Never include secrets" in provider.structured_requests[0][0][0].content


async def test_llm_extractor_falls_back_to_rules_when_the_model_fails():
    provider = ScriptedProvider(structured=[LLMRateLimitError("slow", provider="p")])

    found = await LLMMemoryExtractor(provider).extract_from_message("I live in Pune")

    assert [c.content for c in found] == ["User lives in Pune"]
