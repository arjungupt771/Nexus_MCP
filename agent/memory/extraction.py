"""Memory extraction, classification, importance scoring and evaluation (8.3).

Deterministic rules do the work by default, so extraction is testable and
needs no paid API. The pipeline deliberately stores *little*:

* only the user's own statements are mined (never assistant text),
* questions, hypotheticals and chit-chat are ignored,
* tool results become memories only when they record an action (not raw data),
* every candidate must clear an importance threshold and the privacy filter.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.llm.errors import LLMError
from agent.llm.models import Message, Role
from agent.llm.provider import LLMProvider
from agent.memory.models import (
    DEFAULT_KIND_FOR_TYPE,
    KINDS_BY_TYPE,
    MemoryKind,
    MemorySource,
    MemoryType,
)
from agent.memory.privacy import SensitiveDataFilter
from agent.memory.settings import MemorySettings
from agent.memory.text import content_hash

logger = logging.getLogger(__name__)

_VERBS = {
    "prefer": "prefers",
    "like": "likes",
    "love": "loves",
    "hate": "hates",
    "dislike": "dislikes",
    "enjoy": "enjoys",
    "use": "uses",
    "work": "works",
    "live": "lives",
    "have": "has",
    "need": "needs",
    "want": "wants",
    "am": "is",
    "study": "studies",
    "own": "owns",
    "drive": "drives",
    "speak": "speaks",
    "run": "runs",
    "play": "plays",
    "write": "writes",
}
RELATION_NOUNS = frozenset(
    "manager boss wife husband partner brother sister friend colleague coworker teammate mentor "
    "mother father mom dad son daughter assistant client roommate neighbor".split()
)
MUTATING_VERBS = frozenset(
    "create update delete remove add archive restore schedule send post merge close reopen cancel "
    "move rename write upload insert book set".split()
)
_SKIP_START = re.compile(
    r"^(?:if|what|how|why|when|where|who|can you|could you|would you|do you|is it)\b", re.I
)
_PRONOUN_ONLY = {"it", "that", "this", "them", "those", "these", "one", "him", "her"}
_BASE_IMPORTANCE = {
    MemoryKind.PREFERENCE: 0.70,
    MemoryKind.FACT: 0.60,
    MemoryKind.RELATIONSHIP: 0.65,
    MemoryKind.DECISION: 0.70,
    MemoryKind.OUTCOME: 0.50,
    MemoryKind.TASK: 0.50,
    MemoryKind.SUMMARY: 0.50,
    MemoryKind.KNOWLEDGE: 0.55,
    MemoryKind.EVENT: 0.45,
    MemoryKind.CONTEXT: 0.45,
}


@dataclass
class MemoryCandidate:
    content: str
    type: MemoryType
    kind: MemoryKind
    source: MemorySource
    explicit: bool = False
    importance: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    conversation_id: str | None = None
    ttl_days: float | None = None


def classify(content: str) -> tuple[MemoryType, MemoryKind]:
    """Infer (type, kind) from free text."""
    text = content.lower()

    if re.search(
        r"\b(prefer|prefers|likes?|loves?|hates?|dislikes?|favou?rite|always|never"
        r"|wants? the assistant)\b",
        text,
    ):
        return MemoryType.LONG_TERM, MemoryKind.PREFERENCE

    if re.search(r"\b(decided|decision|let's go with|we'll use|chose|chosen)\b", text):
        return MemoryType.EPISODIC, MemoryKind.DECISION

    if re.search(
        r"\b(created|scheduled|completed|finished|deleted|updated|sent|failed|succeeded|ran)\b",
        text,
    ):
        return MemoryType.EPISODIC, MemoryKind.OUTCOME

    if re.search(
        r"\b(is|are) (?:the user's|my|our) [\w ]+$"
        r"|\b(reports to|works with|manages|is married to)\b",
        text,
    ):
        return MemoryType.SEMANTIC, MemoryKind.RELATIONSHIP

    if re.search(r"\b(user|i|my|me)\b", text):
        return MemoryType.LONG_TERM, MemoryKind.FACT

    return MemoryType.SEMANTIC, MemoryKind.KNOWLEDGE


def resolve_type_kind(
    content: str, memory_type: MemoryType | None, kind: MemoryKind | None
) -> tuple[MemoryType, MemoryKind]:
    if memory_type is None and kind is None:
        return classify(content)

    if memory_type is None:
        assert kind is not None
        return next(t for t, kinds in KINDS_BY_TYPE.items() if kind in kinds), kind

    if kind is None:
        inferred_type, inferred_kind = classify(content)
        return (
            memory_type,
            inferred_kind if inferred_type == memory_type else DEFAULT_KIND_FOR_TYPE[memory_type],
        )

    return memory_type, kind


class ImportanceScorer:
    """Transparent importance in [0, 1]: kind baseline plus small, documented adjustments."""

    def score(self, candidate: MemoryCandidate) -> float:
        value = _BASE_IMPORTANCE[candidate.kind]
        words = len(candidate.content.split())

        if candidate.explicit:
            value += 0.25  # the user asked us to remember it

        if words < 4:
            value -= 0.15
        elif words >= 7:
            value += 0.03

        if re.search(r"\d|[A-Z][a-z]+ [A-Z][a-z]+", candidate.content):
            value += 0.03  # concrete detail (numbers, names)

        if candidate.metadata.get("status") in {"failed", "partial", "error"}:
            value += 0.10  # failures are worth remembering

        if candidate.metadata.get("directive") == "true":
            value += 0.10

        if len((candidate.metadata.get("servers") or "").split(",")) > 1:
            value += 0.05  # multi-server workflows

        return round(min(1.0, max(0.0, value)), 3)


def to_third_person(text: str) -> str:
    """'I prefer tabs' -> 'User prefers tabs' (light, deterministic)."""
    out = re.sub(r"\bI(?:'m| am)\b", "User is", text)
    out = re.sub(r"\bI(?:'ve| have)\b", "User has", out)
    out = re.sub(r"\bI\b", "User", out)
    out = re.sub(r"\bmy\b", "User's", out, flags=re.I)
    out = re.sub(r"\bme\b", "the user", out)
    out = re.sub(r"\bmine\b", "the user's", out)
    return re.sub(r"\bUser (\w+)\b", lambda m: f"User {_third_person_verb(m.group(1))}", out)


_AUXILIARY = frozenset(
    "is has was will can should would must may might could shall had were".split()
)


def _third_person_verb(verb: str) -> str:
    """Conjugate the base verb that followed 'I' ('deploy' -> 'deploys')."""
    lowered = verb.lower()
    irregular = {"do": "does", "go": "goes", "be": "is", "have": "has"}

    if lowered in _VERBS:
        return _VERBS[lowered]

    if lowered in _AUXILIARY:
        return verb

    if lowered in irregular:
        return irregular[lowered]

    if lowered.endswith(("ss", "x", "z", "ch", "sh")):
        return verb + "es"

    if lowered.endswith("s"):  # already third person ("needs")
        return verb

    if re.search(r"[^aeiou]y$", lowered):
        return verb[:-1] + "ies"

    return verb + "s"


_CLAUSE_SPLIT = re.compile(
    r"(?<=[.!?])\s+|\n+|;\s*|,?\s+and\s+(?=(?:my|i|i'm|i've|please)\b)", re.IGNORECASE
)


def _sentences(text: str) -> list[str]:
    """Split into sentences and into clauses that restart with a first-person pattern."""
    return [p.strip() for p in _CLAUSE_SPLIT.split(text) if p and p.strip()]


def _clean(sentence: str) -> str:
    return sentence.strip().rstrip(".!").strip()


_R = re.IGNORECASE
_EXPLICIT = re.compile(
    r"^(?:please\s+)?(?:remember|keep in mind|don't forget|do not forget)(?:\s+that|\s*:)?\s+(.+)$",
    _R,
)
_PREFERENCE = re.compile(
    r"^i (?:really |absolutely )?(prefer|like|love|enjoy|hate|dislike) (.+)$", _R
)
_FAVORITE = re.compile(r"^my favou?rite ([\w ]+?) is (.+)$", _R)
_DIRECTIVE = re.compile(r"^(?:please\s+)?(always|never) (.+)$|^from now on,? (.+)$", _R)
_NAME = re.compile(r"^my name is (.+)$", _R)
_CALL_ME = re.compile(r"^(?:please\s+)?call me (.+)$", _R)
_WORK = re.compile(r"^i work (at|for|as|with) (.+)$", _R)
_ROLE = re.compile(r"^i(?:'m| am) (?:a|an) (.+)$", _R)
_LIVE = re.compile(r"^i live in (.+)$|^i(?:'m| am) (?:from|based in|located in) (.+)$", _R)
_REL_A = re.compile(r"^([A-Z][\w'. -]{0,40}?) is my ([\w ]{2,30})$")
_REL_B = re.compile(r"^my ([\w ]{2,25}?) is ([A-Z][\w'. -]{0,40})$", _R)
_MY_X = re.compile(r"^my ([\w ]{2,25}?) (?:is|are) (.+)$", _R)
_DECISION = re.compile(
    r"^(?:let's|lets) ((?:go with|use|pick|choose) .+)$|^i(?:'ve| have)? decided (?:to|on) (.+)$"
    r"|^we(?:'ll| will) ((?:use|go with) .+)$",
    _R,
)


class RuleBasedExtractor:
    """Mines the *user's* messages, summaries, tool outcomes and workflow results."""

    def extract_from_message(
        self, text: str, *, conversation_id: str | None = None
    ) -> list[MemoryCandidate]:
        found: list[MemoryCandidate] = []

        for sentence in _sentences(text):
            cleaned = _clean(sentence)
            words = cleaned.split()

            if sentence.endswith("?") or len(words) < 3 or len(cleaned) > 300:
                continue

            candidate = self._match(cleaned)

            if candidate is not None:
                candidate.conversation_id = conversation_id
                found.append(candidate)

        return list({c.content.lower(): c for c in found}.values())

    def _match(self, s: str) -> MemoryCandidate | None:
        if m := _EXPLICIT.match(s):
            inner = _clean(m.group(1))
            content = to_third_person(inner)
            memory_type, kind = classify(content)
            return MemoryCandidate(content, memory_type, kind, MemorySource.EXPLICIT, explicit=True)

        if _SKIP_START.match(s):
            return None

        mk = MemoryCandidate
        src = MemorySource.USER_MESSAGE

        if m := _PREFERENCE.match(s):
            obj = m.group(2).strip()

            if obj.lower() in _PRONOUN_ONLY:
                return None

            verb = _VERBS.get(m.group(1).lower(), m.group(1))
            return mk(f"User {verb} {obj}", MemoryType.LONG_TERM, MemoryKind.PREFERENCE, src)

        if m := _FAVORITE.match(s):
            return mk(
                f"User's favorite {m.group(1)} is {m.group(2)}",
                MemoryType.LONG_TERM,
                MemoryKind.PREFERENCE,
                src,
            )

        if m := _DIRECTIVE.match(s):
            rest = m.group(2) or m.group(3)
            lead = f"{m.group(1).lower()} " if m.group(1) else ""
            return mk(
                f"User wants the assistant to {lead}{rest}".replace("  ", " "),
                MemoryType.LONG_TERM,
                MemoryKind.PREFERENCE,
                src,
                metadata={"directive": "true"},
            )

        if m := _NAME.match(s):
            return mk(f"User's name is {m.group(1)}", MemoryType.LONG_TERM, MemoryKind.FACT, src)

        if m := _CALL_ME.match(s):
            return mk(
                f"User prefers to be called {m.group(1)}",
                MemoryType.LONG_TERM,
                MemoryKind.PREFERENCE,
                src,
            )

        if m := _WORK.match(s):
            link = {"at": "works at", "for": "works for", "as": "works as", "with": "works with"}[
                m.group(1).lower()
            ]
            return mk(f"User {link} {m.group(2)}", MemoryType.LONG_TERM, MemoryKind.FACT, src)

        if m := _ROLE.match(s):
            if m.group(1).lower().startswith(("bit ", "little ", "big fan", "huge fan")):
                return None

            return mk(f"User is a {m.group(1)}", MemoryType.LONG_TERM, MemoryKind.FACT, src)

        if m := _LIVE.match(s):
            where = m.group(1) or m.group(2)
            verb = "lives in" if m.group(1) else "is based in"
            return mk(f"User {verb} {where}", MemoryType.LONG_TERM, MemoryKind.FACT, src)

        if (m := _REL_A.match(s)) and m.group(2).strip().lower() in RELATION_NOUNS:
            return self._relationship(m.group(1).strip(), m.group(2).strip())

        if (m := _REL_B.match(s)) and m.group(1).strip().lower() in RELATION_NOUNS:
            return self._relationship(m.group(2).strip(), m.group(1).strip())

        if m := _MY_X.match(s):
            noun = m.group(1).strip()

            if noun.lower().startswith("favo") or noun.lower() in RELATION_NOUNS:
                return None

            return mk(
                f"User's {noun} is {m.group(2).strip()}", MemoryType.LONG_TERM, MemoryKind.FACT, src
            )

        if m := _DECISION.match(s):
            what = next(g for g in m.groups() if g)
            return mk(f"Decision: {what.strip()}", MemoryType.EPISODIC, MemoryKind.DECISION, src)

        return None

    @staticmethod
    def _relationship(person: str, relation: str) -> MemoryCandidate:
        return MemoryCandidate(
            f"{person} is the user's {relation.lower()}",
            MemoryType.SEMANTIC,
            MemoryKind.RELATIONSHIP,
            MemorySource.USER_MESSAGE,
            metadata={
                "subject": person,
                "predicate": f"is_user's_{relation.lower().replace(' ', '_')}",
                "object": "user",
            },
        )

    # ---------------------------------------------------------- other sources

    def extract_from_tool_outcomes(self, outcomes: list[Any]) -> list[MemoryCandidate]:
        """Record *actions* (writes) and their failures; never raw read results."""
        found: list[MemoryCandidate] = []

        for outcome in outcomes:
            tool = getattr(outcome, "tool_name", None) or str(getattr(outcome, "spec_name", ""))
            verb = tool.split("_")[0].lower()

            if verb not in MUTATING_VERBS:
                continue

            server = getattr(outcome, "server_id", None) or "unknown"
            arguments = getattr(outcome, "arguments", None) or {}
            label = next(
                (
                    str(arguments[k])[:80]
                    for k in ("title", "summary", "name", "subject")
                    if k in arguments
                ),
                "",
            )
            suffix = f" ('{label}')" if label else ""
            meta = {"server_id": server, "tool_name": tool}

            if getattr(outcome, "success", False):
                content = f"Ran {server}.{tool}{suffix} successfully."
                meta["status"] = "succeeded"
            else:
                code = getattr(outcome, "error_code", None) or "error"
                content = f"Attempt to run {server}.{tool}{suffix} failed ({code})."
                meta.update(status="failed", error_code=str(code))

            found.append(
                MemoryCandidate(
                    content,
                    MemoryType.EPISODIC,
                    MemoryKind.OUTCOME,
                    MemorySource.TOOL_RESULT,
                    metadata=meta,
                )
            )

        return found

    def extract_from_workflow(self, request: str, result: Any) -> list[MemoryCandidate]:
        """One episodic record for a finished workflow that actually used tools."""
        stop = str(getattr(result, "stop_reason", "") or "")
        aggregated = getattr(result, "aggregated", None)
        records = getattr(result, "records", None) or {}
        outcomes = getattr(result, "tool_outcomes", None) or []
        servers: set[str] = set()
        steps = failed = 0

        if aggregated is not None:
            servers = set(aggregated.servers_used)
            steps = len(records)
            failed = len(aggregated.failed)
        elif outcomes:
            servers = {o.server_id for o in outcomes if getattr(o, "server_id", None)}
            steps = len(outcomes)
            failed = sum(1 for o in outcomes if not o.success)

        if steps == 0 and stop != "error":
            return []  # plain conversation: nothing worth an episode

        servers_text = ", ".join(sorted(servers)) or "no tools"
        content = (
            f"Task '{request.strip()[:160]}' finished as {stop or 'unknown'}; "
            f"used {servers_text} ({steps} step(s))"
        )
        content += f", {failed} failed." if failed else "."
        meta = {
            "request_id": str(getattr(result, "request_id", "")),
            "status": stop,
            "servers": ",".join(sorted(servers)),
            "steps": str(steps),
        }

        if getattr(result, "replans", 0):
            meta["replans"] = str(result.replans)

        return [
            MemoryCandidate(
                content, MemoryType.EPISODIC, MemoryKind.TASK, MemorySource.WORKFLOW, metadata=meta
            )
        ]

    def from_summary(self, summary: str, conversation_id: str | None) -> MemoryCandidate | None:
        summary = summary.strip()

        if not summary:
            return None

        return MemoryCandidate(
            summary[:1500],
            MemoryType.LONG_TERM,
            MemoryKind.SUMMARY,
            MemorySource.SUMMARY,
            conversation_id=conversation_id,
        )

    def from_preferences(self, preferences: dict[str, str]) -> list[MemoryCandidate]:
        """Explicitly supplied preferences (e.g. from a settings screen)."""
        return [
            MemoryCandidate(
                f"User's {key.replace('_', ' ')} preference: {value}",
                MemoryType.LONG_TERM,
                MemoryKind.PREFERENCE,
                MemorySource.EXPLICIT,
                explicit=True,
                metadata={"preference_key": key},
            )
            for key, value in preferences.items()
            if value
        ]


# ------------------------------------------------------------------ optional LLM extractor


class _ExtractedItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str = Field(min_length=3, max_length=400)
    type: MemoryType
    kind: MemoryKind
    importance: float = Field(default=0.6, ge=0, le=1)


class _ExtractedMemories(BaseModel):
    model_config = ConfigDict(extra="ignore")

    memories: list[_ExtractedItem] = Field(default_factory=list)


LLM_EXTRACT_PROMPT = (
    "Extract durable memories about the USER from their message: stable preferences, personal "
    "facts they stated, relationships, decisions. Ignore questions, one-off requests, small talk "
    "and anything about "
    "the assistant. Never include secrets, credentials, payment data or health/financial details. "
    "Write each memory as one short third-person sentence. "
    "Return an empty list if nothing is worth keeping."
)


class LLMMemoryExtractor:
    """Optional model-assisted extraction; falls back to the rules on any failure."""

    def __init__(self, provider: LLMProvider, fallback: RuleBasedExtractor | None = None) -> None:
        self._provider = provider
        self._rules = fallback or RuleBasedExtractor()

    # Everything except message mining is deterministic, so it is shared with the rules.
    def extract_from_tool_outcomes(self, outcomes: list[Any]) -> list[MemoryCandidate]:
        return self._rules.extract_from_tool_outcomes(outcomes)

    def extract_from_workflow(self, request: str, result: Any) -> list[MemoryCandidate]:
        return self._rules.extract_from_workflow(request, result)

    def from_summary(self, summary: str, conversation_id: str | None) -> MemoryCandidate | None:
        return self._rules.from_summary(summary, conversation_id)

    def from_preferences(self, preferences: dict[str, str]) -> list[MemoryCandidate]:
        return self._rules.from_preferences(preferences)

    async def extract_from_message(
        self, text: str, *, conversation_id: str | None = None
    ) -> list[MemoryCandidate]:
        try:
            result = await self._provider.generate_structured(
                [
                    Message(role=Role.SYSTEM, content=LLM_EXTRACT_PROMPT),
                    Message(role=Role.USER, content=text[:4000]),
                ],
                _ExtractedMemories,
            )
        except LLMError:
            logger.warning("LLM memory extraction failed; using rules")
            return self._rules.extract_from_message(text, conversation_id=conversation_id)

        found = [
            MemoryCandidate(
                i.content,
                i.type,
                i.kind if i.kind in KINDS_BY_TYPE[i.type] else DEFAULT_KIND_FOR_TYPE[i.type],
                MemorySource.USER_MESSAGE,
                importance=i.importance,
                conversation_id=conversation_id,
            )
            for i in result.memories
        ]
        # Rules still run: they catch explicit "remember ..." requests deterministically.
        seen = {c.content.lower() for c in found}
        found += [
            c
            for c in self._rules.extract_from_message(text, conversation_id=conversation_id)
            if c.content.lower() not in seen
        ]
        return found


# ------------------------------------------------------------------ evaluation


@dataclass
class Rejected:
    preview: str
    reason: str


@dataclass
class Evaluation:
    approved: list[MemoryCandidate] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)


class CandidateEvaluator:
    """Importance scoring + privacy + threshold + in-batch dedupe (graph node 6)."""

    def __init__(
        self,
        settings: MemorySettings,
        privacy: SensitiveDataFilter,
        scorer: ImportanceScorer | None = None,
    ):
        self._settings = settings
        self._privacy = privacy
        self._scorer = scorer or ImportanceScorer()

    def evaluate(self, candidates: list[MemoryCandidate]) -> Evaluation:
        result = Evaluation()
        seen: set[str] = set()

        for candidate in candidates:
            preview = candidate.content[:60]
            check = self._privacy.inspect(candidate.content)

            if check.blocked:
                result.rejected.append(
                    Rejected(
                        preview if not check.categories else "[withheld]",
                        check.reason or "sensitive",
                    )
                )
                continue

            candidate.content = check.text[: self._settings.max_content_chars]

            if candidate.importance is None:
                candidate.importance = self._scorer.score(candidate)

            key = content_hash(candidate.content) + candidate.type.value

            if key in seen:
                result.rejected.append(Rejected(preview, "duplicate_in_batch"))
                continue

            if (
                not candidate.explicit
                and candidate.importance < self._settings.min_importance_to_store
            ):
                result.rejected.append(Rejected(preview, f"below_threshold:{candidate.importance}"))
                continue

            seen.add(key)
            result.approved.append(candidate)

        result.approved.sort(key=lambda c: (-(c.importance or 0), c.content))
        overflow = result.approved[self._settings.max_candidates_per_turn :]
        result.approved = result.approved[: self._settings.max_candidates_per_turn]
        result.rejected += [Rejected(c.content[:60], "turn_limit") for c in overflow]
        return result


def describe(candidates: list[MemoryCandidate]) -> str:  # pragma: no cover - debugging aid
    return json.dumps([c.content for c in candidates])
