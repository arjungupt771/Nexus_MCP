"""Sensitive-information protection for everything that enters memory.

Three layers:

1. **Blocked** content (secrets, credentials, payment cards, government IDs,
   sensitive-topic terms) is never stored as a memory.
2. **PII** (emails, phone numbers) is redacted by default.
3. **Restriction phrases** ("don't remember this", "off the record") make the
   extractor skip that message entirely.

Conversation messages are persisted with secrets redacted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class PiiMode(StrEnum):
    REDACT = "redact"
    ALLOW = "allow"
    BLOCK = "block"


class PrivacyPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pii_mode: PiiMode = PiiMode.REDACT
    # Sensitive personal topics are not auto-remembered unless explicitly allowed.
    allow_sensitive_topics: bool = False
    sensitive_topic_terms: tuple[str, ...] = (
        "diagnos",
        "medication",
        "prescription",
        "therapy",
        "therapist",
        "depress",
        "anxiety",
        "pregnan",
        "religio",
        "political party",
        "sexual orientation",
        "immigration status",
        "visa status",
        "criminal record",
        "salary",
        "net worth",
        "bank account",
    )
    restricted_terms: tuple[str, ...] = ()
    restriction_phrases: tuple[str, ...] = (
        "don't remember",
        "do not remember",
        "don't save",
        "do not save",
        "off the record",
        "forget this",
        "keep this private",
        "between us",
        "not to be stored",
        "don't store",
    )
    redact_messages: bool = True


_SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "api_key": re.compile(
        r"\b(?:ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{10,}|\bsk-[A-Za-z0-9_-]{10,}"
        r"|\bxox[abprs]-[A-Za-z0-9-]{8,}|\bAIza[0-9A-Za-z_-]{20,}|\bAKIA[0-9A-Z]{16}\b"
        r"|\bya29\.[0-9A-Za-z_-]{10,}"
    ),
    "token": re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*|\bBearer\s+[A-Za-z0-9._~+/-]{12,}"
    ),
    "credential": re.compile(
        r"\b(?:password|passwd|pwd|passphrase|secret|api[ _-]?key"
        r"|access[ _-]?token|auth[ _-]?token)"
        r"\s*(?:is|=|:)\s*\S+",
        re.IGNORECASE,
    ),
    "government_id": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}
_CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)\s?|\d{3}[ .-])\d{3}[ .-]\d{4}(?!\w)"
    r"|(?<![\w+])\+\d{1,3}[ .-]?\d{2,4}[ .-]?\d{3,4}[ .-]?\d{3,4}(?!\w)"
)


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]

    if not 13 <= len(digits) <= 19:
        return False

    total = 0

    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit *= 2
            digit -= 9 if digit > 9 else 0

        total += digit

    return total % 10 == 0


@dataclass(frozen=True)
class SensitivityResult:
    blocked: bool
    categories: tuple[str, ...] = ()
    text: str = ""  # input with secrets/PII redacted according to policy
    reasons: tuple[str, ...] = field(default=())

    @property
    def reason(self) -> str | None:
        return f"sensitive:{','.join(self.categories)}" if self.blocked else None


class SensitiveDataFilter:
    def __init__(self, policy: PrivacyPolicy | None = None) -> None:
        self.policy = policy or PrivacyPolicy()

    def is_restricted_request(self, text: str) -> bool:
        """True if the user asked for this not to be remembered."""
        lowered = text.lower()
        return any(phrase in lowered for phrase in self.policy.restriction_phrases)

    def redact_secrets(self, text: str) -> str:
        """Replace secrets/cards/IDs with markers (used for stored messages)."""
        redacted = text

        for category, pattern in _SECRET_PATTERNS.items():
            redacted = pattern.sub(f"[REDACTED:{category}]", redacted)

        return _CARD.sub(
            lambda m: "[REDACTED:payment_card]" if _luhn(m.group()) else m.group(), redacted
        )

    def inspect(self, text: str) -> SensitivityResult:
        categories: list[str] = []

        for category, pattern in _SECRET_PATTERNS.items():
            if pattern.search(text):
                categories.append(category)

        if any(_luhn(m.group()) for m in _CARD.finditer(text)):
            categories.append("payment_card")

        lowered = text.lower()

        if not self.policy.allow_sensitive_topics and any(
            term in lowered for term in self.policy.sensitive_topic_terms
        ):
            categories.append("sensitive_topic")

        if any(term.lower() in lowered for term in self.policy.restricted_terms):
            categories.append("restricted_term")

        # Secrets first so card numbers are never mistaken for phone numbers.
        cleaned = self.redact_secrets(text)
        mode = self.policy.pii_mode
        has_pii = bool(_EMAIL.search(cleaned) or _PHONE.search(cleaned))

        if has_pii and mode is PiiMode.BLOCK:
            categories.append("pii")
        elif has_pii and mode is PiiMode.REDACT:
            cleaned = _PHONE.sub("[phone]", _EMAIL.sub("[email]", cleaned))

        return SensitivityResult(
            blocked=bool(categories),
            categories=tuple(dict.fromkeys(categories)),
            text=cleaned,
        )
