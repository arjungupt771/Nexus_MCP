"""Small deterministic text helpers shared by extraction and retrieval."""

from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and the to of in on for with from by at as is are was were be been it this that these "
    "those my me i we you your our they them he she his her its not no do does did can could "
    "would should will just so if then than too very about into over out up down please "
    "user users use need want tell know show give find".split()
)


def stem(token: str) -> str:
    """Very light plural/-ing stripping; deterministic and dependency-free."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"

    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]

    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token


def tokenize(text: str) -> list[str]:
    """Lowercased, stopword-free, stemmed tokens (order preserved, duplicates kept)."""
    tokens = []

    for raw in _WORD.findall(text.lower()):
        if raw in STOPWORDS or len(raw) < 2:
            continue

        stemmed = stem(raw)

        if stemmed not in STOPWORDS:  # "uses" -> "use" is still a stopword
            tokens.append(stemmed)

    return tokens


def keyword_set(text: str, limit: int = 24) -> list[str]:
    """Unique tokens in first-seen order, capped."""
    return list(dict.fromkeys(tokenize(text)))[:limit]


def normalize_for_hash(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode()).hexdigest()[:32]


def estimate_tokens(text: str) -> int:
    """Cheap, provider-independent estimate (~4 characters per token)."""
    return max(1, (len(text) + 3) // 4)
