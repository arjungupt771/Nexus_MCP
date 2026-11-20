"""Embedding abstraction and a deterministic, dependency-free local embedder."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

from agent.memory.text import tokenize


@runtime_checkable
class Embedder(Protocol):
    """Turns text into fixed-size vectors. Swap in a real model if desired."""

    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Feature-hashing embedder: no model, no network, fully deterministic.

    It captures word, bigram and sub-word (character trigram) overlap, so it
    finds paraphrases that share stems/fragments ("deploying" ~ "deployment")
    but NOT true synonyms. It is the zero-dependency default that keeps the
    core test-suite and local use free of paid services; plug in a neural
    ``Embedder`` for deeper semantic recall.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 16:
            raise ValueError("dimensions must be at least 16")

        self.dimensions = dimensions

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dimensions, 1.0 if (value >> 63) & 1 else -1.0

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        features: list[tuple[str, float]] = [(f"w:{t}", 1.0) for t in tokens]
        features += [(f"b:{a}_{b}", 0.7) for a, b in zip(tokens, tokens[1:], strict=False)]

        for token in tokens:
            padded = f"#{token}#"
            features += [(f"c:{padded[i : i + 3]}", 0.3) for i in range(len(padded) - 2)]

        for feature, weight in features:
            index, sign = self._bucket(feature)
            vector[index] += sign * weight

        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors (0.0 for empty/zero vectors)."""
    if len(a) != len(b) or not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
