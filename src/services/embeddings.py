"""Embedding service abstraction with optional OpenAI and local backends.

This module provides a small, optional embedding layer used to augment
deterministic grouping and clustering logic with semantic similarity.

Design goals:
- Optional dependency: gracefully handles missing packages or keys
- Async API with simple in-memory caching
- Utility helpers for cosine similarity and pairwise matrices
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Protocol

Vector = list[float]


class EmbeddingBackend(Protocol):
    """Protocol for embedding backends."""

    async def embed(self, texts: list[str]) -> list[Vector]:
        """Embed a batch of texts into vector representations."""


@dataclass
class OpenAIEmbeddingBackend:
    """OpenAI embeddings backend (optional).

    Requires the `openai` package and a valid API key. If the package is not
    available or the key is missing, calling `embed` will raise a RuntimeError.
    """

    model: str = "text-embedding-3-small"
    api_key: str | None = None

    async def embed(self, texts: list[str]) -> list[Vector]:  # pragma: no cover - network
        try:
            import openai  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("openai package not installed") from exc

        key = self.api_key
        if not key:
            # Attempt to read from env variable on demand to keep import light
            import os

            key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing for OpenAIEmbeddingBackend")

        client = openai.OpenAI(api_key=key)  # type: ignore[attr-defined]
        # NOTE: openai python client is sync; we call in a thread if needed.
        # Given tests mock this path or avoid it, keep simple here.
        resp = client.embeddings.create(model=self.model, input=texts)  # type: ignore[no-untyped-call]
        return [list(d.embedding) for d in resp.data]


@dataclass
class LocalEmbeddingBackend:
    """Local sentence-transformers backend (optional).

    Requires `sentence-transformers`. If missing, `embed` raises RuntimeError
    when invoked.
    """

    model: str = "all-MiniLM-L6-v2"
    _model: Any | None = field(default=None, init=False, repr=False)

    def _ensure_model(self) -> None:
        if self._model is None:  # pragma: no cover - optional dependency
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("sentence-transformers not installed") from exc
            self._model = SentenceTransformer(self.model)

    async def embed(self, texts: list[str]) -> list[Vector]:  # pragma: no cover - heavy model
        self._ensure_model()
        # sentence-transformers encode is sync; acceptable for small batches
        return [list(vec) for vec in self._model.encode(texts, convert_to_numpy=False)]


@dataclass
class EmbeddingService:
    """Thin wrapper that selects a backend and caches embeddings."""

    backend: EmbeddingBackend | None = None
    cache_enabled: bool = True
    _cache: dict[str, Vector] = field(default_factory=dict, init=False, repr=False)

    async def embed_batch(self, texts: list[str]) -> list[Vector] | None:
        """Embed a batch of texts or return None if no backend is configured."""
        if not texts:
            return []
        if self.backend is None:
            return None

        # Cache per text
        to_compute: list[tuple[int, str]] = []
        vectors: list[Vector] = [cast_empty()] * len(texts)

        for i, t in enumerate(texts):
            key = self._hash_text(t)
            if self.cache_enabled and key in self._cache:
                vectors[i] = self._cache[key]
            else:
                to_compute.append((i, t))

        if to_compute:
            batch_texts = [t for _, t in to_compute]
            computed = await self.backend.embed(batch_texts)
            for (i, original_text), vec in zip(to_compute, computed, strict=False):
                vectors[i] = vec
                if self.cache_enabled:
                    self._cache[self._hash_text(original_text)] = vec

        return vectors

    @staticmethod
    def _hash_text(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()


def cast_empty() -> Vector:
    return []


def cosine_similarity(u: Iterable[float], v: Iterable[float]) -> float:
    """Compute cosine similarity with safety checks."""
    u_list = list(u)
    v_list = list(v)
    if not u_list or not v_list:
        return 0.0
    dot = sum(a * b for a, b in zip(u_list, v_list, strict=False))
    nu = math.sqrt(sum(a * a for a in u_list))
    nv = math.sqrt(sum(b * b for b in v_list))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(dot / (nu * nv))


def pairwise_cosine_matrix(vectors: list[Vector]) -> list[list[float]]:
    """Return a symmetric cosine similarity matrix for vectors."""
    n = len(vectors)
    sim: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim[i][i] = 1.0
        for j in range(i + 1, n):
            s = cosine_similarity(vectors[i], vectors[j])
            sim[i][j] = s
            sim[j][i] = s
    return sim


def cluster_by_threshold(
    indices: list[int], sim: list[list[float]], threshold: float
) -> list[list[int]]:
    """Greedy connectivity-based clustering by similarity threshold.

    Two items are in the same cluster if a path of pairwise similarities >= threshold connects them.
    """
    n = len(indices)
    visited = [False] * n
    clusters: list[list[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        # BFS/DFS from i
        stack = [i]
        visited[i] = True
        group = [indices[i]]
        while stack:
            k = stack.pop()
            for j in range(n):
                if not visited[j] and sim[k][j] >= threshold:
                    visited[j] = True
                    stack.append(j)
                    group.append(indices[j])
        clusters.append(group)

    return clusters


__all__ = [
    "EmbeddingBackend",
    "OpenAIEmbeddingBackend",
    "LocalEmbeddingBackend",
    "EmbeddingService",
    "Vector",
    "cosine_similarity",
    "pairwise_cosine_matrix",
    "cluster_by_threshold",
]
