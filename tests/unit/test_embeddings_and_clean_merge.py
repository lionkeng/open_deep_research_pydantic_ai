"""Tests for embedding-assisted grouping and citation guardrails.

These tests avoid network calls by using a fake embedding backend and by
exercising local guardrail utilities without invoking an LLM.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from services.synthesis_tools import SynthesisTools


class _FakeEmbeddingService:
    """Deterministic fake embedding service for tests."""

    def __init__(self) -> None:
        # Map substrings to simple 2D vectors for cosine clustering
        self._map: list[tuple[str, list[float]]] = [
            ("ai adoption", [1.0, 0.0]),
            ("artificial intelligence adoption", [0.99, 0.01]),
            ("ai implementation", [0.98, 0.02]),
            ("unrelated", [0.0, 1.0]),
        ]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        out: list[list[float]] = []
        for t in texts:
            tl = t.lower()
            vec = [0.0, 1.0]
            for key, v in self._map:
                if key in tl:
                    vec = v
                    break
            out.append(list(vec))
        return out


def test_group_similar_claims_with_embeddings() -> None:
    svc = _FakeEmbeddingService()
    tools = SynthesisTools(embedding_service=svc, enable_embedding_similarity=True, similarity_threshold=0.9)

    claims = [
        ("AI adoption is increasing across industries", "s1"),
        ("Artificial intelligence adoption is growing in many sectors", "s2"),
        ("AI implementation is accelerating across different industries", "s3"),
        ("This unrelated statement should not cluster", "s4"),
    ]

    groups = tools._group_similar_claims(claims)
    # Expect at least one group with the first three claims
    assert any(len(g) >= 2 and all(c[1] in {"s1", "s2", "s3"} for c in g) for g in groups)


def test_marker_guardrail_equality() -> None:
    from agents.report_generator import ReportGeneratorAgent

    a = "Insight supported by [S1] and [S2]."
    b = "Refined insight still citing [S1] and [S2]."
    c = "Oops dropped [S2]."

    assert ReportGeneratorAgent._markers_equal(a, b) is True
    assert ReportGeneratorAgent._markers_equal(a, c) is False
