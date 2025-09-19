"""Unit tests for the synthesis engine clustering heuristics."""

from __future__ import annotations

import math

from models.research_executor import HierarchicalFinding, ImportanceLevel
from services.synthesis_engine import SynthesisEngine


def _make_finding(text: str, *, category: str, importance: ImportanceLevel) -> HierarchicalFinding:
    """Helper to build findings with consistent metadata."""

    return HierarchicalFinding(
        finding=text,
        supporting_evidence=[f"Evidence for {text}"],
        category=category,
        importance=importance,
    )


def test_cluster_findings_small_corpus_uses_loose_thresholds() -> None:
    """Small corpora should keep all terms and still cluster cleanly."""

    engine = SynthesisEngine(min_cluster_size=2)

    # Vectorizer configuration adapts to the small document count
    vectorizer = engine._create_vectorizer(4)
    assert vectorizer.min_df == 1
    assert math.isclose(vectorizer.max_df, 1.0)

    findings = [
        _make_finding("AI adoption drives revenue growth", category="business", importance=ImportanceLevel.HIGH),
        _make_finding("Revenue climbs with AI automation", category="business", importance=ImportanceLevel.HIGH),
        _make_finding("Teams report efficiency gains from automation", category="operations", importance=ImportanceLevel.MEDIUM),
        _make_finding("Automation improves operational efficiency metrics", category="operations", importance=ImportanceLevel.MEDIUM),
    ]

    clusters = engine.cluster_findings(findings)

    assert clusters  # Ensure the workflow succeeded
    for cluster in clusters:
        assert 0.0 <= cluster.coherence_score <= 1.0


def test_cluster_findings_large_corpus_scales_thresholds_and_reduces() -> None:
    """Larger corpora should tighten thresholds and still cluster without error."""

    engine = SynthesisEngine(min_cluster_size=2)

    vectorizer = engine._create_vectorizer(60)
    assert vectorizer.min_df == 2  # switches to absolute cutoff
    assert math.isclose(vectorizer.max_df, 0.85)

    findings: list[HierarchicalFinding] = []
    for topic_idx in range(3):
        base_topic = f"Topic {topic_idx}"
        for doc_idx in range(20):
            findings.append(
                _make_finding(
                    text=(
                        f"{base_topic} explores pattern {doc_idx} with focus on "
                        f"innovation strategy {topic_idx} and adoption outcomes"
                    ),
                    category=f"category-{topic_idx}",
                    importance=ImportanceLevel.MEDIUM,
                )
            )

    clusters = engine.cluster_findings(findings)

    assert clusters  # Workflow should succeed on the larger corpus as well
    for cluster in clusters:
        assert 0.0 <= cluster.coherence_score <= 1.0
