"""Unit tests for deterministic outline generation in the research executor."""

from __future__ import annotations

from agents.research_outline import build_section_outline, synthesize_headline
from models.research_executor import HierarchicalFinding, ImportanceLevel, ThemeCluster


def test_synthesize_headline_removes_citations_and_limits_words() -> None:
    """Headlines should strip citations and respect word caps."""

    text = "Key finding: Solar adoption accelerates in 2024 [S1]. Additional context follows."

    headline = synthesize_headline(text, max_words=6)

    assert "[S1]" not in headline
    # Should capture salient terms and capitalize appropriately
    assert headline == "Solar Adoption Accelerates 2024"


def test_synthesize_headline_drops_generic_prefixes_and_duplicates() -> None:
    """Noise like "finding source" should be stripped from synthesized titles."""

    text = (
        "Finding source biggest builders which commonly have in-house builder scale, "
        "cost structure, and market behavior."
    )

    headline = synthesize_headline(text, max_words=10)

    assert headline.startswith("Biggest Builders"), headline
    assert "Finding" not in headline
    assert "Source" not in headline


def test_build_section_outline_trims_bullets_and_collects_sources() -> None:
    """Outline bullets should be concise and evidence IDs deduplicated."""

    primary_finding = HierarchicalFinding(
        finding="Solar adoption surged in 2024 [S1].",
        supporting_evidence=["Utility-scale installations doubled year-over-year. [S3]"],
        importance=ImportanceLevel.CRITICAL,
        importance_score=0.92,
        confidence_score=0.81,
        source_ids=["src-1"],
        supporting_source_ids=["src-3"],
    )
    secondary_finding = HierarchicalFinding(
        finding="Battery storage lowers costs; [S2]",
        importance=ImportanceLevel.HIGH,
        importance_score=0.88,
        confidence_score=0.6,
        source_ids=["src-2"],
    )

    cluster = ThemeCluster(
        theme_name="Renewable adoption momentum",
        description="Evidence of accelerating clean energy deployments",
        importance_score=0.9,
        coherence_score=0.8,
        findings=[primary_finding, secondary_finding],
    )

    outline = build_section_outline([cluster], max_sections=3, max_bullets=3)

    assert outline, "Expected outline entries for populated cluster"
    entry = outline[0]

    assert entry.title.startswith("Solar Adoption Surged"), entry.title
    assert entry.bullets == [
        "Solar adoption surged in 2024",
        "Battery storage lowers costs",
    ]
    assert entry.salient_evidence_ids == ["src-1", "src-3", "src-2"]
