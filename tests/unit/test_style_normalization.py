from __future__ import annotations

from agents.report_clean_merge import apply_style_normalization
from models.report_generator import ReportSection, ResearchReport


def test_style_normalization_titles_and_prefixes() -> None:
    report = ResearchReport(
        title="T",
        executive_summary="Unexpected discovery: Exec insight [S1].",
        introduction="Implication: Intro text [S2].",
        sections=[
            ReportSection(
                title="Finding 3 — Modularization gains",
                content="Evidence: Offsite methods reduce cycle time [S3].",
                subsections=[
                    ReportSection(
                        title="Pattern Analysis and Contradictions",
                        content="Decision takeaway: Resolve apparent conflict [S4].",
                    )
                ],
            )
        ],
        conclusions="Observation: Conclude [S5].",
        recommendations=["Finding: Shift mix [S6]."],
        references=[],
        appendices={"annex": "Evidence: Annex detail [S7]."},
        quality_score=0.5,
    )

    out = apply_style_normalization(report)

    # Labels stripped from fields
    assert out.executive_summary.startswith("Exec insight")
    assert out.introduction.startswith("Intro text")
    assert out.sections[0].content.startswith("Offsite methods")
    assert out.sections[0].subsections[0].content.startswith("Resolve apparent")
    assert out.conclusions.startswith("Conclude")
    assert out.recommendations[0].startswith("Shift mix")
    assert out.appendices["annex"].startswith("Annex detail")

    # Titles normalized / synthesized
    # First section title should not retain the label and should be a natural headline
    assert not out.sections[0].title.lower().startswith("finding")
    assert out.sections[0].title and out.sections[0].title[0].isupper()
    assert out.sections[0].subsections[0].title == "Resolve Apparent Conflict"
