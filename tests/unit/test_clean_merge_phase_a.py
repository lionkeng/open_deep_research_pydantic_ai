"""Phase A clean-merge utility tests.

These tests avoid any network/LLM calls and validate local utilities
and templates introduced in Phase 4 foundations.
"""

from __future__ import annotations

from agents.report_clean_merge import (
    CLEAN_MERGE_INSTRUCTIONS_TEMPLATE,
    length_ok,
    marker_counts,
)


def test_marker_counts_detects_duplicates() -> None:
    a = "Finding supported by [S1] and [S2] and again [S1]."
    b = "Same markers but different counts [S1] and [S2]."
    counts_a = marker_counts(a)
    counts_b = marker_counts(b)
    assert counts_a != counts_b
    assert counts_a.get("S1") == 2
    assert counts_b.get("S1") == 1


def test_length_ok_bounds() -> None:
    before = "x" * 100
    assert length_ok(before, "x" * 100) is True
    # 16% shorter should fail under 15% tolerance
    assert length_ok(before, "x" * 84) is False
    # 10% longer should pass
    assert length_ok(before, "x" * 110) is True


def test_clean_merge_instructions_constant_present() -> None:

    filled = CLEAN_MERGE_INSTRUCTIONS_TEMPLATE.format(
        field_name="executive_summary",
        thesis="AI will shift work",
        tone="polished",
        terminology="AI, automation",
        outline="Intro, Findings, Conclusion",
        prev_snippet="Prev text",
        next_snippet="Next text",
        transition_cues="Therefore; However",
    )
    assert "You are a senior editor" in filled
    assert "Field: executive_summary" in filled
    assert "Hard Constraints" in filled
    assert "Output only valid JSON" in filled
