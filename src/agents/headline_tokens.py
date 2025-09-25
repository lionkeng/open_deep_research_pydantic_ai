"""Shared constants for normalizing research report headings."""

from __future__ import annotations

# Tokens that indicate a heading starts with generic scaffolding rather than
# substantive content. Used by both outline synthesis and retry normalization to
# decide when to replace or skip leading terms like "Finding" or "Source".
GENERIC_SECTION_STARTERS = {
    "finding",
    "findings",
    "insight",
    "insights",
    "observation",
    "observations",
    "source",
    "sources",
}


__all__ = ["GENERIC_SECTION_STARTERS"]
