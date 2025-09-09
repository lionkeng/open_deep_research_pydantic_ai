"""Data models for the deep research system."""

# Core models
# Phase 2 model imports
from .compression import CompressedContent, CompressedSection
from .core import (
    ClarificationResult,
    ResearchPriority,
    ResearchStage,
    ResearchState,
)
from .report_generator import ReportMetadata, ReportSection, ResearchReport
from .research_executor import ResearchFinding, ResearchResults, ResearchSource

__all__ = [
    # Core models
    "ResearchStage",
    "ResearchPriority",
    "ClarificationResult",
    "ResearchState",
    # Research executor models
    "ResearchSource",
    "ResearchFinding",
    "ResearchResults",
    # Compression models
    "CompressedSection",
    "CompressedContent",
    # Report generator models
    "ReportSection",
    "ReportMetadata",
    "ResearchReport",
]
