"""Data models for the deep research system."""

# Core models
# Phase 2 model imports
from .brief_generator import ResearchBrief, ResearchMethodology, ResearchObjective
from .compression import CompressedContent, CompressedSection
from .core import (
    BriefGenerationResult,
    ClarificationResult,
    ResearchPriority,
    ResearchStage,
    ResearchState,
    TransformedQueryResult,
)
from .query_transformation import TransformedQuery
from .report_generator import ReportMetadata, ReportSection, ResearchReport
from .research_executor import ResearchFinding, ResearchResults, ResearchSource

__all__ = [
    # Core models
    "ResearchStage",
    "ResearchPriority",
    "ClarificationResult",
    "TransformedQueryResult",
    "BriefGenerationResult",
    "ResearchState",
    # Query transformation models
    "TransformedQuery",
    # Brief generator models
    "ResearchObjective",
    "ResearchMethodology",
    "ResearchBrief",
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
