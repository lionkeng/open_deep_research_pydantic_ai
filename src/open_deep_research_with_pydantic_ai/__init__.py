"""Deep Research with Pydantic AI - AI-powered research system."""

from open_deep_research_with_pydantic_ai.core.workflow import workflow
from open_deep_research_with_pydantic_ai.models.research import (
    ResearchBrief,
    ResearchFinding,
    ResearchReport,
    ResearchStage,
    ResearchState,
)

__version__ = "1.0.0"
__all__ = [
    "workflow",
    "ResearchState",
    "ResearchStage",
    "ResearchBrief",
    "ResearchFinding",
    "ResearchReport",
]
