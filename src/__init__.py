"""Deep Research with Pydantic AI - AI-powered research system."""

from core.workflow import workflow
from models.research import (
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
