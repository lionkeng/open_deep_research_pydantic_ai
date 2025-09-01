"""Deep Research with Pydantic AI - AI-powered research system."""

from .core.workflow import workflow
from .models.brief_generator import ResearchBrief
from .models.core import ResearchStage, ResearchState
from .models.report_generator import ResearchReport
from .models.research_executor import ResearchFinding

__version__ = "1.0.0"
__all__ = [
    "workflow",
    "ResearchState",
    "ResearchStage",
    "ResearchBrief",
    "ResearchFinding",
    "ResearchReport",
]
