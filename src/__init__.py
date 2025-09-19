"""Deep Research with Pydantic AI - AI-powered research system."""

# Note: Environment variables are automatically loaded by src/core/__init__.py
# when importing from core.workflow

from .core.workflow import ResearchWorkflow
from .models.core import ResearchStage, ResearchState
from .models.report_generator import ResearchReport

__version__ = "1.0.0"
__all__ = [
    "ResearchWorkflow",
    "ResearchState",
    "ResearchStage",
    "ResearchReport",
]
