"""Deep Research with Pydantic AI - AI-powered research system."""

from pathlib import Path

from dotenv import load_dotenv

from .core.workflow import workflow
from .models.brief_generator import ResearchBrief
from .models.core import ResearchStage, ResearchState
from .models.report_generator import ResearchReport
from .models.research_executor import ResearchFinding

# Load .env file on module import to ensure API keys are available
# This happens before agents are lazily initialized
for path in [Path.cwd() / ".env"] + [p / ".env" for p in Path.cwd().parents]:
    if path.exists():
        _ = load_dotenv(path, override=True)
        break

__version__ = "1.0.0"
__all__ = [
    "workflow",
    "ResearchState",
    "ResearchStage",
    "ResearchBrief",
    "ResearchFinding",
    "ResearchReport",
]
