"""Stage metadata and user-facing descriptions for the research workflow.

This module provides rich metadata about each research stage, including
user-friendly descriptions, agent information, and educational content.
"""

from dataclasses import dataclass
from typing import Dict, List

from ..models.core import ResearchStage


@dataclass(frozen=True)
class AgentInfo:
    """Information about an AI agent."""
    
    name: str
    icon: str
    purpose: str
    specialization: str


@dataclass(frozen=True)
class StageMetadata:
    """Comprehensive metadata for a research stage."""
    
    title: str
    description: str
    agent_info: AgentInfo
    activities: List[str]
    user_benefit: str
    typical_duration_seconds: tuple[int, int]  # (min, max)
    complexity_factors: List[str]
    success_indicators: List[str]


# Comprehensive stage metadata
STAGE_METADATA: Dict[ResearchStage, StageMetadata] = {
    ResearchStage.PENDING: StageMetadata(
        title="Initializing Research",
        description="Preparing your research environment and validating your query",
        agent_info=AgentInfo(
            name="System Coordinator",
            icon="⚡",
            purpose="Setting up your research session",
            specialization="System initialization and validation"
        ),
        activities=[
            "Validating research parameters",
            "Initializing AI agents",
            "Preparing research environment"
        ],
        user_benefit="Ensures all systems are ready for optimal research",
        typical_duration_seconds=(5, 15),
        complexity_factors=["Query length", "System load"],
        success_indicators=["Environment ready", "Agents initialized"]
    ),
    
    ResearchStage.CLARIFICATION: StageMetadata(
        title="Understanding Your Query",
        description="Our AI analyzes your question to ensure we research exactly what you need",
        agent_info=AgentInfo(
            name="Clarification Specialist",
            icon="🎯",
            purpose="Understanding your research needs",
            specialization="Query analysis and requirement clarification"
        ),
        activities=[
            "Analyzing query scope and intent",
            "Identifying potential ambiguities",
            "Preparing clarifying questions if needed",
            "Validating research objectives"
        ],
        user_benefit="Ensures focused, relevant research results tailored to your needs",
        typical_duration_seconds=(15, 45),
        complexity_factors=[
            "Query ambiguity level",
            "Technical complexity",
            "Scope breadth"
        ],
        success_indicators=[
            "Clear research objectives",
            "Defined scope boundaries",
            "Resolved ambiguities"
        ]
    ),
    
    ResearchStage.BRIEF_GENERATION: StageMetadata(
        title="Planning Your Research",
        description="Creating a comprehensive research strategy tailored to your specific needs",
        agent_info=AgentInfo(
            name="Research Planner",
            icon="📋",
            purpose="Designing your research strategy",
            specialization="Research methodology and planning"
        ),
        activities=[
            "Identifying key research areas",
            "Planning search methodology",
            "Setting success criteria",
            "Defining research deliverables",
            "Creating execution timeline"
        ],
        user_benefit="Guarantees systematic, thorough investigation with clear objectives",
        typical_duration_seconds=(20, 60),
        complexity_factors=[
            "Research scope complexity",
            "Methodology requirements",
            "Number of research angles"
        ],
        success_indicators=[
            "Comprehensive research plan",
            "Clear methodology defined",
            "Success metrics established"
        ]
    ),
    
    ResearchStage.RESEARCH_EXECUTION: StageMetadata(
        title="Conducting Deep Research",
        description="Our AI researchers gather and analyze information from multiple authoritative sources",
        agent_info=AgentInfo(
            name="Research Executor",
            icon="🔬",
            purpose="Gathering comprehensive information",
            specialization="Multi-source research and analysis"
        ),
        activities=[
            "Searching academic databases",
            "Analyzing web sources and articles",
            "Cross-referencing findings",
            "Evaluating source credibility",
            "Synthesizing information",
            "Fact-checking claims"
        ],
        user_benefit="Comprehensive, accurate, and up-to-date information from trusted sources",
        typical_duration_seconds=(90, 300),
        complexity_factors=[
            "Query technical depth",
            "Number of sources required",
            "Information availability",
            "Cross-validation needs"
        ],
        success_indicators=[
            "Multiple sources consulted",
            "High-quality findings identified",
            "Claims verified and validated"
        ]
    ),
    
    ResearchStage.COMPRESSION: StageMetadata(
        title="Analyzing and Organizing",
        description="Processing and organizing research findings into coherent themes and insights",
        agent_info=AgentInfo(
            name="Information Synthesizer",
            icon="🧠",
            purpose="Organizing research findings",
            specialization="Data compression and thematic analysis"
        ),
        activities=[
            "Analyzing research findings",
            "Identifying key themes",
            "Removing duplicate information",
            "Prioritizing important insights",
            "Creating content structure"
        ],
        user_benefit="Well-organized, focused information without redundancy",
        typical_duration_seconds=(30, 90),
        complexity_factors=[
            "Volume of research data",
            "Thematic complexity",
            "Information overlap"
        ],
        success_indicators=[
            "Clear thematic organization",
            "Reduced information redundancy",
            "Key insights prioritized"
        ]
    ),
    
    ResearchStage.REPORT_GENERATION: StageMetadata(
        title="Creating Your Report",
        description="Crafting a comprehensive, well-structured report from your research findings",
        agent_info=AgentInfo(
            name="Report Writer",
            icon="📝",
            purpose="Creating your final report",
            specialization="Technical writing and report structuring"
        ),
        activities=[
            "Structuring report sections",
            "Writing executive summary",
            "Developing detailed analysis",
            "Adding citations and references",
            "Formatting for clarity",
            "Quality assurance review"
        ],
        user_benefit="Professional, comprehensive report ready for immediate use",
        typical_duration_seconds=(45, 120),
        complexity_factors=[
            "Report length requirements",
            "Technical detail level",
            "Citation complexity"
        ],
        success_indicators=[
            "Well-structured document",
            "Clear, professional writing",
            "Properly cited sources"
        ]
    ),
    
    ResearchStage.COMPLETED: StageMetadata(
        title="Research Complete",
        description="Your research has been successfully completed and is ready for review",
        agent_info=AgentInfo(
            name="System Coordinator",
            icon="✅",
            purpose="Finalizing your research",
            specialization="Quality assurance and delivery"
        ),
        activities=[
            "Final quality checks",
            "Report validation",
            "Delivery preparation"
        ],
        user_benefit="High-quality research ready for your use",
        typical_duration_seconds=(5, 15),
        complexity_factors=["Report size", "Validation requirements"],
        success_indicators=["Research delivered", "Quality verified"]
    )
}


def get_stage_metadata(stage: ResearchStage) -> StageMetadata:
    """Get metadata for a specific research stage.
    
    Args:
        stage: The research stage to get metadata for
        
    Returns:
        StageMetadata for the specified stage
        
    Raises:
        KeyError: If the stage is not recognized
    """
    if stage not in STAGE_METADATA:
        raise KeyError(f"No metadata available for stage: {stage}")
    
    return STAGE_METADATA[stage]


def get_all_stage_names() -> List[str]:
    """Get user-friendly names for all research stages.
    
    Returns:
        List of stage titles in workflow order
    """
    workflow_order = [
        ResearchStage.PENDING,
        ResearchStage.CLARIFICATION,
        ResearchStage.BRIEF_GENERATION,
        ResearchStage.RESEARCH_EXECUTION,
        ResearchStage.COMPRESSION,
        ResearchStage.REPORT_GENERATION,
        ResearchStage.COMPLETED,
    ]
    
    return [STAGE_METADATA[stage].title for stage in workflow_order]


def get_stage_by_index(index: int) -> ResearchStage:
    """Get research stage by workflow index (0-based).
    
    Args:
        index: Index in the workflow (0-6)
        
    Returns:
        ResearchStage at the specified index
        
    Raises:
        IndexError: If index is out of range
    """
    workflow_order = [
        ResearchStage.PENDING,
        ResearchStage.CLARIFICATION,
        ResearchStage.BRIEF_GENERATION,
        ResearchStage.RESEARCH_EXECUTION,
        ResearchStage.COMPRESSION,
        ResearchStage.REPORT_GENERATION,
        ResearchStage.COMPLETED,
    ]
    
    if not 0 <= index < len(workflow_order):
        raise IndexError(f"Stage index {index} out of range (0-{len(workflow_order)-1})")
    
    return workflow_order[index]


def get_stage_index(stage: ResearchStage) -> int:
    """Get workflow index for a research stage.
    
    Args:
        stage: The research stage
        
    Returns:
        Index of the stage in the workflow (0-6)
    """
    workflow_order = [
        ResearchStage.PENDING,
        ResearchStage.CLARIFICATION,
        ResearchStage.BRIEF_GENERATION,
        ResearchStage.RESEARCH_EXECUTION,
        ResearchStage.COMPRESSION,
        ResearchStage.REPORT_GENERATION,
        ResearchStage.COMPLETED,
    ]
    
    try:
        return workflow_order.index(stage)
    except ValueError as e:
        raise ValueError(f"Unknown stage: {stage}") from e