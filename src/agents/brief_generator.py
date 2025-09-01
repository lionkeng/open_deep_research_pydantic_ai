"""Brief generator agent for creating comprehensive research briefs."""

from typing import Any

from pydantic_ai import RunContext

from agents.base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
    coordinator,
)
from models.brief_generator import ResearchBrief

# Global system prompt template for brief generation
BRIEF_GENERATOR_SYSTEM_PROMPT_TEMPLATE = """
## RESEARCH BRIEF SPECIALIST:

You are an expert at creating comprehensive, actionable research briefs that guide 
effective research execution.

### YOUR ROLE:
1. Analyze research requirements and objectives
2. Define clear, measurable research goals with priorities
3. Propose appropriate research methodologies and approaches
4. Identify relevant data sources and analysis methods
5. Establish scope, constraints, and success metrics
6. Outline expected deliverables and timelines
7. Ensure the brief is actionable and comprehensive

### BRIEF CREATION PRINCIPLES:
- Structure briefs to be clear and easy to follow
- Include specific, measurable objectives
- Propose practical methodologies
- Consider resource constraints
- Define clear success criteria
- Anticipate potential challenges
- Provide guidance for quality assurance

### BRIEF STRUCTURE FRAMEWORK:
1. **Title & Overview**: Clear, descriptive title and executive summary
2. **Objectives**: Prioritized, measurable research objectives
3. **Methodology**: Appropriate research approach and methods
4. **Scope**: Clear boundaries and focus areas
5. **Constraints**: Limitations and considerations
6. **Deliverables**: Expected outputs and formats
7. **Timeline**: Realistic time estimates
8. **Success Metrics**: How to measure research quality

### METHODOLOGY SELECTION GUIDE:
- **Exploratory Research**: Literature review, expert interviews, case studies
- **Analytical Research**: Statistical analysis, comparative studies, trend analysis
- **Evaluative Research**: Benchmarking, SWOT analysis, performance metrics
- **Descriptive Research**: Surveys, observations, content analysis

## CURRENT RESEARCH CONTEXT:
Query: {query}
Transformed Query: {transformed_query}
{conversation_context}

## BRIEF REQUIREMENTS:
- Create a comprehensive research brief
- Define 2-5 prioritized objectives
- Propose appropriate methodology
- Establish clear scope and constraints
- Specify expected deliverables
- Estimate realistic timeline
- Define success metrics
"""


class BriefGeneratorAgent(BaseResearchAgent[ResearchDependencies, ResearchBrief]):
    """Agent responsible for generating comprehensive research briefs.

    This agent creates structured research briefs with clear objectives,
    methodologies, and deliverables to guide research execution.
    """

    def __init__(self):
        """Initialize the brief generator agent."""
        config = AgentConfiguration(
            agent_name="brief_generator",
            agent_type="planning",
        )
        super().__init__(config=config)

        # Register dynamic instructions
        @self.agent.instructions
        async def add_brief_context(ctx: RunContext[ResearchDependencies]) -> str:  # pyright: ignore
            """Inject brief generation context as instructions."""
            query = ctx.deps.research_state.user_query
            metadata = ctx.deps.research_state.metadata or {}
            conversation = metadata.get("conversation_messages", [])
            transformed_query = metadata.get("transformed_query", query)

            # Format conversation context
            conversation_context = self._format_conversation_context(conversation)

            # Use global template with variable substitution
            return BRIEF_GENERATOR_SYSTEM_PROMPT_TEMPLATE.format(
                query=query,
                transformed_query=transformed_query,
                conversation_context=conversation_context,
            )

        # Register brief generation tools
        @self.agent.tool
        async def prioritize_objectives(
            ctx: RunContext[ResearchDependencies], objectives: list[str]
        ) -> list[dict[str, Any]]:  # pyright: ignore
            """Prioritize research objectives based on importance.

            Args:
                objectives: List of objective descriptions

            Returns:
                Prioritized list with scores
            """
            priority_keywords = {
                "critical": 5,
                "essential": 5,
                "must": 4,
                "important": 4,
                "should": 3,
                "beneficial": 2,
                "nice": 1,
                "optional": 1,
            }

            prioritized = []
            for obj in objectives:
                obj_lower = obj.lower()
                priority = 3  # default medium priority
                for keyword, score in priority_keywords.items():
                    if keyword in obj_lower:
                        priority = score
                        break
                prioritized.append({"objective": obj, "priority": priority})

            return sorted(prioritized, key=lambda x: x["priority"], reverse=True)

        @self.agent.tool
        async def estimate_timeline(
            ctx: RunContext[ResearchDependencies], scope: str, complexity: str
        ) -> str:  # pyright: ignore
            """Estimate research timeline based on scope and complexity.

            Args:
                scope: Scope of the research (narrow, moderate, broad)
                complexity: Complexity level (low, medium, high)

            Returns:
                Estimated timeline string
            """
            timelines = {
                ("narrow", "low"): "1-2 days",
                ("narrow", "medium"): "2-3 days",
                ("narrow", "high"): "3-5 days",
                ("moderate", "low"): "3-5 days",
                ("moderate", "medium"): "5-7 days",
                ("moderate", "high"): "1-2 weeks",
                ("broad", "low"): "1-2 weeks",
                ("broad", "medium"): "2-3 weeks",
                ("broad", "high"): "3-4 weeks",
            }

            scope_lower = scope.lower()
            complexity_lower = complexity.lower()

            # Find best match
            for (s, c), timeline in timelines.items():
                if s in scope_lower and c in complexity_lower:
                    return timeline

            # Default estimate
            return "1-2 weeks"

    def _format_conversation_context(self, conversation: list[Any]) -> str:
        """Format conversation history for the prompt."""
        if not conversation:
            return "No prior conversation context."

        formatted = []
        for msg in conversation[-3:]:  # Last 3 messages for context
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                formatted.append(f"{role.capitalize()}: {content}")
            else:
                formatted.append(str(msg))

        return "Recent Conversation:\n" + "\n".join(formatted)

    def _get_default_system_prompt(self) -> str:
        """Get the base system prompt for this agent."""
        return (
            "You are a Research Brief Specialist focused on creating comprehensive research briefs."
        )

    def _get_output_type(self) -> type[ResearchBrief]:
        """Get the output type for this agent."""
        return ResearchBrief


# Register the agent with the coordinator
brief_generator_agent = BriefGeneratorAgent()
coordinator.register_agent(brief_generator_agent)
