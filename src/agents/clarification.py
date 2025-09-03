"""Clarification agent for identifying when user queries need additional information."""

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from src.models.api_models import ConversationMessage

from .base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
)

# Global system prompt template for structured clarification assessment
CLARIFICATION_SYSTEM_PROMPT_TEMPLATE = """
## SYSTEMATIC CLARIFICATION ASSESSMENT FRAMEWORK:

Analyze the query using these four critical dimensions:

### 1. AUDIENCE LEVEL & PURPOSE:
  - Who is this research for? (academic, business, personal, student)
  - What background knowledge level? (beginner, expert, professional)
  - What's the intended use? (presentation, analysis, decision-making, learning)

### 2. SCOPE & FOCUS AREAS:
  - Is the topic too broad without specific focus?
  - Are there multiple aspects that need prioritization?
  - What depth of coverage is needed? (overview vs deep-dive)

### 3. SOURCE & QUALITY REQUIREMENTS:
  - Academic papers vs. general sources?
  - Specific time periods or geographic regions?
  - Industry vs. theoretical focus?
  - Credibility and recency requirements?

### 4. DELIVERABLE SPECIFICATIONS:
  - What format of information is most useful?
  - Are there specific questions to be answered?
  - Any particular frameworks or methodologies to apply?
  - Length and detail requirements?

## PATTERN RECOGNITION EXAMPLES:

**QUERIES TYPICALLY REQUIRING CLARIFICATION:**
  • "What is [broad topic]?" → Ask about audience, purpose, specific focus
  • "Research [general field]" → Ask about scope, depth, intended use
  • "Compare [general category]" → Ask which items, criteria, timeframe
  • "Analyze [industry/trend]" → Ask about aspects, timeframe, geography
  • "How does [broad concept] work?" → Ask about technical level, applications

**QUERIES TYPICALLY NOT REQUIRING CLARIFICATION:**
  • "Compare ResNet-50 vs VGG-16 for ImageNet classification" → Specific, technical
  • "Implement Python function for Fibonacci using dynamic programming" → Clear task
  • "Current stock price of Apple Inc. (AAPL)" → Specific, factual, time-bound
  • "Step-by-step tutorial for Docker deployment on AWS" → Clear deliverable

## PROFESSIONAL QUESTION FORMATION:
If clarification is needed, craft questions that:
  • Use bullet points or numbered lists for clarity
  • Are concise while gathering all necessary information
  • Don't ask for information already provided
  • Focus on the missing critical dimensions identified above
  • Help narrow scope from broad to actionable

## CONVERSATION CONTEXT:
{conversation_context}

## ASSESSMENT INSTRUCTION:
Based on your systematic analysis above, determine if the query provides sufficient
information across all four critical dimensions for comprehensive research. Consider
the pattern examples and conversation history.
"""


class ClarifyWithUser(BaseModel):
    """Model for user clarification requests with structured assessment framework."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question."
    )
    question: str = Field(
        default="", description="A question to ask the user to clarify the report scope"
    )
    verification: str = Field(
        default="", description="Verification message that we will start research"
    )
    # Structured qualitative assessment fields
    missing_dimensions: list[str] = Field(
        default_factory=list,
        description="List of missing context dimensions from 4-category framework",
    )
    assessment_reasoning: str = Field(
        default="", description="Detailed reasoning behind the clarification decision"
    )
    suggested_clarifications: list[str] = Field(
        default_factory=list,
        description="Suggested clarification topics to address missing dimensions",
    )


class ClarificationAgent(BaseResearchAgent[ResearchDependencies, ClarifyWithUser]):
    """Agent responsible for determining if clarification is needed from the user.

    Uses structured LLM approach with 4-category assessment framework:
    - Audience Level & Purpose analysis
    - Scope & Focus Areas evaluation
    - Source & Quality Requirements assessment
    - Deliverable Specifications identification
    """

    def __init__(self):
        """Initialize the clarification agent."""
        config = AgentConfiguration(
            agent_name="clarification_agent",
            agent_type="clarification",
        )
        super().__init__(config=config)

        # Register dynamic instructions for assessment framework
        @self.agent.instructions
        async def add_assessment_framework(ctx: RunContext[ResearchDependencies]) -> str:  # pyright: ignore
            """Inject structured clarification assessment framework as instructions."""
            query = ctx.deps.research_state.user_query
            metadata = ctx.deps.research_state.metadata
            conversation = metadata.conversation_messages if metadata else []

            # Format conversation context for template substitution
            conversation_context = self._format_conversation_context(conversation, query)

            # Use global template with variable substitution
            return CLARIFICATION_SYSTEM_PROMPT_TEMPLATE.format(
                conversation_context=conversation_context
            )

    def _format_conversation_context(
        self,
        conversation: list[ConversationMessage],
        query: str,
    ) -> str:
        """Format conversation history for the prompt."""
        if not conversation:
            return f"Initial Query: {query}\n(No prior conversation)"

        # Format conversation messages
        formatted: list[str] = []
        for msg in conversation:
            # With TypedDict, we know these fields exist
            role = msg["role"]
            content = msg["content"]
            formatted.append(f"{role.capitalize()}: {content}")

        context = "\n".join(formatted[-4:])  # Last 4 messages for context
        return f"Recent Conversation:\n{context}\nCurrent Query: {query}"

    def _get_output_type(self) -> type[ClarifyWithUser]:
        """Get the output type for this agent."""
        return ClarifyWithUser

    def _get_default_system_prompt(self) -> str:
        """Get the basic system prompt which defines Agent role (required by base class)."""
        return """You are a research clarification specialist who determines whether user
queries require additional information before comprehensive research can begin. You analyze
queries across multiple dimensions to assess if they provide sufficient context for
high-quality research."""

    def _register_tools(self) -> None:
        """Register clarification-specific tools.

        For this agent, we don't need complex tools - the LLM will analyze
        the conversation history directly to determine if clarification is needed.
        """
        # No tools needed for simple clarification assessment
        pass


# Create module-level instance
clarification_agent = ClarificationAgent()
