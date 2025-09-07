"""Clarification agent for identifying when user queries need additional information."""

from typing import Any, Self

import logfire
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import RunContext

from models.api_models import ConversationMessage
from models.clarification import ClarificationQuestion, ClarificationRequest

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
When generating clarification questions:
  • Create separate questions for different aspects (don't combine unrelated topics)
  • Mark questions as required if they're critical for accurate research
  • Mark questions as optional if they would enhance but aren't essential
  • Use choice questions when there are clear alternatives
  • Order questions by importance (most critical first)
  • Provide context for complex or technical questions
  • Keep each question focused on one specific aspect

## CONVERSATION CONTEXT:
{conversation_context}

## ASSESSMENT INSTRUCTION:
Based on your systematic analysis above, determine if the query provides sufficient
information across all four critical dimensions for comprehensive research. If not,
generate structured clarification questions as separate ClarificationQuestion objects.
"""


class ClarifyWithUser(BaseModel):
    """Agent output for clarification needs with multi-question support."""

    needs_clarification: bool = Field(description="Whether clarification is needed from the user")
    request: ClarificationRequest | None = Field(
        default=None, description="Structured clarification request with one or more questions"
    )
    reasoning: str = Field(description="Explanation of why clarification is or isn't needed")

    # Structured qualitative assessment fields
    missing_dimensions: list[str] = Field(
        default_factory=list,
        description="List of missing context dimensions from 4-category framework",
    )
    assessment_reasoning: str = Field(
        default="", description="Detailed reasoning behind the clarification decision"
    )

    @model_validator(mode="after")
    def validate_request_consistency(self) -> Self:
        """Ensure request presence matches needs_clarification."""
        if self.needs_clarification and not self.request:
            # If clarification is needed but no request provided, create a minimal one
            # This shouldn't happen if the agent works correctly, but provides fallback
            self.request = ClarificationRequest(
                questions=[
                    ClarificationQuestion(
                        question="Could you provide more details about your research needs?",
                        is_required=True,
                    )
                ]
            )
        if not self.needs_clarification and self.request:
            # If no clarification needed, clear any request
            self.request = None
        return self


class ClarificationAgent(BaseResearchAgent[ResearchDependencies, ClarifyWithUser]):
    """Agent responsible for determining if clarification is needed from the user.

    Uses structured LLM approach with 4-category assessment framework:
    - Audience Level & Purpose analysis
    - Scope & Focus Areas evaluation
    - Source & Quality Requirements assessment
    - Deliverable Specifications identification

    Now supports multiple questions with UUID-based tracking.
    """

    def __init__(
        self,
        config: AgentConfiguration | None = None,
        dependencies: ResearchDependencies | None = None,
    ):
        """Initialize the clarification agent.

        Args:
            config: Optional agent configuration. If not provided, uses defaults.
            dependencies: Optional research dependencies.
        """
        if config is None:
            config = AgentConfiguration(
                agent_name="clarification_agent",
                agent_type="clarification",
            )
        super().__init__(config=config, dependencies=dependencies)

        # Register dynamic instructions for assessment framework
        @self.agent.instructions
        async def add_assessment_framework(ctx: RunContext[ResearchDependencies]) -> str:
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
high-quality research.

When clarification is needed, you generate structured questions as separate ClarificationQuestion
objects within a ClarificationRequest. Each question should:
- Have a unique purpose and focus on one aspect
- Be marked as required or optional appropriately
- Have an order value indicating priority (0 = highest)
- Include question_type: "text", "choice", or "multi_choice"
- Provide choices array for choice questions
- Include context field when additional explanation helps
"""

    def _register_tools(self) -> None:
        """Register clarification-specific tools.

        For this agent, we don't need complex tools - the LLM will analyze
        the conversation history directly to determine if clarification is needed.
        """
        # No tools needed for simple clarification assessment
        pass


# Lazy initialization of module-level instance
_clarification_agent_instance = None


def get_clarification_agent() -> ClarificationAgent:
    """Get or create the clarification agent instance."""
    global _clarification_agent_instance
    if _clarification_agent_instance is None:
        _clarification_agent_instance = ClarificationAgent()
        logfire.info("Initialized clarification_agent agent")
    return _clarification_agent_instance


# For backward compatibility, create a property-like access
class _LazyAgent:
    """Lazy proxy for ClarificationAgent that delays instantiation."""

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the actual agent instance."""
        return getattr(get_clarification_agent(), name)


clarification_agent = _LazyAgent()
