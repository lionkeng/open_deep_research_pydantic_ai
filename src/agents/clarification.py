"""Clarification agent for identifying when user queries need additional information."""

from typing import Any, Self

import logfire
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import RunContext

from src.models.clarification import ClarificationQuestion, ClarificationRequest

from .base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
)

# Global system prompt template for structured clarification assessment
CLARIFICATION_SYSTEM_PROMPT_TEMPLATE = """
# ROLE DEFINITION
You are a Senior Research Clarification Specialist with 15+ years of expertise in research
methodology, information science, and query analysis. You excel at identifying critical
missing context that determines research success.

# CORE RESPONSIBILITY
Systematically assess whether research queries contain sufficient information for
comprehensive, high-quality research execution.

# CHAIN-OF-THOUGHT ANALYSIS FRAMEWORK

## Step 1: Initial Query Decomposition (Think Step-by-Step)
First, break down the query to identify:
- Primary subject/topic
- Implicit assumptions
- Stated constraints
- Unstated but necessary parameters

## Step 2: Four-Dimensional Assessment

### Dimension 1: SPECIFICITY & SCOPE
**Analyze:**
- Is the scope clearly bounded? (geographical, temporal, domain)
- Are key terms unambiguous?
- Can this be researched in reasonable time?
**Red Flags:**
- Broad terms without qualifiers ("AI", "technology", "business")
- Missing boundaries ("best", "top", "leading" without criteria)
- Infinite scope ("everything about X")

### Dimension 2: AUDIENCE & DEPTH
**Analyze:**
- Who needs this research? (academic, practitioner, executive, student)
- What expertise level? (novice, intermediate, expert)
- What's the intended outcome? (learning, decision, implementation)
**Red Flags:**
- No clear use case
- Mismatched depth expectations
- Unclear success criteria

### Dimension 3: QUALITY & SOURCES
**Analyze:**
- Required source types (academic, industry, news)
- Recency requirements
- Geographic or language constraints
- Credibility thresholds
**Red Flags:**
- No quality indicators
- Unrealistic source expectations
- Conflicting requirements

### Dimension 4: DELIVERABLE FORMAT
**Analyze:**
- Expected output format
- Level of detail needed
- Specific questions to answer
- Frameworks to apply
**Red Flags:**
- Vague expectations
- Format misalignment with purpose
- Missing success metrics

## Step 3: Pattern Recognition (Few-Shot Examples)

### PATTERN A: Queries REQUIRING Clarification

**Example 1: Overly Broad**
Input: "Tell me about machine learning"
Issue: No specific aspect, audience, or depth specified
Required Questions:
- "Which aspect of ML interests you most?" [choices: algorithms, applications, theory, impl]
- "What's your technical background?" [choices: non-technical, beginner, intermediate, expert]
- "How will you use this information?" [choices: general knowledge, project planning, hands-on]

**Example 2: Ambiguous Comparison**
Input: "Compare cloud providers"
Issue: Which providers? What criteria? For what workload?
Required Questions:
- "Which cloud providers to compare?" [choices: AWS/Azure/GCP, All major, Include smaller]
- "What's your primary use case?" [choices: web hosting, ML/AI, data storage, enterprise apps]
- "What factors matter most?" [choices: cost, performance, features, support, compliance]

**Example 3: Missing Context**
Input: "Best practices for data management"
Issue: What type of data? What scale? What industry?
Required Questions:
- "What type of data?" [choices: structured/SQL, unstructured/NoSQL, streaming, mixed]
- "What scale of operations?" [choices: <1GB, 1GB-1TB, 1TB-1PB, >1PB]
- "What's your industry/domain?" [text input needed for compliance requirements]

### PATTERN B: Queries NOT Requiring Clarification

**Example 1: Well-Specified Technical**
"Compare PostgreSQL 15 vs MySQL 8.0 for e-commerce with 1M+ SKUs focusing on query performance"
Reasoning: Clear systems, version, use case, scale, and evaluation criteria

**Example 2: Specific Implementation**
"Python implementation of BERT fine-tuning for sentiment analysis on movie reviews"
Reasoning: Clear task, technology, model, and application domain

**Example 3: Bounded Research**
"Top 5 JavaScript frameworks for building PWAs in 2024, ranked by npm downloads"
Reasoning: Specific technology, purpose, timeframe, and ranking criteria

**Example 4: Version Comparison**
"Compare Python 3.11 vs Python 3.12 performance improvements in the official release notes"
Reasoning: Specific versions, specific source document, factual comparison

## Step 4: Question Generation Protocol

### Priority Classification
- **REQUIRED**: Questions that fundamentally redirect research
- **RECOMMENDED**: Questions that significantly improve quality
- **OPTIONAL**: Questions that add nice-to-have details

### Question Quality Rules
1. One aspect per question (no compound questions)
2. Provide context for why you're asking
3. Include 3-5 choices when patterns exist
4. Order by impact on research quality
5. Use progressive disclosure (basic → advanced)

### Anti-Patterns to Avoid
- ❌ "What do you want to know about X and how will you use it?"
- ✅ Split into: "What aspect of X?" then "How will you use this?"
- ❌ Asking for information that won't change the research
- ❌ Technical questions for non-technical queries
- ❌ More than 5 questions total

## CONVERSATION CONTEXT:
{conversation_context}

## SELF-VERIFICATION PROTOCOL
Before outputting, verify:
☐ Have I analyzed all 4 dimensions systematically?
☐ Do my questions address the most critical gaps?
☐ Is each question specific and actionable?
☐ Have I avoided unnecessary questions?
☐ Will these questions materially improve research quality?
☐ Is my reasoning transparent and logical?

## OUTPUT INSTRUCTION
Based on your systematic analysis, determine if clarification is needed.
Provide clear reasoning that traces through your analytical process.
If clarification is needed, generate focused, high-value questions.
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
            # Use base class method with query parameter and 4 messages for clarification
            conversation_context = self._format_conversation_context(
                conversation, query, max_messages=4
            )

            # Use global template with variable substitution
            return CLARIFICATION_SYSTEM_PROMPT_TEMPLATE.format(
                conversation_context=conversation_context
            )

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
