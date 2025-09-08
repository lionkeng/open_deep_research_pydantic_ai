"""Query transformation agent for optimizing research queries."""

from typing import Any

import logfire
from pydantic_ai import RunContext

from src.models.metadata import ResearchMetadata
from src.models.query_transformation import TransformedQuery

from .base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
)

# Global system prompt template for query transformation
QUERY_TRANSFORMATION_SYSTEM_PROMPT_TEMPLATE = """
## QUERY TRANSFORMATION SPECIALIST

Transform research queries using clarification insights to create specific,
actionable research questions.

## AVAILABLE CONTEXT:
Original Query: {original_query}
{conversation_context}

## CLARIFICATION ANALYSIS:
{clarification_assessment}

## IDENTIFIED GAPS:
{missing_dimensions}

## CLARIFICATION QUESTIONS:
{clarification_questions}

## USER RESPONSES:
{clarification_answers}

## TRANSFORMATION STRATEGY:

### 1. Use Clarification Insights
- Address each identified ambiguity from the assessment
- Incorporate user responses where available
- Make explicit assumptions for unanswered questions

### 2. Handle Partial Information
- If question answered → Use the response directly
- If question pending → Make reasonable assumption and document it
- If dimension missing → Define reasonable scope and note it

### 3. Transformation Rules
- Every ambiguity must be addressed (resolved or assumed)
- Assumptions must be explicit and reasonable
- Maintain original intent while adding specificity
- Break complex queries into 3-5 supporting questions

## OUTPUT REQUIREMENTS:

### Core Transformation
- **transformed_query**: Clear, specific version addressing all ambiguities
- **supporting_questions**: 3-5 sub-questions that decompose the main query
- **search_keywords**: 3-10 key terms for research
- **excluded_terms**: Terms to avoid irrelevant results

### Scope Definition
- **research_scope**: Clear boundaries of the research
- **temporal_scope**: Time period (if relevant)
- **geographic_scope**: Location boundaries (if relevant)
- **domain_scope**: Specific field/industry (if relevant)

### Clarification Tracking
- **assumptions_made**: List each assumption for unresolved ambiguities
- **ambiguities_resolved**: List what was clarified
- **ambiguities_remaining**: List what still needs clarification
- **clarification_coverage**: % of questions answered (0.0-1.0)

### Quality Metrics
- **specificity_score**: How specific is the result (0=vague, 1=very specific)
- **confidence_score**: Confidence in transformation quality
- **transformation_strategy**: Approach used (decomposition/specification/scoping/assumption-based)
- **transformation_rationale**: Explain the transformation approach

## EXAMPLES WITH CLARIFICATION:

### Example 1: Partial Clarification
Original: "How does AI work?"
Assessment: "Too broad - needs domain, application, technical depth"
Missing: ["Specific AI type", "Application domain", "Technical level"]
Questions Asked: ["Which AI field?", "What application?", "Technical depth?"]
User Response: Only answered "Healthcare" for application

Transformation:
- transformed_query: "How do ML algorithms work in healthcare diagnostics,
  focusing on neural networks at an intermediate technical level?"
- assumptions_made: ["AI type: Machine learning/neural networks", "Technical level: Intermediate"]
- ambiguities_resolved: ["Application domain: Healthcare"]
- ambiguities_remaining: []
- supporting_questions: [
  "What are the key ML algorithms used in healthcare?",
  "How do neural networks process medical data?",
  "What are the accuracy rates and limitations?"
]

### Example 2: No Clarification Responses
Original: "Best practices for data"
Assessment: "Vague - needs data type, context, purpose"
Missing: ["Data type", "Industry/domain", "Specific operation"]
Questions: All pending

Transformation:
- transformed_query: "What are best practices for structured data management
  in software development, focusing on storage, processing, and security?"
- assumptions_made: [
  "Data type: Structured/database data",
  "Domain: Software development",
  "Focus: Storage, processing, security"
]
- ambiguities_resolved: []
- ambiguities_remaining: ["Specific data format", "Scale requirements"]
- confidence_score: 0.6 (lower due to assumptions)
"""


class QueryTransformationAgent(BaseResearchAgent[ResearchDependencies, TransformedQuery]):
    """Agent responsible for transforming and optimizing user queries for research.

    This agent takes vague or broad queries and transforms them into specific,
    actionable research questions with clear scope and search strategies.
    """

    def __init__(
        self,
        config: AgentConfiguration | None = None,
        dependencies: ResearchDependencies | None = None,
    ):
        """Initialize the query transformation agent.

        Args:
            config: Optional agent configuration. If not provided, uses defaults.
            dependencies: Optional research dependencies.
        """
        if config is None:
            config = AgentConfiguration(
                agent_name="query_transformation",
                agent_type="transformation",
            )
        super().__init__(config=config, dependencies=dependencies)

        # Register dynamic instructions
        @self.agent.instructions
        async def add_transformation_context(ctx: RunContext[ResearchDependencies]) -> str:
            """Inject comprehensive query transformation context as instructions."""
            query = ctx.deps.research_state.user_query
            metadata = ctx.deps.research_state.metadata
            conversation = metadata.conversation_messages if metadata else []

            # Format conversation context
            conversation_context = self._format_conversation_context(conversation)

            # Build comprehensive clarification context
            context_parts = self._build_clarification_context(metadata)

            # Use enhanced template with all context components
            return QUERY_TRANSFORMATION_SYSTEM_PROMPT_TEMPLATE.format(
                original_query=query,
                conversation_context=conversation_context,
                clarification_assessment=context_parts["assessment"],
                missing_dimensions=context_parts["missing_dimensions"],
                clarification_questions=context_parts["questions"],
                clarification_answers=context_parts["answers"],
            )

    def _build_clarification_context(self, metadata: ResearchMetadata | None) -> dict[str, str]:
        """Build comprehensive clarification context from metadata.

        Args:
            metadata: Research metadata containing clarification data

        Returns:
            Dictionary with formatted context components
        """
        default_context = {
            "assessment": "No clarification assessment available.",
            "missing_dimensions": "No missing dimensions identified.",
            "questions": "No clarification questions asked.",
            "answers": "No user responses provided.",
        }

        if not metadata or not metadata.clarification:
            return default_context

        clarification = metadata.clarification
        context = {}

        # Extract assessment and reasoning
        if clarification.assessment:
            assessment_text = (
                clarification.assessment.get("assessment_reasoning") or "Query assessment pending."
            )
            context["assessment"] = assessment_text
        else:
            context["assessment"] = default_context["assessment"]

        # Extract missing dimensions
        if clarification.assessment and clarification.assessment.get("missing_dimensions"):
            dimensions = clarification.assessment.get("missing_dimensions", [])
            context["missing_dimensions"] = "\n".join(f"- {dim}" for dim in dimensions)
        else:
            context["missing_dimensions"] = default_context["missing_dimensions"]

        # Extract questions
        if clarification.request and clarification.request.questions:
            questions_text = []
            for i, q in enumerate(clarification.request.get_sorted_questions(), 1):
                required = " [REQUIRED]" if q.is_required else " [OPTIONAL]"
                questions_text.append(f"{i}. {q.question}{required}")
                if q.context:
                    questions_text.append(f"   Context: {q.context}")
                if q.choices:
                    questions_text.append(f"   Options: {', '.join(q.choices)}")
            context["questions"] = "\n".join(questions_text)
        else:
            context["questions"] = default_context["questions"]

        # Extract answers
        if clarification.request and clarification.response:
            answers_text = []
            for q in clarification.request.get_sorted_questions():
                answer = clarification.response.get_answer_for_question(q.id)
                if answer:
                    if answer.skipped:
                        answers_text.append(f"Q: {q.question}\nA: [SKIPPED]")
                    else:
                        answers_text.append(f"Q: {q.question}\nA: {answer.answer}")
            if answers_text:
                context["answers"] = "\n".join(answers_text)
            else:
                context["answers"] = "User provided no responses."
        else:
            context["answers"] = default_context["answers"]

        return context

    def _get_default_system_prompt(self) -> str:
        """Get the base system prompt for this agent."""
        return "You are a Query Transformation Specialist focused on optimizing research queries."

    def _get_output_type(self) -> type[TransformedQuery]:
        """Get the output type for this agent."""
        return TransformedQuery


# Lazy initialization of module-level instance
_query_transformation_agent_instance = None


def get_query_transformation_agent() -> QueryTransformationAgent:
    """Get or create the query transformation agent instance."""
    global _query_transformation_agent_instance
    if _query_transformation_agent_instance is None:
        _query_transformation_agent_instance = QueryTransformationAgent()
        logfire.info("Initialized query_transformation agent")
    return _query_transformation_agent_instance


# For backward compatibility, create a property-like access
class _LazyAgent:
    """Lazy proxy for QueryTransformationAgent that delays instantiation."""

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the actual agent instance."""
        return getattr(get_query_transformation_agent(), name)


query_transformation_agent = _LazyAgent()
