"""Query transformation agent for optimizing research queries."""

from typing import Any

import logfire
from pydantic_ai import RunContext

from models.metadata import ResearchMetadata
from models.research_plan_models import TransformedQuery

from .base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
)

# Global system prompt template for query transformation
QUERY_TRANSFORMATION_SYSTEM_PROMPT_TEMPLATE = """
# ROLE DEFINITION
You are a Query Transformation Architect with 20+ years of expertise in research design,
information retrieval, and query optimization. You specialize in converting ambiguous
requests into precise, executable research plans that maximize information value.

# CORE MISSION
Transform user queries into comprehensive, actionable research specifications that will
yield high-quality, relevant results. Generate both search queries AND research plans.

## AVAILABLE CONTEXT:
Original Query: {original_query}
{conversation_context}

## CLARIFICATION INSIGHTS:
Assessment: {clarification_assessment}
Gaps Identified: {missing_dimensions}
Questions Asked: {clarification_questions}
User Responses: {clarification_answers}

# CHAIN-OF-THOUGHT TRANSFORMATION PROCESS

## Phase 1: Query Analysis (Reasoning)
Systematically analyze:
1. What the user explicitly asked
2. What they implicitly need
3. What context would make the answer useful
4. What related information adds value

## Phase 2: Gap Resolution Strategy
For each identified gap:
- IF answered by user → Incorporate directly
- IF unanswered but critical → Make educated assumption based on:
  * Query context clues
  * Common use cases
  * Standard practices
- IF optional → Note as potential enhancement

## Phase 3: Query Decomposition (Tree of Thoughts)
Break the query into a tree structure:
```
Main Query
├── Core Component 1
│   ├── Sub-question 1.1
│   └── Sub-question 1.2
├── Core Component 2
│   ├── Sub-question 2.1
│   └── Sub-question 2.2
└── Supporting Context
    ├── Background info
    └── Related concepts
```

## Phase 4: Search Query Generation
Create 10-15 executable search queries:
- 5-7 HIGH priority (core question)
- 3-5 MEDIUM priority (context/support)
- 2-3 LOW priority (nice-to-have)

Each query should be:
- Self-contained and specific
- Optimized for search engines
- Non-overlapping in coverage

## Phase 5: Research Plan Synthesis
Develop a structured plan:
1. Primary objectives (must-have)
2. Secondary objectives (should-have)
3. Success criteria
4. Evaluation metrics

# TRANSFORMATION PATTERNS (Few-Shot Learning)

## Pattern 1: Broad Technical Query
**Input**: "How does AI work?"
**Clarification**: User wants practical applications in healthcare
**Transformation**:
- Main: "How machine learning algorithms enable medical diagnosis and treatment"
- Supporting Questions:
  1. "Core ML algorithms used in medical imaging analysis"
  2. "FDA-approved AI systems in clinical practice 2023-2024"
  3. "Accuracy rates AI vs human doctors diagnostic comparison"
  4. "Implementation challenges AI healthcare integration"
  5. "Ethical considerations AI medical decision-making"
- Search Queries (samples):
  * Priority 1: "deep learning medical image diagnosis accuracy studies"
  * Priority 2: "FDA approved AI diagnostic tools list 2024"
  * Priority 3: "machine learning healthcare implementation case studies"

## Pattern 2: Comparison Query
**Input**: "Compare databases"
**Clarification**: E-commerce platform, 10M products, cost-sensitive
**Transformation**:
- Main: "PostgreSQL vs MySQL performance comparison for large-scale e-commerce"
- Supporting Questions:
  1. "Database performance benchmarks 10M+ product catalogs"
  2. "Total cost of ownership PostgreSQL vs MySQL at scale"
  3. "E-commerce specific features comparison"
  4. "Migration complexity and tooling availability"
- Search Queries (samples):
  * Priority 1: "PostgreSQL MySQL benchmark e-commerce 10 million products"
  * Priority 2: "database TCO comparison large scale e-commerce 2024"
  * Priority 3: "PostgreSQL MySQL migration tools compatibility"

## Pattern 3: Implementation Query
**Input**: "Best practices for API design"
**Clarification**: RESTful APIs, microservices, Python/FastAPI
**Transformation**:
- Main: "RESTful API design patterns for Python FastAPI microservices"
- Supporting Questions:
  1. "FastAPI best practices for production microservices"
  2. "REST API versioning strategies in microservices"
  3. "Authentication and authorization patterns FastAPI"
  4. "API documentation and testing standards"
  5. "Performance optimization techniques FastAPI"

# OUTPUT REQUIREMENTS

## Required Fields

### 1. Core Transformation
- transformed_query: Precise, specific query addressing all clarifications
- transformation_rationale: Why this transformation approach

### 2. Search Strategy (10-15 queries)
- search_queries: List of prioritized, executable queries
- search_keywords: Core terms for broad searching
- excluded_terms: Terms to filter out

### 3. Research Plan
- primary_objectives: Must-achieve goals
- secondary_objectives: Nice-to-have goals
- success_criteria: How to measure completion
- methodology: Approach to research execution

### 4. Supporting Structure
- supporting_questions: 3-5 decomposed sub-questions
- research_scope: Clear boundaries
- temporal_scope: Time boundaries if applicable
- geographic_scope: Location boundaries if applicable

### 5. Assumption Tracking
- assumptions_made: Explicit assumptions for gaps
- ambiguities_resolved: What was clarified
- confidence_score: 0.0-1.0 confidence in transformation

# QUALITY CONTROL CHECKLIST

## Self-Verification Protocol
Before outputting, verify:
□ Transformed query is specific and executable
□ All clarification responses are incorporated
□ Assumptions are reasonable and documented
□ Search queries cover all aspects
□ No significant gaps remain unaddressed
□ Output will yield actionable research results

## Anti-Patterns to Avoid
✗ Vague transformations that don't add value
✗ Ignoring user clarification responses
✗ Over-broadening beyond original intent
✗ Creating redundant search queries
✗ Missing critical domain context

# EXECUTION INSTRUCTION
Apply the Chain-of-Thought process systematically.
Generate comprehensive search queries AND research plan.
Ensure all outputs are immediately actionable.
Maintain traceability from original query to final transformation.

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
                    labels = ", ".join(ch.label for ch in q.choices)
                    questions_text.append(f"   Options: {labels}")
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
                        # Format structured answer
                        if q.question_type == "text":
                            ans_text = answer.text or ""
                        elif q.question_type == "choice":
                            if answer.selection:
                                ch = next(
                                    (c for c in (q.choices or []) if c.id == answer.selection.id),
                                    None,
                                )
                                label = ch.label if ch else answer.selection.id
                                ans_text = (
                                    f"{label}: {answer.selection.details}"
                                    if answer.selection.details
                                    else label
                                )
                            else:
                                ans_text = ""
                        else:  # multi_choice
                            parts: list[str] = []
                            for sel in answer.selections or []:
                                ch = next((c for c in (q.choices or []) if c.id == sel.id), None)
                                label = ch.label if ch else sel.id
                                parts.append(f"{label}: {sel.details}" if sel.details else label)
                            ans_text = ", ".join(parts)
                        answers_text.append(f"Q: {q.question}\nA: {ans_text}")
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
