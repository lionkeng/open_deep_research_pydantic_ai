"""User Clarification Agent for validating and refining research requests."""

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from open_deep_research_with_pydantic_ai.agents.base import (
    BaseResearchAgent,
    ResearchDependencies,
    coordinator,
)
from open_deep_research_with_pydantic_ai.core.events import (
    emit_stage_completed,
)
from open_deep_research_with_pydantic_ai.models.research import ResearchStage


class ClarificationResult(BaseModel):
    """Result of the clarification process."""

    is_clear: bool = Field(description="Whether the query is clear enough to proceed")
    clarified_query: str = Field(description="Clarified version of the query")
    clarifying_questions: list[str] = Field(
        default_factory=list, description="Questions to ask if query needs clarification"
    )
    scope_validation: str = Field(description="Validation of research scope")
    estimated_complexity: str = Field(
        default="medium", description="Estimated complexity: simple, medium, complex"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Any warnings about the research request"
    )


class ClarificationAgent(BaseResearchAgent[ResearchDependencies, ClarificationResult]):
    """Agent responsible for clarifying and validating research requests."""

    def __init__(self, model: str = "openai:gpt-4o"):
        """Initialize the clarification agent."""
        super().__init__(
            name="clarification_agent",
            model=model,
            output_type=ClarificationResult,
        )

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for clarification."""
        return """You are a research clarification specialist. Your role is to:

1. Validate that research requests are clear and well-defined
2. Identify any ambiguities or missing information
3. Suggest clarifying questions when needed
4. Ensure the scope is reasonable and achievable
5. Refine queries to be more specific and actionable

When analyzing a query:
- Check if the topic is clearly defined
- Verify that objectives are specific
- Identify any vague or ambiguous terms
- Assess if the scope is too broad or narrow
- Consider potential ethical or legal concerns

Always provide:
- A clarified version of the query (even if original is clear)
- Validation of the research scope
- Complexity assessment (simple, medium, complex)
- Any necessary clarifying questions
- Warnings about potential issues

Be helpful but thorough in ensuring the research request is well-defined."""

    def _register_tools(self) -> None:
        """Register clarification-specific tools.

        Note: These functions are registered as AI agent tools and called
        dynamically at runtime, not directly in code. Linter warnings about
        'function not accessed' are false positives.
        """

        @self.agent.tool
        async def validate_scope(  # pyright: ignore[reportUnusedFunction]
            _ctx: RunContext[ResearchDependencies], query: str
        ) -> dict[str, str]:
            """Validate the scope of a research query.

            Args:
                _ctx: Run context with dependencies (unused)
                query: Query to validate

            Returns:
                Validation results
            """
            # Check for common scope issues
            issues: list[str] = []
            suggestions: list[str] = []

            # Check query length
            if len(query.split()) < 5:
                issues.append("Query is too brief")
                suggestions.append("Provide more detail about what you want to research")

            if len(query.split()) > 500:
                issues.append("Query is extremely long")
                suggestions.append("Consider breaking down into multiple focused queries")

            # Check for vague terms
            vague_terms = [
                "everything about",
                "all aspects",
                "general information",
                "stuff about",
                "things related to",
            ]
            for term in vague_terms:
                if term.lower() in query.lower():
                    issues.append(f"Query contains vague term: '{term}'")
                    suggestions.append("Be more specific about what aspects you want to research")
                    break

            # Check for overly broad topics
            broad_topics = [
                "artificial intelligence",
                "climate change",
                "human history",
                "the universe",
                "consciousness",
            ]
            for topic in broad_topics:
                if topic.lower() in query.lower() and len(query.split()) < 15:
                    issues.append(f"Topic '{topic}' is very broad")
                    suggestions.append(f"Narrow down to specific aspects of {topic}")
                    break

            return {
                "has_issues": str(len(issues) > 0),
                "issues": "; ".join(issues) if issues else "None",
                "suggestions": "; ".join(suggestions) if suggestions else "None",
            }

        @self.agent.tool
        async def assess_complexity(  # pyright: ignore[reportUnusedFunction]
            _ctx: RunContext[ResearchDependencies], query: str
        ) -> str:
            """Assess the complexity of a research query.

            Args:
                _ctx: Run context with dependencies (unused)
                query: Query to assess

            Returns:
                Complexity level
            """
            # Simple heuristics for complexity assessment
            word_count = len(query.split())

            # Check for complex research indicators
            complex_indicators = [
                "comparative analysis",
                "systematic review",
                "meta-analysis",
                "longitudinal",
                "interdisciplinary",
                "comprehensive evaluation",
                "multiple perspectives",
                "historical evolution",
                "future projections",
                "causal relationships",
            ]

            complex_count = sum(
                1 for indicator in complex_indicators if indicator.lower() in query.lower()
            )

            # Check for technical domains
            technical_domains = [
                "quantum",
                "neurological",
                "pharmaceutical",
                "cryptographic",
                "genomic",
                "algorithmic",
                "theoretical physics",
                "advanced mathematics",
            ]

            technical_count = sum(
                1 for domain in technical_domains if domain.lower() in query.lower()
            )

            # Determine complexity
            if complex_count >= 2 or technical_count >= 1 or word_count > 100:
                return "complex"
            elif complex_count == 1 or word_count > 50:
                return "medium"
            else:
                return "simple"

    async def clarify_query(self, query: str, deps: ResearchDependencies) -> ClarificationResult:
        """Clarify and validate a research query.

        Args:
            query: Original research query
            deps: Research dependencies

        Returns:
            Clarification result
        """
        prompt = f"""Please analyze and clarify the following research query:

Query: {query}

Provide:
1. A clear assessment of whether the query is well-defined
2. A refined/clarified version of the query
3. Any clarifying questions if needed
4. Validation of the research scope
5. Complexity assessment
6. Any warnings or concerns"""

        result = await self.run(prompt, deps, stream=True)

        # Update research state
        deps.research_state.clarified_query = result.clarified_query

        # Emit stage completed event
        await emit_stage_completed(
            request_id=deps.research_state.request_id,
            stage=ResearchStage.CLARIFICATION,
            success=result.is_clear,
            result=result,
        )

        return result


# Register the agent with the coordinator
clarification_agent = ClarificationAgent()
coordinator.register_agent(clarification_agent)
