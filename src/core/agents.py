"""Central agent registry following Pydantic-AI best practices.

This module defines all research agents as module-level singletons and provides
a type-safe registry for agent coordination, eliminating circular import issues.
"""

from __future__ import annotations

from typing import Any

import logfire
from pydantic_ai import Agent, ModelRetry, RunContext

from agents.base import ResearchDependencies
from models.core import BriefGenerationResult, ClarificationResult, TransformedQueryResult

# Clarification Agent following Pydantic-AI patterns
clarification_agent = Agent[ResearchDependencies, ClarificationResult](
    "openai:gpt-4o",
    deps_type=ResearchDependencies,
    output_type=ClarificationResult,
    system_prompt="""You are a research clarification assistant following structured
assessment protocols.

Your role is to assess whether user queries need clarification to produce high-quality research.

CRITICAL ASSESSMENT CRITERIA:
Ask for clarification if ANY of these essential dimensions are missing:

1. **Audience Level & Purpose**:
   - Who is this research for? (academic, business, personal, student)
   - What background knowledge level? (beginner, expert, professional)
   - What's the intended use? (presentation, analysis, decision-making, learning)

2. **Scope & Focus Areas**:
   - Is the topic too broad without specific focus?
   - Are there multiple aspects that need prioritization?
   - What depth of coverage is needed?

3. **Context & Constraints**:
   - Specific time periods or geographic regions?
   - Industry vs. theoretical focus?
   - Any particular perspectives or methodologies to apply?

EXAMPLES OF QUERIES REQUIRING CLARIFICATION:
- "What is [broad topic]?" → Ask about audience, purpose, specific focus
- "Compare [general category]" → Ask which specific items, criteria, timeframe
- "Research [technology]" → Ask about technical depth, applications, audience
- "Analyze [industry]" → Ask about specific aspects, timeframe, geographic scope

OUTPUT REQUIREMENTS:
- If clarification is needed: provide detailed question addressing missing dimensions
- If no clarification needed: provide brief verification acknowledging sufficient detail
- Always include confidence score and breadth assessment
- Explain your reasoning clearly""",
)


@clarification_agent.system_prompt
async def clarification_system_prompt(ctx: RunContext[ResearchDependencies]) -> str:
    """Dynamic system prompt with current context."""
    current_query = ctx.deps.research_state.user_query
    return f"""Current research context: analyzing query "{current_query}"

Apply the clarification assessment criteria to determine if this query provides
sufficient detail for comprehensive research."""


@clarification_agent.tool
async def assess_query_breadth(ctx: RunContext[ResearchDependencies], query: str) -> dict[str, Any]:
    """Assess the breadth and specificity of a research query."""

    # Broad indicators that suggest need for clarification
    broad_indicators = [
        "what is",
        "how does",
        "explain",
        "overview of",
        "research",
        "analyze",
        "compare",
        "study",
        "investigate",
    ]

    # Check for missing context dimensions
    query_lower = query.lower()
    missing_context = {
        "audience_level": not any(
            word in query_lower
            for word in ["beginner", "expert", "academic", "business", "student", "professional"]
        ),
        "purpose": not any(
            word in query_lower
            for word in ["for", "because", "need", "purpose", "use", "presentation", "analysis"]
        ),
        "scope_definition": len(query.split()) < 8,  # Very short queries lack scope
        "temporal_context": not any(
            word in query_lower
            for word in ["recent", "current", "2024", "2023", "past", "future", "during", "since"]
        ),
        "specificity": any(indicator in query_lower for indicator in broad_indicators),
    }

    breadth_score = sum(missing_context.values()) / len(missing_context)

    return {
        "breadth_score": breadth_score,
        "missing_dimensions": [k for k, v in missing_context.items() if v],
        "has_broad_indicators": any(indicator in query_lower for indicator in broad_indicators),
        "word_count": len(query.split()),
        "assessment": "broad"
        if breadth_score > 0.6
        else "specific"
        if breadth_score < 0.3
        else "moderate",
    }


@clarification_agent.output_validator
async def validate_clarification_result(
    ctx: RunContext[ResearchDependencies], output: ClarificationResult
) -> ClarificationResult:
    """Validate clarification output and ensure quality."""

    # Check for consistency between assessment and output
    if output.needs_clarification and not output.question.strip():
        raise ModelRetry("When clarification is needed, you must provide a detailed question")

    if not output.needs_clarification and not output.verification.strip():
        raise ModelRetry("When no clarification is needed, you must provide verification text")

    # Check question quality if clarification is needed
    if output.needs_clarification and len(output.question.split()) < 10:
        raise ModelRetry(
            "Clarification questions should be detailed and specific (at least 10 words)"
        )

    # Ensure reasoning is provided
    if not output.assessment_reasoning.strip():
        raise ModelRetry("You must provide clear reasoning for your clarification decision")

    return output


# Query Transformation Agent
transformation_agent = Agent[ResearchDependencies, TransformedQueryResult](
    "openai:gpt-4o",
    deps_type=ResearchDependencies,
    output_type=TransformedQueryResult,
    system_prompt="""You are a query transformation specialist that converts broad
research queries into specific, actionable research questions.

TRANSFORMATION PRINCIPLES:

1. **Maximize Specificity**: Include all known context and requirements
2. **Preserve Intent**: Maintain the user's original research intent
3. **Add Structure**: Break complex queries into structured components
4. **Identify Gaps**: Note what information is still missing or could be refined
5. **Enhance Clarity**: Make queries more precise and actionable

TRANSFORMATION PROCESS:
- Analyze the original query for implicit assumptions
- Incorporate any clarification responses provided
- Add specificity without making unwarranted assumptions
- Generate supporting questions that complement the main query
- Assess the complexity and scope of research required

OUTPUT REQUIREMENTS:
- Transform query to be more specific and actionable
- Provide clear rationale for all transformations made
- Include specificity score reflecting improvement achieved
- List supporting questions that enhance research coverage""",
)


@transformation_agent.tool
async def enhance_query_specificity(
    ctx: RunContext[ResearchDependencies], original_query: str, clarification_data: dict[str, Any]
) -> dict[str, Any]:
    """Enhance query specificity using clarification responses."""

    # Extract clarification responses if available
    clarification_responses = clarification_data.get("clarification_responses", {})

    # Analyze query characteristics with explicit types
    word_count: int = len(original_query.split())
    query_analysis = {
        "word_count": word_count,
        "has_temporal_context": any(
            word in original_query.lower()
            for word in ["recent", "current", "2024", "2023", "past", "future", "when", "since"]
        ),
        "has_geographic_context": any(
            word in original_query.lower()
            for word in ["in", "at", "country", "region", "global", "local", "worldwide"]
        ),
        "domain_indicators": [],
    }

    # Estimate specificity improvement potential
    base_specificity = min(0.9, word_count / 20)  # Longer queries more specific
    if clarification_responses:
        base_specificity += min(0.3, len(clarification_responses) * 0.1)

    return {
        "base_specificity": base_specificity,
        "query_analysis": query_analysis,
        "clarification_count": len(clarification_responses),
        "enhancement_potential": 1.0 - base_specificity,
    }


@transformation_agent.output_validator
async def validate_transformation_result(
    ctx: RunContext[ResearchDependencies], output: TransformedQueryResult
) -> TransformedQueryResult:
    """Validate transformation quality and completeness."""

    # Ensure meaningful transformation occurred
    if output.original_query.strip().lower() == output.transformed_query.strip().lower():
        raise ModelRetry(
            "You must meaningfully transform the query - "
            "it should be more specific than the original"
        )

    # Check for minimum specificity improvement
    if output.specificity_score < 0.2:
        raise ModelRetry(
            "Transformation should significantly improve specificity (score should be at least 0.2)"
        )

    # Ensure rationale explains the changes
    if len(output.transformation_rationale.split()) < 15:
        raise ModelRetry(
            "Provide a detailed rationale explaining how and why you transformed the query"
        )

    # Check for supporting questions when dealing with complex topics
    if output.complexity_assessment == "high" and len(output.supporting_questions) == 0:
        raise ModelRetry(
            "High complexity topics should include supporting questions for comprehensive research"
        )

    return output


# Brief Generation Agent
brief_agent = Agent[ResearchDependencies, BriefGenerationResult](
    "openai:gpt-4o",
    deps_type=ResearchDependencies,
    output_type=BriefGenerationResult,
    system_prompt="""You are a research brief specialist that creates comprehensive research plans.

Your role is to generate detailed research briefs that guide thorough investigation of topics.

BRIEF GENERATION PROCESS:

1. **Analyze Requirements**: Review the research query and any transformation context
2. **Identify Scope**: Determine research boundaries and key focus areas
3. **Define Objectives**: Establish specific, measurable research goals
4. **Plan Methodology**: Suggest appropriate research approaches and methods
5. **Assess Complexity**: Evaluate research difficulty and resource requirements
6. **Anticipate Challenges**: Identify potential obstacles and mitigation strategies

BRIEF STRUCTURE:
- Clear research objectives and scope definition
- Key research areas and questions to investigate
- Suggested methodologies and approaches
- Expected complexity and time estimates
- Potential challenges and success criteria
- Recommended sources and resources

QUALITY STANDARDS:
- Brief should be actionable and specific
- Include concrete next steps for research execution
- Balance comprehensiveness with practical feasibility
- Provide realistic complexity and time assessments""",
)


@brief_agent.tool
async def analyze_research_requirements(
    ctx: RunContext[ResearchDependencies], transformed_query: str, specificity_score: float
) -> dict[str, Any]:
    """Analyze research requirements from transformed query."""

    # Estimate complexity based on query characteristics
    query_words = transformed_query.split()
    complexity_indicators = {
        "comparative": any(
            word in transformed_query.lower()
            for word in ["compare", "versus", "vs", "contrast", "difference"]
        ),
        "analytical": any(
            word in transformed_query.lower()
            for word in ["analyze", "analysis", "evaluate", "assess", "examine"]
        ),
        "temporal": any(
            word in transformed_query.lower()
            for word in ["trend", "over time", "historical", "future", "forecast"]
        ),
        "quantitative": any(
            word in transformed_query.lower()
            for word in ["measure", "metric", "data", "statistics", "number"]
        ),
    }

    # Base complexity assessment
    complexity_score = len([k for k, v in complexity_indicators.items() if v]) / len(
        complexity_indicators
    )

    # Adjust based on specificity
    if specificity_score > 0.8:
        estimated_complexity = "low" if complexity_score < 0.3 else "medium"
    elif specificity_score > 0.5:
        estimated_complexity = "medium" if complexity_score < 0.6 else "high"
    else:
        estimated_complexity = "high"  # Low specificity usually means high complexity

    return {
        "complexity_indicators": complexity_indicators,
        "complexity_score": complexity_score,
        "estimated_complexity": estimated_complexity,
        "query_length": len(query_words),
        "research_domains": [],  # Would be populated with domain detection
    }


@brief_agent.output_validator
async def validate_brief_result(
    ctx: RunContext[ResearchDependencies], output: BriefGenerationResult
) -> BriefGenerationResult:
    """Validate brief generation quality and completeness."""

    # Check brief length and content quality
    if len(output.brief_text.split()) < 50:
        raise ModelRetry(
            "Research brief should be more detailed and comprehensive (at least 50 words)"
        )

    # Ensure key research areas are identified
    if not output.key_research_areas:
        raise ModelRetry("You must identify at least one key research area")

    # Check for reasonable confidence given complexity
    if output.estimated_complexity == "high" and output.confidence_score > 0.9:
        raise ModelRetry(
            "High confidence seems unrealistic for high complexity research - "
            "consider lowering confidence or complexity assessment"
        )

    # Ensure practical elements are included for medium/high complexity
    if output.estimated_complexity in ["medium", "high"]:
        if not output.methodology_suggestions:
            raise ModelRetry("For medium/high complexity research, include methodology suggestions")
        if not output.potential_challenges:
            raise ModelRetry("For medium/high complexity research, identify potential challenges")

    return output


class AgentCoordinator:
    """Type-safe agent coordinator using Pydantic-AI agents."""

    def __init__(self):
        self.agents = {
            "clarification": clarification_agent,
            "transformation": transformation_agent,
            "brief": brief_agent,
        }
        self._agent_stats = {name: {"calls": 0, "errors": 0} for name in self.agents}

    def get_agent(self, agent_type: str) -> Agent[ResearchDependencies, Any]:
        """Get agent by type with validation."""
        if agent_type not in self.agents:
            available = ", ".join(self.agents.keys())
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {available}")
        return self.agents[agent_type]

    async def run_agent(
        self, agent_type: str, prompt: str, deps: ResearchDependencies, **kwargs: Any
    ) -> Any:
        """Run agent with error tracking and logging."""
        agent = self.get_agent(agent_type)

        try:
            self._agent_stats[agent_type]["calls"] += 1

            logfire.info(f"Running {agent_type} agent", prompt=prompt[:100])
            result = await agent.run(prompt, deps=deps)

            logfire.info(f"Agent {agent_type} completed successfully")
            return result

        except Exception as e:
            self._agent_stats[agent_type]["errors"] += 1
            logfire.error(f"Agent {agent_type} failed: {e}", error=str(e))
            raise

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Get agent usage statistics."""
        return self._agent_stats.copy()

    def reset_stats(self) -> None:
        """Reset agent statistics."""
        for name in self._agent_stats:
            self._agent_stats[name] = {"calls": 0, "errors": 0}


# Global coordinator instance
coordinator = AgentCoordinator()
