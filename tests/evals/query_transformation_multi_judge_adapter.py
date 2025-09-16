"""Multi-Judge Adapter for QueryTransformationAgent Evaluation.

This module provides the adapter implementation for evaluating QueryTransformationAgent
outputs using the generalized multi-judge framework.
"""

from typing import List, Dict, Any, Optional

from tests.evals.base_multi_judge import (
    AgentEvaluationAdapter,
    EvaluationDimension,
    JudgeExpertise
)
from models.research_plan_models import TransformedQuery


class QueryTransformationMultiJudgeAdapter(AgentEvaluationAdapter[str, TransformedQuery]):
    """Adapter for evaluating QueryTransformationAgent outputs with multi-judge consensus."""

    def get_evaluation_dimensions(self) -> List[EvaluationDimension]:
        """Return evaluation dimensions specific to query transformation tasks."""
        return [
            EvaluationDimension(
                name="search_query_relevance",
                description="How relevant and useful are the generated search queries for the original query?",
                weight=1.3  # High weight - core functionality
            ),
            EvaluationDimension(
                name="objective_coverage",
                description="How well do the research objectives cover the key aspects of the query?",
                weight=1.2
            ),
            EvaluationDimension(
                name="plan_coherence",
                description="How logical and well-structured is the research plan?",
                weight=1.1
            ),
            EvaluationDimension(
                name="query_diversity",
                description="How diverse and comprehensive are the search queries?",
                weight=1.0
            ),
            EvaluationDimension(
                name="methodology_quality",
                description="How appropriate and thorough is the research methodology?",
                weight=1.0
            ),
            EvaluationDimension(
                name="transformation_completeness",
                description="Does the transformation capture all important aspects of the original query?",
                weight=1.2
            ),
            EvaluationDimension(
                name="actionability",
                description="How actionable and executable is the transformed query plan?",
                weight=1.1
            )
        ]

    def format_output_for_evaluation(self, output: TransformedQuery) -> str:
        """Format transformation output into a string for evaluation."""
        lines = [
            f"Confidence Score: {output.confidence_score:.2f}",
            "\nResearch Plan:",
            f"Main Objective: {output.research_plan.main_objective}"
        ]

        # Add objectives
        if output.research_plan.objectives:
            lines.append("\nObjectives:")
            for i, obj in enumerate(output.research_plan.objectives, 1):
                lines.append(f"{i}. {obj.objective}")
                if obj.key_questions:
                    for q in obj.key_questions[:2]:  # Limit to first 2 questions
                        lines.append(f"   - {q}")

        # Add methodology
        if output.research_plan.methodology:
            lines.append(f"\nMethodology: {output.research_plan.methodology}")

        # Add search queries
        if output.search_queries.queries:
            lines.append(f"\nSearch Queries ({len(output.search_queries.queries)} total):")
            for i, sq in enumerate(output.search_queries.queries[:5], 1):  # Limit to first 5
                lines.append(f"{i}. {sq.query} (Intent: {sq.search_intent})")

        # Add key topics
        if output.key_topics:
            lines.append(f"\nKey Topics: {', '.join(output.key_topics[:5])}")

        return "\n".join(lines)

    def create_evaluation_prompt(
        self,
        input: str,
        output: TransformedQuery,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create the evaluation prompt for judges."""

        # Format objectives
        objectives_text = "\n".join([
            f"- {obj.objective}" for obj in output.research_plan.objectives
        ]) if output.research_plan.objectives else "No objectives generated"

        # Format search queries
        search_queries_text = "\n".join([
            f"- {sq.query} (Intent: {sq.search_intent})"
            for sq in output.search_queries.queries[:10]  # Limit to first 10
        ]) if output.search_queries.queries else "No search queries generated"

        prompt = f"""
        Original Query: {input}

        Transformation Output:

        Main Objective: {output.research_plan.main_objective}

        Research Objectives ({len(output.research_plan.objectives)}):
        {objectives_text}

        Methodology: {output.research_plan.methodology or "Not specified"}

        Search Queries ({len(output.search_queries.queries)}):
        {search_queries_text}

        Key Topics: {', '.join(output.key_topics) if output.key_topics else 'None identified'}

        Confidence Score: {output.confidence_score:.2f}
        """

        if context:
            prompt += f"\n\nAdditional Context: {context}"

        prompt += """

        Please evaluate this query transformation according to the dimensions specified in your system prompt.
        Consider how well the agent decomposed the query into actionable research objectives and search queries.
        """

        return prompt

    def is_output_valid(self, output: TransformedQuery) -> bool:
        """Check if the transformation output is valid for evaluation."""
        # Valid if it has objectives and search queries
        return (
            output.research_plan is not None and
            len(output.research_plan.objectives) > 0 and
            output.search_queries is not None and
            len(output.search_queries.queries) > 0
        )

    def get_expertise_context(self, expertise: JudgeExpertise) -> str:
        """Get expertise-specific context for query transformation evaluation."""
        contexts = {
            JudgeExpertise.TECHNICAL: (
                "You have particular expertise in technical and programming-related research. "
                "Focus on the technical accuracy and specificity of search queries and objectives. "
            ),
            JudgeExpertise.SCIENTIFIC: (
                "You have particular expertise in scientific research methodology. "
                "Evaluate the research plan's rigor, methodology, and systematic approach. "
            ),
            JudgeExpertise.BUSINESS: (
                "You have particular expertise in business research and analysis. "
                "Consider market research aspects, competitive analysis, and business implications. "
            ),
            JudgeExpertise.CREATIVE: (
                "You have particular expertise in creative and exploratory research. "
                "Evaluate how well the transformation captures innovative angles and perspectives. "
            ),
            JudgeExpertise.GENERAL: (
                "You are a general-purpose evaluator with broad research expertise. "
                "Consider all aspects of the query transformation holistically. "
            )
        }
        return contexts.get(expertise, contexts[JudgeExpertise.GENERAL])
