"""Query transformation agent for converting broad queries into specific research questions.

This agent takes user responses to clarification questions and transforms them into
focused, actionable research questions that can be effectively researched.
"""

import re
from datetime import datetime
from typing import Any

import logfire

from open_deep_research_with_pydantic_ai.agents.base import (
    BaseResearchAgent,
    ResearchDependencies,
    coordinator,
)
from open_deep_research_with_pydantic_ai.models.research import TransformedQuery


class QueryTransformationAgent(BaseResearchAgent[ResearchDependencies, TransformedQuery]):
    """Agent responsible for transforming broad queries into specific research questions.

    This agent takes the original query along with user responses to clarification
    questions and generates focused, actionable research questions.
    """

    def __init__(self):
        """Initialize the query transformation agent."""
        super().__init__(
            name="query_transformation_agent",
            output_type=TransformedQuery,
        )

    def _get_default_system_prompt(self) -> str:
        """Get the system prompt for query transformation."""
        return """You are a research query transformation specialist. Your role is to transform
broad, vague, or unclear research queries into specific, focused, and actionable research
questions.

Given:
1. An original research query from the user
2. User responses to clarification questions
3. Assessment of what aspects need more specificity

Your task is to:
1. Synthesize the original query with the clarification responses
2. Identify the core research intent and objectives
3. Transform the query into specific, focused research questions
4. Provide supporting questions if needed (max 2)
5. Explain your transformation rationale

Guidelines for transformation:
- Preserve the user's original intent while adding specificity
- Include temporal constraints when provided (time periods, dates)
- Add geographical focus when specified
- Incorporate domain-specific constraints and context
- Break complex queries into a primary question + supporting questions
- Ensure questions are researchable with available sources
- Make questions specific enough to guide focused research

Guidelines for specificity scoring:
- 0.0-0.3: Very broad, needs significant refinement
- 0.4-0.6: Moderately specific, some aspects could be refined
- 0.7-0.9: Highly specific, well-defined scope
- 1.0: Perfectly specific, no further refinement needed

Output the transformed query with:
- A primary research question (the main focus)
- Supporting questions (if needed, max 2)
- Transformation rationale explaining the changes
- Specificity score (0.0-1.0)
- Any remaining missing dimensions that could be further refined"""

    def _register_tools(self) -> None:
        """Register transformation-specific tools."""
        # No specific tools needed for this agent - it processes structured input
        pass

    async def transform_query(
        self,
        original_query: str,
        clarification_responses: dict[str, str],
        conversation_context: list[str] = None,
        deps: ResearchDependencies = None,
    ) -> TransformedQuery:
        """Transform a query based on clarification responses.

        Args:
            original_query: The original user query
            clarification_responses: Dict mapping questions to user responses
            conversation_context: Optional conversation history
            deps: Research dependencies

        Returns:
            TransformedQuery with the transformation results
        """
        try:
            logfire.info(
                "Starting query transformation",
                original_query=original_query,
                num_clarifications=len(clarification_responses),
            )

            # Build the transformation prompt
            prompt = self._build_transformation_prompt(
                original_query, clarification_responses, conversation_context
            )

            # Run the transformation
            if deps:
                result = await self.run(prompt, deps)
            else:
                # Fallback transformation if no deps provided
                result = self._create_fallback_transformation(
                    original_query, clarification_responses
                )

            # Store responses in the result
            result.clarification_responses = clarification_responses

            # Calculate transformation quality metrics
            result = self._enhance_transformation_metadata(result, original_query)

            logfire.info(
                "Query transformation completed",
                specificity_score=result.specificity_score,
                transformed_query=result.transformed_query[:100] + "...",
            )

            return result

        except Exception as e:
            logfire.error(
                f"Error in query transformation: {str(e)}",
                original_query=original_query,
                exc_info=True,
            )

            # Return fallback transformation
            return self._create_fallback_transformation(original_query, clarification_responses)

    def _build_transformation_prompt(
        self,
        original_query: str,
        clarification_responses: dict[str, str],
        conversation_context: list[str] = None,
    ) -> str:
        """Build the prompt for query transformation."""
        prompt_parts = [f"Original Query: {original_query}", "", "Clarification Responses:"]

        for question, response in clarification_responses.items():
            prompt_parts.append(f"Q: {question}")
            prompt_parts.append(f"A: {response}")
            prompt_parts.append("")

        if conversation_context:
            prompt_parts.extend(["Conversation Context:", "\n".join(conversation_context), ""])

        prompt_parts.extend(
            [
                "Transform this query into specific, focused research questions based on:",
                "1. The original user intent",
                "2. The clarification responses provided",
                "3. Research best practices for specificity and scope",
                "",
                "Provide your transformation following the system prompt guidelines.",
            ]
        )

        return "\n".join(prompt_parts)

    def _create_fallback_transformation(
        self,
        original_query: str,
        clarification_responses: dict[str, str],
    ) -> TransformedQuery:
        """Create a fallback transformation when AI processing fails."""
        logfire.warning("Creating fallback transformation due to processing error")

        # Basic transformation: combine original query with key responses
        enhanced_query = original_query
        context_additions = []

        for question, response in clarification_responses.items():
            if response.strip() and response.lower() not in ["no", "none", "n/a", ""]:
                if "time" in question.lower() or "when" in question.lower():
                    context_additions.append(f"during {response}")
                elif "where" in question.lower() or "region" in question.lower():
                    context_additions.append(f"in {response}")
                elif len(response) < 50:  # Short, specific responses
                    context_additions.append(response)

        if context_additions:
            enhanced_query = f"{original_query} ({', '.join(context_additions[:3])})"

        return TransformedQuery(
            original_query=original_query,
            transformed_query=enhanced_query,
            supporting_questions=[],
            transformation_rationale="Fallback transformation applied due to processing error",
            specificity_score=0.4,  # Conservative score for fallback
            missing_dimensions=["detailed scope analysis", "comprehensive transformation"],
            clarification_responses=clarification_responses,
            transformation_metadata={"method": "fallback", "error": "AI transformation failed"},
        )

    def _enhance_transformation_metadata(
        self, result: TransformedQuery, original_query: str
    ) -> TransformedQuery:
        """Enhance the transformation result with additional metadata."""
        metadata = result.transformation_metadata or {}

        # Calculate transformation metrics
        original_words = set(original_query.lower().split())
        transformed_words = set(result.transformed_query.lower().split())

        metadata.update(
            {
                "original_word_count": len(original_query.split()),
                "transformed_word_count": len(result.transformed_query.split()),
                "word_overlap_ratio": len(original_words & transformed_words)
                / max(len(original_words), 1),
                "transformation_timestamp": datetime.now().isoformat(),
                "agent": "query_transformation_agent",
            }
        )

        result.transformation_metadata = metadata
        return result

    async def validate_transformation_quality(
        self,
        transformation: TransformedQuery,
        deps: ResearchDependencies,
    ) -> dict[str, Any]:
        """Validate the quality of a transformation."""
        # This would use the LLM to validate with a validation prompt, but for now use basic
        # validation

        try:
            # This would use the LLM to validate, but for now return basic validation
            basic_validation = self._basic_transformation_validation(transformation)

            logfire.info(
                "Transformation validation completed",
                overall_quality=basic_validation["overall_score"],
            )

            return basic_validation

        except Exception as e:
            logfire.error(f"Error in transformation validation: {str(e)}")
            return {"overall_score": 5.0, "error": str(e)}

    def _basic_transformation_validation(self, transformation: TransformedQuery) -> dict[str, Any]:
        """Perform basic validation of transformation quality."""
        scores = {}

        # Specificity: compare length and detail
        original_len = len(transformation.original_query.split())
        transformed_len = len(transformation.transformed_query.split())
        specificity_score = min(10, max(1, (transformed_len / max(original_len, 1)) * 5))
        scores["specificity"] = specificity_score

        # Researchability: check for specific terms, constraints
        specific_terms = len(
            re.findall(
                r"\b\d{4}\b|\b(in|during|for|with|using)\b",
                transformation.transformed_query.lower(),
            )
        )
        researchability_score = min(10, max(3, 5 + specific_terms))
        scores["researchability"] = researchability_score

        # Intent preservation: word overlap
        original_words = set(transformation.original_query.lower().split())
        transformed_words = set(transformation.transformed_query.lower().split())
        overlap_ratio = len(original_words & transformed_words) / max(len(original_words), 1)
        intent_score = max(3, min(10, overlap_ratio * 10))
        scores["intent_preservation"] = intent_score

        # Clarity: basic length and structure check
        clarity_score = 8 if len(transformation.transformed_query) > 20 else 5
        scores["clarity"] = clarity_score

        overall_score = sum(scores.values()) / len(scores)

        return {
            "scores": scores,
            "overall_score": overall_score,
            "validation_method": "basic_heuristic",
        }


# Register the agent with the coordinator
query_transformation_agent = QueryTransformationAgent()
coordinator.register_agent(query_transformation_agent)
