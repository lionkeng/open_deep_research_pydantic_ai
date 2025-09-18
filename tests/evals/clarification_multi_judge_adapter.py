"""Multi-Judge Adapter for ClarificationAgent Evaluation.

This module provides the adapter implementation for evaluating ClarificationAgent
outputs using the generalized multi-judge framework.
"""

from typing import Any

from agents.clarification import ClarifyWithUser
from tests.evals.base_multi_judge import AgentEvaluationAdapter, EvaluationDimension, JudgeExpertise


class ClarificationMultiJudgeAdapter(AgentEvaluationAdapter[str, ClarifyWithUser]):
    """Adapter for evaluating ClarificationAgent outputs with multi-judge consensus."""

    def get_evaluation_dimensions(self) -> list[EvaluationDimension]:
        """Return evaluation dimensions specific to clarification tasks."""
        return [
            EvaluationDimension(
                name="relevance",
                description="How relevant are the clarification questions to the original query?",
                weight=1.2,
            ),
            EvaluationDimension(
                name="ambiguity_detection",
                description="How well does the agent identify key ambiguities that need clarification?",
                weight=1.3,  # Highest weight - core functionality
            ),
            EvaluationDimension(
                name="helpfulness",
                description="Would answering these clarification questions lead to better research results?",
                weight=1.1,
            ),
            EvaluationDimension(
                name="clarity",
                description="Are the clarification questions clear, specific, and well-formulated?",
                weight=1.0,
            ),
            EvaluationDimension(
                name="completeness",
                description="Does the clarification cover all major ambiguities in the query?",
                weight=1.1,
            ),
            EvaluationDimension(
                name="framework_adherence",
                description="Does the agent follow the 4-dimension framework (Scope, Audience, Detail, Purpose)?",
                weight=1.0,
            ),
        ]

    def format_output_for_evaluation(self, output: ClarifyWithUser) -> str:
        """Format clarification output into a string for evaluation."""
        if not output.needs_clarification or not output.request:
            return "No clarification needed"

        lines = [f"Needs Clarification: {output.needs_clarification}", "\nQuestions:"]

        for q in output.request.questions:
            lines.append(f"- {q.question}")
            if q.question_type == "choice" and q.choices:
                lines.append(f"  Choices: {', '.join(q.choices)}")
            lines.append(f"  Type: {q.question_type}, Required: {q.is_required}")

        if output.missing_dimensions:
            lines.append(f"\nMissing Dimensions: {', '.join(output.missing_dimensions)}")

        if hasattr(output, "assessment_reasoning") and output.assessment_reasoning:
            lines.append(f"\nReasoning: {output.assessment_reasoning}")

        return "\n".join(lines)

    def create_evaluation_prompt(
        self, input: str, output: ClarifyWithUser, context: dict[str, Any] | None = None
    ) -> str:
        """Create the evaluation prompt for judges."""

        # Format questions if present
        if output.request and output.request.questions:
            questions_text = "\n".join(
                [
                    f"- {q.question} (Type: {q.question_type}, Required: {q.is_required})"
                    for q in output.request.questions
                ]
            )
        else:
            questions_text = "No clarification questions generated"

        prompt = f"""
        Original Query: {input}

        Clarification Response:
        Needs Clarification: {output.needs_clarification}

        Questions:
        {questions_text}

        Missing Dimensions Identified: {", ".join(output.missing_dimensions) if output.missing_dimensions else "None"}
        Agent's Reasoning: {output.assessment_reasoning if hasattr(output, "assessment_reasoning") else "Not provided"}
        """

        if context:
            prompt += f"\n\nAdditional Context: {context}"

        prompt += """

        Please evaluate this clarification response according to the dimensions specified in your system prompt.
        Consider whether the agent correctly identified ambiguities and asked appropriate clarification questions.
        """

        return prompt

    def is_output_valid(self, output: ClarifyWithUser) -> bool:
        """Check if the clarification output is valid for evaluation."""
        # Valid if it either doesn't need clarification or has questions
        if not output.needs_clarification:
            return True
        return output.request is not None and len(output.request.questions) > 0

    def get_expertise_context(self, expertise: JudgeExpertise) -> str:
        """Get expertise-specific context for clarification evaluation."""
        contexts = {
            JudgeExpertise.TECHNICAL: (
                "You have particular expertise in technical and programming-related queries. "
                "Pay special attention to technical ambiguities and implementation details. "
            ),
            JudgeExpertise.SCIENTIFIC: (
                "You have particular expertise in scientific research and academic queries. "
                "Focus on research methodology clarity and scientific rigor. "
            ),
            JudgeExpertise.BUSINESS: (
                "You have particular expertise in business and commercial queries. "
                "Consider business context, stakeholder perspectives, and practical implications. "
            ),
            JudgeExpertise.CREATIVE: (
                "You have particular expertise in creative and artistic queries. "
                "Evaluate how well the clarifications capture creative intent and vision. "
            ),
            JudgeExpertise.GENERAL: (
                "You are a general-purpose evaluator with broad expertise. "
                "Consider all aspects of the clarification holistically. "
            ),
        }
        return contexts.get(expertise, contexts[JudgeExpertise.GENERAL])
