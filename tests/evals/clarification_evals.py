"""
Evaluation framework for ClarificationAgent using Pydantic Evals.

This module provides comprehensive evaluation capabilities for the clarification agent,
including custom evaluators, metrics, and LLM-as-judge patterns.
"""

import asyncio
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import httpx

from pydantic import BaseModel, Field
from pydantic_evals import Dataset, Case, Evaluator, evaluate, Report
from pydantic_ai import Agent

from src.agents.clarification import ClarificationAgent, ClarifyWithUser
from src.agents.base import ResearchDependencies
from src.models.api_models import APIKeys, ResearchMetadata
from src.models.core import ResearchState, ResearchStage
from src.models.clarification import ClarificationQuestion, ClarificationRequest


class ClarificationInput(BaseModel):
    """Input model for clarification evaluation."""
    query: str = Field(description="User query to evaluate")
    context: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional conversation context"
    )


class ClarificationExpectedOutput(BaseModel):
    """Expected output for clarification evaluation."""
    needs_clarification: bool = Field(description="Whether clarification is needed")
    min_questions: Optional[int] = Field(
        default=None,
        description="Minimum number of questions expected"
    )
    max_questions: Optional[int] = Field(
        default=None,
        description="Maximum number of questions expected"
    )
    dimension_categories: Optional[List[str]] = Field(
        default=None,
        description="Expected dimension categories from 4-framework"
    )
    key_themes: Optional[List[str]] = Field(
        default=None,
        description="Key themes that should appear in clarification questions"
    )
    expected_question_types: Optional[List[str]] = Field(
        default=None,
        description="Expected question types (text, choice, multi_choice)"
    )


class BinaryAccuracyEvaluator(Evaluator):
    """Evaluates binary correctness of clarification decision."""

    def evaluate(self, output: ClarifyWithUser, expected: ClarificationExpectedOutput) -> Dict[str, Any]:
        """Evaluate if clarification decision matches expected."""
        correct = output.needs_clarification == expected.needs_clarification
        return {
            "score": 1.0 if correct else 0.0,
            "correct": correct,
            "predicted": output.needs_clarification,
            "expected": expected.needs_clarification
        }


class DimensionCoverageEvaluator(Evaluator):
    """Evaluates coverage of the 4-dimension framework."""

    DIMENSION_KEYWORDS = {
        "audience_level": ["audience", "level", "technical", "background", "expertise", "beginner", "expert"],
        "scope_focus": ["scope", "focus", "aspect", "specific", "broad", "area", "domain"],
        "source_quality": ["source", "credibility", "academic", "industry", "quality", "reliability"],
        "deliverable": ["deliverable", "format", "output", "report", "summary", "presentation", "depth"]
    }

    def evaluate(self, output: ClarifyWithUser, expected: ClarificationExpectedOutput) -> Dict[str, Any]:
        """Evaluate dimension framework coverage."""
        if not output.needs_clarification or not output.request:
            # If no clarification needed, this evaluator is not applicable
            return {"score": None, "applicable": False}

        # Combine all text for analysis
        questions_text = " ".join([q.question for q in output.request.questions])
        all_text = " ".join([
            questions_text,
            " ".join(output.missing_dimensions),
            output.assessment_reasoning
        ]).lower()

        # Check which dimensions are covered
        covered_dimensions = []
        for dimension, keywords in self.DIMENSION_KEYWORDS.items():
            if any(keyword in all_text for keyword in keywords):
                covered_dimensions.append(dimension)

        # Calculate coverage score
        coverage_score = len(covered_dimensions) / len(self.DIMENSION_KEYWORDS)

        # Check against expected dimensions if provided
        dimension_match_score = 1.0
        if expected.dimension_categories:
            matched = sum(1 for exp_dim in expected.dimension_categories
                         if any(exp_dim in dim for dim in covered_dimensions))
            dimension_match_score = matched / len(expected.dimension_categories) if expected.dimension_categories else 0

        final_score = (coverage_score + dimension_match_score) / 2

        return {
            "score": final_score,
            "covered_dimensions": covered_dimensions,
            "coverage_rate": coverage_score,
            "dimension_match_score": dimension_match_score,
            "total_dimensions": len(self.DIMENSION_KEYWORDS)
        }


class QuestionRelevanceEvaluator(Evaluator):
    """Evaluates relevance and quality of clarification questions."""

    def evaluate(self, output: ClarifyWithUser, expected: ClarificationExpectedOutput) -> Dict[str, Any]:
        """Evaluate question relevance and quality."""
        if not output.needs_clarification or not output.request or not output.request.questions:
            return {"score": None, "applicable": False}

        scores = []
        question_scores = []

        for question in output.request.questions:
            q_scores = []

            # Check if question is not empty and substantial
            question_length_score = min(len(question.question) / 100, 1.0)  # Normalize to 0-1
            q_scores.append(question_length_score)

            # Check if question ends with question mark (basic quality)
            has_question_mark = 1.0 if question.question.strip().endswith("?") else 0.5
            q_scores.append(has_question_mark)

            # Check theme coverage if expected themes provided
            if expected.key_themes:
                question_lower = question.question.lower()
                theme_matches = sum(1 for theme in expected.key_themes
                                  if theme.lower() in question_lower)
                theme_score = theme_matches / len(expected.key_themes) if expected.key_themes else 0
                q_scores.append(theme_score)

            question_scores.append(sum(q_scores) / len(q_scores))

        # Average score across all questions
        avg_question_score = sum(question_scores) / len(question_scores) if question_scores else 0
        scores.append(avg_question_score)

        # Check if reasoning is provided
        reasoning_score = min(len(output.assessment_reasoning) / 100, 1.0) if output.assessment_reasoning else 0
        scores.append(reasoning_score)

        # Check question count expectations
        if expected.min_questions or expected.max_questions:
            count = len(output.request.questions)
            count_score = 1.0
            if expected.min_questions and count < expected.min_questions:
                count_score = count / expected.min_questions
            elif expected.max_questions and count > expected.max_questions:
                count_score = expected.max_questions / count
            scores.append(count_score)

        final_score = sum(scores) / len(scores)

        return {
            "score": final_score,
            "num_questions": len(output.request.questions),
            "has_reasoning": bool(output.assessment_reasoning),
            "question_scores": question_scores,
            "theme_coverage": theme_score if expected.key_themes else None
        }


class ConsistencyEvaluator(Evaluator):
    """Evaluates consistency across multiple runs of the same query."""

    def __init__(self, num_runs: int = 3):
        self.num_runs = num_runs

    async def evaluate_async(self, agent: ClarificationAgent, query: str) -> Dict[str, Any]:
        """Run multiple times and check consistency."""
        results = []

        # Create dependencies
        async with httpx.AsyncClient() as http_client:
            for _ in range(self.num_runs):
                state = ResearchState(
                    request_id=f"consistency-test-{_}",
                    user_query=query
                )
                deps = ResearchDependencies(
                    http_client=http_client,
                    api_keys=APIKeys(
                        openai=os.getenv("OPENAI_API_KEY")
                    ),
                    research_state=state
                )

                result = await agent.agent.run(query, deps=deps)
                results.append(result.output)

        # Check consistency of binary decision
        decisions = [r.needs_clarification for r in results]
        decision_consistency = all(d == decisions[0] for d in decisions)

        # Check consistency of dimensions (if clarification needed)
        dimension_consistency = 1.0
        if decisions[0]:  # If clarification is needed
            all_dimensions = [set(r.missing_dimensions) for r in results]
            if all_dimensions:
                # Calculate Jaccard similarity between dimension sets
                intersection = set.intersection(*all_dimensions) if all_dimensions else set()
                union = set.union(*all_dimensions) if all_dimensions else set()
                dimension_consistency = len(intersection) / len(union) if union else 1.0

        consistency_score = (1.0 if decision_consistency else 0.5) * dimension_consistency

        return {
            "score": consistency_score,
            "decision_consistency": decision_consistency,
            "dimension_consistency": dimension_consistency,
            "num_runs": self.num_runs,
            "all_decisions": decisions
        }


class MultiQuestionEvaluator(Evaluator):
    """Evaluates multi-question clarification capabilities."""

    def evaluate(self, output: ClarifyWithUser, expected: ClarificationExpectedOutput) -> Dict[str, Any]:
        """Evaluate multi-question generation and diversity."""
        if not output.needs_clarification or not output.request:
            return {"score": None, "applicable": False}

        questions = output.request.questions
        scores = []

        # 1. Question count score
        count = len(questions)
        count_score = 1.0
        if expected.min_questions and count < expected.min_questions:
            count_score = count / expected.min_questions
        elif expected.max_questions and count > expected.max_questions:
            count_score = expected.max_questions / count
        scores.append(count_score)

        # 2. Question type diversity
        question_types = set(q.question_type for q in questions)
        expected_types = expected.expected_question_types or ["text", "choice", "multi_choice"]
        type_coverage = len(question_types.intersection(expected_types)) / len(expected_types)
        scores.append(type_coverage)

        # 3. Required vs optional balance
        required_count = sum(1 for q in questions if q.is_required)
        optional_count = len(questions) - required_count
        balance_score = 1.0
        if required_count == 0 or optional_count == 0:
            balance_score = 0.5  # Penalize if all questions are same type
        scores.append(balance_score)

        # 4. Question uniqueness (no duplicate questions)
        unique_questions = len(set(q.question.lower().strip() for q in questions))
        uniqueness_score = unique_questions / len(questions) if questions else 0
        scores.append(uniqueness_score)

        # 5. Question ordering
        ordered_questions = output.request.get_sorted_questions()
        ordering_score = 1.0 if ordered_questions == sorted(questions, key=lambda q: q.order) else 0.5
        scores.append(ordering_score)

        final_score = sum(scores) / len(scores)

        return {
            "score": final_score,
            "num_questions": count,
            "question_types": list(question_types),
            "required_count": required_count,
            "optional_count": optional_count,
            "uniqueness_score": uniqueness_score,
            "type_coverage": type_coverage,
            "balance_score": balance_score
        }


class LLMJudgeEvaluator(Evaluator):
    """Uses an LLM to judge the quality of clarification questions."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.judge_agent = Agent(
            model=model,
            system_prompt="""You are an expert evaluator of clarification questions.
            Evaluate the quality of clarification questions based on:
            1. Relevance to the original query
            2. Identification of key ambiguities
            3. Helpfulness for providing better answers
            4. Clarity and specificity of the question"""
        )

    async def evaluate_async(
        self,
        query: str,
        output: ClarifyWithUser,
        expected: Optional[ClarificationExpectedOutput] = None
    ) -> Dict[str, Any]:
        """Use LLM to judge clarification quality."""

        if not output.needs_clarification or not output.request:
            return {"score": None, "applicable": False}

        questions_text = "\n".join([f"- {q.question} (Type: {q.question_type}, Required: {q.is_required})"
                                    for q in output.request.questions])

        evaluation_prompt = f"""
        Original Query: {query}

        Clarification Response:
        - Questions:\n{questions_text}
        - Missing Dimensions: {', '.join(output.missing_dimensions)}
        - Reasoning: {output.assessment_reasoning}

        Please evaluate this clarification on a scale of 0-10 for:
        1. Relevance (0-10): How relevant is the clarification to the query?
        2. Ambiguity Detection (0-10): How well does it identify the key ambiguities?
        3. Helpfulness (0-10): Would the answer help provide better research?
        4. Clarity (0-10): Is the clarification question clear and specific?

        Provide your evaluation as a JSON object with these scores and a brief explanation.
        """

        result = await self.judge_agent.run(evaluation_prompt)

        # Parse the LLM's evaluation (assuming it returns structured JSON)
        try:
            eval_data = json.loads(result.output) if isinstance(result.output, str) else result.output

            scores = [
                eval_data.get("relevance", 0) / 10,
                eval_data.get("ambiguity_detection", 0) / 10,
                eval_data.get("helpfulness", 0) / 10,
                eval_data.get("clarity", 0) / 10
            ]

            final_score = sum(scores) / len(scores)

            return {
                "score": final_score,
                "relevance": eval_data.get("relevance"),
                "ambiguity_detection": eval_data.get("ambiguity_detection"),
                "helpfulness": eval_data.get("helpfulness"),
                "clarity": eval_data.get("clarity"),
                "explanation": eval_data.get("explanation", "")
            }
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "score": None,
                "error": f"Failed to parse LLM evaluation: {e}"
            }


def create_clarification_dataset() -> Dataset:
    """Create evaluation dataset for clarification agent."""

    cases = [
        # Clear queries (should NOT need clarification)
        Case(
            name="bitcoin_price",
            inputs=ClarificationInput(query="What is the current Bitcoin price in USD?"),
            expected_output=ClarificationExpectedOutput(needs_clarification=False),
            evaluators=[BinaryAccuracyEvaluator()]
        ),
        Case(
            name="specific_code",
            inputs=ClarificationInput(query="Implement quicksort in Python with O(n log n) complexity"),
            expected_output=ClarificationExpectedOutput(needs_clarification=False),
            evaluators=[BinaryAccuracyEvaluator()]
        ),

        # Ambiguous queries (SHOULD need clarification)
        Case(
            name="broad_ai",
            inputs=ClarificationInput(query="What is AI?"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["audience_level", "scope_focus", "deliverable"],
                key_themes=["artificial intelligence", "specific", "aspect", "level"]
            ),
            evaluators=[
                BinaryAccuracyEvaluator(),
                DimensionCoverageEvaluator(),
                QuestionRelevanceEvaluator()
            ]
        ),
        Case(
            name="ambiguous_python",
            inputs=ClarificationInput(query="Tell me about Python"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["scope_focus"],
                key_themes=["programming", "language", "snake"]
            ),
            evaluators=[
                BinaryAccuracyEvaluator(),
                DimensionCoverageEvaluator(),
                QuestionRelevanceEvaluator()
            ]
        ),
        Case(
            name="vague_research",
            inputs=ClarificationInput(query="Research climate change"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["scope_focus", "deliverable", "source_quality"],
                key_themes=["aspect", "focus", "specific", "purpose"]
            ),
            evaluators=[
                BinaryAccuracyEvaluator(),
                DimensionCoverageEvaluator(),
                QuestionRelevanceEvaluator()
            ]
        ),

        # Edge cases
        Case(
            name="minimal_query",
            inputs=ClarificationInput(query="?"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                key_themes=["question", "help", "clarify"]
            ),
            evaluators=[
                BinaryAccuracyEvaluator(),
                QuestionRelevanceEvaluator()
            ]
        )
    ]

    return Dataset(
        name="clarification_agent_evaluation",
        cases=cases,
        description="Comprehensive evaluation dataset for ClarificationAgent"
    )


async def run_clarification_evaluation():
    """Run complete evaluation of clarification agent."""

    # Create agent
    agent = ClarificationAgent()

    # Create dataset
    dataset = create_clarification_dataset()

    # Define the task function that will be evaluated
    async def clarification_task(inputs: ClarificationInput) -> ClarifyWithUser:
        """Task function for evaluation."""
        async with httpx.AsyncClient() as http_client:
            state = ResearchState(
                request_id="eval-test",
                user_id="test-user",
                session_id="test-session",
                user_query=inputs.query,
                current_stage=ResearchStage.CLARIFICATION,
                metadata=ResearchMetadata()
            )
            deps = ResearchDependencies(
                http_client=http_client,
                api_keys=APIKeys(openai=os.getenv("OPENAI_API_KEY")),
                research_state=state
            )

            result = await agent.agent.run(inputs.query, deps=deps)
            return result.output

    # Run evaluation
    report = await dataset.evaluate(clarification_task)

    return report


def generate_evaluation_report(report: Report) -> str:
    """Generate human-readable evaluation report."""

    output = ["=" * 60]
    output.append("CLARIFICATION AGENT EVALUATION REPORT")
    output.append("=" * 60)

    # Overall metrics
    output.append("\nOVERALL METRICS:")
    output.append("-" * 40)

    total_cases = len(report.cases)
    binary_scores = [case.evaluations.get("BinaryAccuracyEvaluator", {}).get("score", 0)
                     for case in report.cases
                     if "BinaryAccuracyEvaluator" in case.evaluations]

    if binary_scores:
        accuracy = sum(binary_scores) / len(binary_scores)
        output.append(f"Binary Accuracy: {accuracy:.2%}")

    # Per-category breakdown
    output.append("\nPER-CATEGORY PERFORMANCE:")
    output.append("-" * 40)

    for case in report.cases:
        output.append(f"\n{case.name}:")
        for evaluator_name, evaluation in case.evaluations.items():
            if evaluation.get("score") is not None:
                output.append(f"  {evaluator_name}: {evaluation['score']:.2f}")
                if "explanation" in evaluation:
                    output.append(f"    {evaluation['explanation']}")

    # Common patterns
    output.append("\nCOMMON PATTERNS:")
    output.append("-" * 40)

    clarification_needed = sum(1 for case in report.cases
                              if case.output and case.output.needs_clarification)
    output.append(f"Cases needing clarification: {clarification_needed}/{total_cases}")

    # Dimension coverage analysis
    all_dimensions = []
    for case in report.cases:
        if case.output and case.output.needs_clarification:
            all_dimensions.extend(case.output.missing_dimensions)

    if all_dimensions:
        from collections import Counter
        dimension_counts = Counter(all_dimensions)
        output.append("\nMost common missing dimensions:")
        for dim, count in dimension_counts.most_common(5):
            output.append(f"  - {dim}: {count} times")

    return "\n".join(output)


if __name__ == "__main__":
    # Run evaluation
    report = asyncio.run(run_clarification_evaluation())
    print(generate_evaluation_report(report))
