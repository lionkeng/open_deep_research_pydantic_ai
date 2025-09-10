"""
Query Transformation Agent Evaluation Framework using Pydantic Evals.

This module provides comprehensive evaluation capabilities for the query transformation agent,
following pydantic-ai evaluation patterns with custom evaluators, metrics, and LLM-as-judge approaches.
"""

import asyncio
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import httpx

from pydantic import BaseModel, Field, SecretStr
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport
from pydantic_ai import Agent

from src.agents.query_transformation import QueryTransformationAgent
from src.agents.base import ResearchDependencies
from src.models.metadata import ResearchMetadata
from src.models.core import ResearchState, ResearchStage
from src.models.research_plan_models import TransformedQuery
from src.models.api_models import APIKeys


class QueryTransformationInput(BaseModel):
    """Input model for query transformation evaluation."""
    query: str = Field(description="User query to transform")
    context: Optional[str] = Field(
        default=None,
        description="Optional context for the query"
    )
    complexity: str = Field(
        default="medium",
        description="Expected complexity level: simple, medium, complex"
    )
    domain: Optional[str] = Field(
        default=None,
        description="Domain classification (technical, scientific, business, etc.)"
    )


class QueryTransformationExpectedOutput(BaseModel):
    """Expected output for query transformation evaluation."""
    min_search_queries: Optional[int] = Field(
        default=None,
        description="Minimum number of search queries expected"
    )
    max_search_queries: Optional[int] = Field(
        default=None,
        description="Maximum number of search queries expected"
    )
    min_objectives: Optional[int] = Field(
        default=None,
        description="Minimum number of research objectives expected"
    )
    max_objectives: Optional[int] = Field(
        default=None,
        description="Maximum number of research objectives expected"
    )
    expected_search_themes: Optional[List[str]] = Field(
        default=None,
        description="Expected themes in search queries"
    )
    expected_objective_themes: Optional[List[str]] = Field(
        default=None,
        description="Expected themes in research objectives"
    )
    query_types_expected: Optional[List[str]] = Field(
        default=None,
        description="Expected search query types (factual, analytical, etc.)"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Expected confidence score for transformation (0.0-1.0)"
    )
    max_response_time: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Maximum acceptable response time in seconds"
    )
    transformation_quality: Optional[str] = Field(
        default=None,
        description="Expected transformation quality: excellent, good, fair, poor"
    )


class SearchQueryRelevanceEvaluator(Evaluator):
    """Evaluates relevance of generated search queries to original query."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate search query relevance."""
        output = ctx.output
        expected = ctx.expected_output or QueryTransformationExpectedOutput()

        search_queries = output.search_queries.queries
        original_query = output.original_query

        # Calculate relevance scores
        relevance_scores = []
        for query in search_queries:
            score = self._calculate_relevance_score(original_query, query.query)
            relevance_scores.append(score)

        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

        # Check expected themes coverage
        theme_coverage = 1.0
        if expected.expected_search_themes:
            covered_themes = 0
            for theme in expected.expected_search_themes:
                if any(theme.lower() in q.query.lower() for q in search_queries):
                    covered_themes += 1
            theme_coverage = covered_themes / len(expected.expected_search_themes)

        final_score = (avg_relevance + theme_coverage) / 2

        return final_score

    def _calculate_relevance_score(self, original: str, transformed: str) -> float:
        """Calculate relevance score between original and transformed query."""
        original_words = set(original.lower().split())
        transformed_words = set(transformed.lower().split())

        if not original_words:
            return 0.0

        overlap = original_words.intersection(transformed_words)
        return len(overlap) / len(original_words)


class ObjectiveCoverageEvaluator(Evaluator):
    """Evaluates coverage and quality of research objectives."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate objective coverage."""
        output = ctx.output
        expected = ctx.expected_output or QueryTransformationExpectedOutput()

        objectives = output.research_plan.objectives

        # Count validation
        count_score = 1.0
        if expected.min_objectives and len(objectives) < expected.min_objectives:
            count_score = len(objectives) / expected.min_objectives
        elif expected.max_objectives and len(objectives) > expected.max_objectives:
            count_score = expected.max_objectives / len(objectives)

        # Quality metrics
        quality_scores = {
            "specificity": self._evaluate_specificity(objectives),
            "diversity": self._evaluate_diversity(objectives),
            "alignment": self._evaluate_alignment(objectives, output.original_query)
        }

        # Theme coverage
        theme_coverage = 1.0
        if expected.expected_objective_themes:
            covered_themes = 0
            for theme in expected.expected_objective_themes:
                if any(theme.lower() in obj.objective.lower() for obj in objectives):
                    covered_themes += 1
            theme_coverage = covered_themes / len(expected.expected_objective_themes)

        avg_quality = sum(quality_scores.values()) / len(quality_scores)
        final_score = (count_score + avg_quality + theme_coverage) / 3

        return final_score

    def _evaluate_specificity(self, objectives) -> float:
        """Evaluate specificity of objectives."""
        if not objectives:
            return 0.0

        specificity_scores = []
        action_verbs = {"analyze", "evaluate", "compare", "investigate", "examine", "assess"}

        for obj in objectives:
            words = obj.objective.lower().split()
            has_action_verb = any(verb in words for verb in action_verbs)
            word_count = len(words)

            score = 0.5 if has_action_verb else 0.0
            if 5 <= word_count <= 20:
                score += 0.5

            specificity_scores.append(score)

        return sum(specificity_scores) / len(specificity_scores)

    def _evaluate_diversity(self, objectives) -> float:
        """Evaluate diversity of objectives."""
        if len(objectives) <= 1:
            return 1.0

        first_words = set()
        for obj in objectives:
            words = obj.objective.lower().split()
            if words:
                first_words.add(words[0])

        return len(first_words) / len(objectives)

    def _evaluate_alignment(self, objectives, original_query: str) -> float:
        """Evaluate alignment with original query."""
        if not objectives:
            return 0.0

        query_words = set(original_query.lower().split())
        alignment_scores = []

        for obj in objectives:
            obj_words = set(obj.objective.lower().split())
            overlap = query_words.intersection(obj_words)
            alignment_scores.append(len(overlap) / len(query_words) if query_words else 0.0)

        return max(alignment_scores) if alignment_scores else 0.0


class PlanCoherenceEvaluator(Evaluator):
    """Evaluates research plan structure and coherence."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate plan coherence."""
        output = ctx.output
        expected = ctx.expected_output or QueryTransformationExpectedOutput()

        plan = output.research_plan

        coherence_scores = {
            "methodology_quality": self._evaluate_methodology(plan.methodology),
            "deliverables_clarity": self._evaluate_deliverables(plan),
            "success_metrics": self._evaluate_success_metrics(plan)
        }

        avg_score = sum(coherence_scores.values()) / len(coherence_scores)

        return avg_score

    def _evaluate_methodology(self, methodology) -> float:
        """Evaluate methodology quality."""
        if not methodology:
            return 0.3

        score = 0.0
        if methodology.approach and len(methodology.approach) > 10:
            score += 0.25
        if methodology.data_sources and len(methodology.data_sources) >= 2:
            score += 0.25
        if methodology.analysis_methods and len(methodology.analysis_methods) >= 1:
            score += 0.25
        if methodology.quality_criteria and len(methodology.quality_criteria) >= 1:
            score += 0.25

        return score

    def _evaluate_deliverables(self, plan) -> float:
        """Evaluate deliverables clarity."""
        if not plan.expected_deliverables:
            return 0.3
        if len(plan.expected_deliverables) < 2:
            return 0.6
        elif len(plan.expected_deliverables) <= 5:
            return 1.0
        else:
            return 0.8

    def _evaluate_success_metrics(self, plan) -> float:
        """Evaluate success metrics definition."""
        if not plan.success_metrics:
            return 0.4
        return 1.0 if len(plan.success_metrics) >= 2 else 0.7


class QueryDiversityEvaluator(Evaluator):
    """Evaluates diversity and coverage of search queries."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate query diversity."""
        output = ctx.output
        expected = ctx.expected_output or QueryTransformationExpectedOutput()

        search_queries = output.search_queries.queries

        diversity_metrics = {
            "lexical_diversity": self._evaluate_lexical_diversity(search_queries),
            "type_diversity": self._evaluate_type_diversity(search_queries),
            "length_variation": self._evaluate_length_variation(search_queries)
        }

        avg_score = sum(diversity_metrics.values()) / len(diversity_metrics)

        return avg_score

    def _evaluate_lexical_diversity(self, queries) -> float:
        """Evaluate lexical diversity."""
        if len(queries) <= 1:
            return 1.0

        all_words = []
        for query in queries:
            all_words.extend(query.query.lower().split())

        unique_words = set(all_words)
        if not all_words:
            return 0.0

        diversity_ratio = len(unique_words) / len(all_words)
        expected_ratio = max(0.3, 1.0 - (0.1 * len(queries)))

        return min(1.0, diversity_ratio / expected_ratio)

    def _evaluate_type_diversity(self, queries) -> float:
        """Evaluate diversity of query types."""
        if not queries:
            return 0.0

        query_types = set(q.query_type for q in queries)

        if len(queries) <= 2:
            expected_types = 1
        elif len(queries) <= 5:
            expected_types = 2
        else:
            expected_types = 3

        return min(1.0, len(query_types) / expected_types)

    def _evaluate_length_variation(self, queries) -> float:
        """Evaluate variation in query lengths."""
        if len(queries) <= 1:
            return 1.0

        lengths = [len(q.query.split()) for q in queries]
        if not lengths:
            return 0.0

        avg_length = sum(lengths) / len(lengths)
        if avg_length == 0:
            return 0.0

        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        cv = std_dev / avg_length

        if cv < 0.1:
            return 0.5
        elif cv < 0.2:
            return 0.8
        elif cv <= 0.5:
            return 1.0
        else:
            return 0.7

    def _count_unique_terms(self, queries) -> int:
        """Count unique terms across all queries."""
        all_terms = set()
        for query in queries:
            all_terms.update(query.query.lower().split())
        return len(all_terms)


class TransformationAccuracyEvaluator(Evaluator):
    """Evaluates overall transformation accuracy and completeness."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate transformation accuracy."""
        output = ctx.output
        expected = ctx.expected_output or QueryTransformationExpectedOutput()

        accuracy_metrics = {
            "query_preservation": self._evaluate_query_preservation(output),
            "completeness": self._evaluate_completeness(output),
            "consistency": self._evaluate_consistency(output)
        }

        avg_score = sum(accuracy_metrics.values()) / len(accuracy_metrics)

        return avg_score

    def _evaluate_query_preservation(self, transformed: TransformedQuery) -> float:
        """Evaluate if original query intent is preserved."""
        original_words = set(transformed.original_query.lower().split())

        preserved_in_objectives = any(
            len(original_words.intersection(set(obj.objective.lower().split()))) > 0
            for obj in transformed.research_plan.objectives
        )

        preserved_in_queries = any(
            len(original_words.intersection(set(q.query.lower().split()))) > 0
            for q in transformed.search_queries.queries
        )

        preservation_score = 0.0
        if preserved_in_objectives:
            preservation_score += 0.5
        if preserved_in_queries:
            preservation_score += 0.5

        return preservation_score

    def _evaluate_completeness(self, transformed: TransformedQuery) -> float:
        """Evaluate transformation completeness."""
        completeness_checks = {
            "has_objectives": len(transformed.research_plan.objectives) > 0,
            "has_search_queries": len(transformed.search_queries.queries) > 0,
            "has_plan": transformed.research_plan is not None,
            "has_methodology": transformed.research_plan.methodology is not None,
            "has_deliverables": len(transformed.research_plan.expected_deliverables) > 0,
        }

        return sum(completeness_checks.values()) / len(completeness_checks)

    def _evaluate_consistency(self, transformed: TransformedQuery) -> float:
        """Evaluate internal consistency."""
        obj_count = len(transformed.research_plan.objectives)
        query_count = len(transformed.search_queries.queries)

        consistency_score = 1.0

        if query_count < obj_count:
            consistency_score *= 0.8

        if obj_count > 0:
            queries_per_obj = query_count / obj_count
            if queries_per_obj < 1:
                consistency_score *= 0.7
            elif queries_per_obj > 5:
                consistency_score *= 0.9

        return consistency_score


class LLMJudgeEvaluator(Evaluator):
    """Uses an LLM to judge transformation quality."""

    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model
        self.judge_agent = Agent(
            model=model,
            system_prompt="""You are an expert evaluator of query transformations.
            Evaluate the quality based on:
            1. Relevance to original query
            2. Completeness of research plan
            3. Quality of search queries
            4. Overall coherence and usefulness"""
        )

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Synchronous evaluate method required by Evaluator."""
        # For now, return a placeholder. In practice, you'd call evaluate_async
        return 0.8

    async def evaluate_async(
        self,
        query: str,
        output: TransformedQuery,
        expected: Optional[QueryTransformationExpectedOutput] = None
    ) -> Dict[str, Any]:
        """Use LLM to judge transformation quality."""

        objectives_text = "\n".join([f"- {obj.objective}" for obj in output.research_plan.objectives])
        queries_text = "\n".join([f"- {q.query} (Type: {q.query_type})" for q in output.search_queries.queries])

        evaluation_prompt = f"""
        Original Query: {query}

        Transformation Output:
        Objectives:
{objectives_text}

        Search Queries:
{queries_text}

        Research Plan: {output.research_plan.methodology.approach if output.research_plan.methodology else 'No methodology'}

        Please evaluate this transformation on a scale of 0-10 for:
        1. Relevance (0-10): How well does it address the original query?
        2. Completeness (0-10): Is the transformation comprehensive?
        3. Quality (0-10): Are the objectives and queries well-formed?
        4. Coherence (0-10): Is the overall plan coherent?

        Provide your evaluation as a JSON object with these scores and brief explanation.
        """

        result = await self.judge_agent.run(evaluation_prompt)

        try:
            eval_data = json.loads(result.output) if isinstance(result.output, str) else result.output

            scores = [
                eval_data.get("relevance", 0) / 10,
                eval_data.get("completeness", 0) / 10,
                eval_data.get("quality", 0) / 10,
                eval_data.get("coherence", 0) / 10
            ]

            final_score = sum(scores) / len(scores)

            return {
                "score": final_score,
                "relevance": eval_data.get("relevance"),
                "completeness": eval_data.get("completeness"),
                "quality": eval_data.get("quality"),
                "coherence": eval_data.get("coherence"),
                "explanation": eval_data.get("explanation", "")
            }
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "score": None,
                "error": f"Failed to parse LLM evaluation: {e}"
            }


def create_query_transformation_dataset() -> Dataset:
    """Create comprehensive evaluation dataset for query transformation agent from YAML."""
    from pathlib import Path

    # Try to load from YAML file if it exists
    yaml_path = Path(__file__).parent / "evaluation_datasets" / "query_transformation_dataset.yaml"
    if yaml_path.exists():
        try:
            from tests.evals.query_transformation_dataset_loader import load_dataset_from_yaml
            return load_dataset_from_yaml(yaml_path)
        except ImportError:
            pass  # Fall back to hardcoded dataset

    # Fallback: Golden Standard Cases - Clear expected outcomes
    golden_cases = [
        # Should produce comprehensive transformations
        Case(
            name="golden_complex_ai",
            inputs=QueryTransformationInput(
                query="How does machine learning work in healthcare?",
                complexity="medium",
                domain="technical"
            ),
            expected_output=QueryTransformationExpectedOutput(
                min_search_queries=5,
                max_search_queries=12,
                min_objectives=2,
                max_objectives=4,
                expected_search_themes=["machine learning", "healthcare", "algorithms", "medical"],
                expected_objective_themes=["analyze", "understand", "applications"],
                query_types_expected=["factual", "analytical", "exploratory"]
            ),
            evaluators=[
                SearchQueryRelevanceEvaluator(),
                ObjectiveCoverageEvaluator(),
                PlanCoherenceEvaluator(),
                TransformationAccuracyEvaluator()
            ]
        ),
        Case(
            name="golden_specific_technical",
            inputs=QueryTransformationInput(
                query="Compare PostgreSQL vs MySQL performance for e-commerce",
                complexity="medium",
                domain="technical"
            ),
            expected_output=QueryTransformationExpectedOutput(
                min_search_queries=4,
                max_search_queries=8,
                min_objectives=2,
                max_objectives=3,
                expected_search_themes=["PostgreSQL", "MySQL", "performance", "e-commerce"],
                expected_objective_themes=["compare", "analyze", "evaluate"]
            ),
            evaluators=[
                SearchQueryRelevanceEvaluator(),
                ObjectiveCoverageEvaluator(),
                QueryDiversityEvaluator()
            ]
        ),
        Case(
            name="golden_broad_research",
            inputs=QueryTransformationInput(
                query="Research climate change impacts",
                complexity="complex",
                domain="scientific"
            ),
            expected_output=QueryTransformationExpectedOutput(
                min_search_queries=6,
                max_search_queries=15,
                min_objectives=3,
                max_objectives=5,
                expected_search_themes=["climate change", "impacts", "environment", "research"],
                expected_objective_themes=["investigate", "analyze", "assess"]
            ),
            evaluators=[
                SearchQueryRelevanceEvaluator(),
                ObjectiveCoverageEvaluator(),
                PlanCoherenceEvaluator(),
                QueryDiversityEvaluator(),
                TransformationAccuracyEvaluator()
            ]
        )
    ]

    # Domain-specific cases
    technical_cases = [
        Case(
            name="tech_microservices",
            inputs=QueryTransformationInput(
                query="Best practices for microservices architecture",
                complexity="medium",
                domain="technical"
            ),
            expected_output=QueryTransformationExpectedOutput(
                min_search_queries=4,
                max_search_queries=10,
                expected_search_themes=["microservices", "architecture", "best practices"]
            ),
            evaluators=[SearchQueryRelevanceEvaluator(), ObjectiveCoverageEvaluator()]
        ),
        Case(
            name="tech_optimization",
            inputs=QueryTransformationInput(
                query="How to optimize database queries",
                complexity="simple",
                domain="technical"
            ),
            expected_output=QueryTransformationExpectedOutput(
                min_search_queries=3,
                max_search_queries=6,
                expected_search_themes=["database", "optimization", "queries"]
            ),
            evaluators=[SearchQueryRelevanceEvaluator(), QueryDiversityEvaluator()]
        )
    ]

    # Business cases
    business_cases = [
        Case(
            name="biz_market_analysis",
            inputs=QueryTransformationInput(
                query="Analyze market trends for electric vehicles",
                complexity="medium",
                domain="business"
            ),
            expected_output=QueryTransformationExpectedOutput(
                min_search_queries=5,
                max_search_queries=10,
                expected_search_themes=["market", "trends", "electric vehicles"]
            ),
            evaluators=[SearchQueryRelevanceEvaluator(), ObjectiveCoverageEvaluator()]
        )
    ]

    # Edge cases
    edge_cases = [
        Case(
            name="edge_minimal",
            inputs=QueryTransformationInput(query="AI"),
            expected_output=QueryTransformationExpectedOutput(
                min_search_queries=3,
                max_search_queries=8
            ),
            evaluators=[SearchQueryRelevanceEvaluator(), TransformationAccuracyEvaluator()]
        ),
        Case(
            name="edge_very_specific",
            inputs=QueryTransformationInput(
                query="What is the molecular structure of caffeine C8H10N4O2?",
                complexity="simple"
            ),
            expected_output=QueryTransformationExpectedOutput(
                min_search_queries=2,
                max_search_queries=5
            ),
            evaluators=[SearchQueryRelevanceEvaluator(), TransformationAccuracyEvaluator()]
        )
    ]

    all_cases = golden_cases + technical_cases + business_cases + edge_cases

    return Dataset(
        cases=all_cases
    )


async def run_query_transformation_evaluation():
    """Run complete evaluation of query transformation agent."""

    # Create agent
    agent = QueryTransformationAgent()

    # Create dataset
    dataset = create_query_transformation_dataset()

    # Define the task function that will be evaluated
    async def transformation_task(inputs: QueryTransformationInput) -> TransformedQuery:
        """Task function for evaluation."""
        async with httpx.AsyncClient() as http_client:
            state = ResearchState(
                request_id="eval-test",
                user_id="test-user",
                session_id="test-session",
                user_query=inputs.query,
                current_stage=ResearchStage.RESEARCH_EXECUTION,
                metadata=ResearchMetadata()
            )
            deps = ResearchDependencies(
                http_client=http_client,
                api_keys=APIKeys(openai=SecretStr(openai_key) if (openai_key := os.getenv("OPENAI_API_KEY")) else None),
                research_state=state
            )

            result = await agent.agent.run(inputs.query, deps=deps)
            return result.output

    # Run evaluation
    report = await dataset.evaluate(transformation_task)

    return report


def generate_evaluation_report(report: EvaluationReport) -> str:
    """Generate human-readable evaluation report."""

    output = ["=" * 60]
    output.append("QUERY TRANSFORMATION AGENT EVALUATION REPORT")
    output.append("=" * 60)

    # Overall metrics
    output.append("\nOVERALL METRICS:")
    output.append("-" * 40)

    total_cases = len(report.cases)
    all_scores = []

    for case in report.cases:
        case_scores = [eval_result.get("score", 0)
                      for eval_result in case.evaluations.values()
                      if eval_result.get("score") is not None]
        if case_scores:
            all_scores.extend(case_scores)

    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        output.append(f"Average Score: {avg_score:.2%}")

    # Per-case breakdown
    output.append("\nPER-CASE PERFORMANCE:")
    output.append("-" * 40)

    for case in report.cases:
        output.append(f"\n{case.name}:")
        for evaluator_name, evaluation in case.evaluations.items():
            if evaluation.get("score") is not None:
                output.append(f"  {evaluator_name}: {evaluation['score']:.2f}")

    # Transformation patterns
    output.append("\nTRANSFORMATION PATTERNS:")
    output.append("-" * 40)

    total_objectives = 0
    total_queries = 0
    for case in report.cases:
        if case.output:
            total_objectives += len(case.output.research_plan.objectives)
            total_queries += len(case.output.search_queries.queries)

    if total_cases > 0:
        output.append(f"Average objectives per case: {total_objectives / total_cases:.1f}")
        output.append(f"Average queries per case: {total_queries / total_cases:.1f}")

    return "\n".join(output)


if __name__ == "__main__":
    # Run evaluation
    report = asyncio.run(run_query_transformation_evaluation())
    print(generate_evaluation_report(report))
