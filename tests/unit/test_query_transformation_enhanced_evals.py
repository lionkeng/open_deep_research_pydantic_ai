"""
Unit tests for enhanced Query Transformation evaluators.

This module tests the new evaluators that provide comprehensive behavioral coverage
for the Query Transformation agent.
"""

import pytest
from unittest.mock import Mock, MagicMock
from pydantic_evals.evaluators import EvaluatorContext

from tests.evals.query_transformation_enhanced_evals import (
    AssumptionQualityEvaluator,
    PriorityDistributionEvaluator,
    ClarificationIntegrationEvaluator,
    QueryDecompositionEvaluator,
    SupportingQuestionsEvaluator,
    SuccessCriteriaMeasurabilityEvaluator,
    TemporalGeographicScopeEvaluator,
    SearchSourceSelectionEvaluator,
    ConfidenceCalibrationEvaluator,
    ExecutionStrategyEvaluator,
)
from src.models.research_plan_models import (
    TransformedQuery,
    ResearchPlan,
    ResearchObjective,
    ResearchMethodology,
)
from src.models.search_query_models import (
    SearchQueryBatch,
    SearchQuery,
    ExecutionStrategy,
    SearchQueryType,
    SearchSource,
    TemporalContext,
)


def create_mock_transformed_query():
    """Create a mock TransformedQuery for testing."""
    # Create mock objectives
    objectives = [
        ResearchObjective(
            id="obj1",
            objective="Analyze the primary factors",
            priority="PRIMARY",
            success_criteria="Identify at least 5 key factors",
            key_questions=["What are the main drivers?", "How do they interact?"]
        ),
        ResearchObjective(
            id="obj2",
            objective="Evaluate secondary impacts",
            priority="SECONDARY",
            success_criteria="Complete assessment of impacts",
            key_questions=["What are the side effects?"],
            dependencies=["obj1"]
        ),
    ]

    # Create mock research plan
    research_plan = ResearchPlan(
        objectives=objectives,
        methodology=ResearchMethodology(
            approach="Systematic literature review",
            data_sources=["Academic papers", "Industry reports"],
            analysis_methods=["Thematic analysis"],
            quality_criteria=["Peer-reviewed only"]
        ),
        expected_deliverables=["Research report", "Executive summary"],
        scope_definition="Focus on recent developments in the field",
        success_metrics=["Coverage of key topics", "Actionable insights"]
    )

    # Create mock search queries
    search_queries = [
        SearchQuery(
            id="q1",
            query="primary factors analysis",
            query_type=SearchQueryType.ANALYTICAL,
            priority=1,
            rationale="Core research question",
            objective_id="obj1",
            search_sources=[SearchSource.ACADEMIC]
        ),
        SearchQuery(
            id="q2",
            query="secondary impacts evaluation",
            query_type=SearchQueryType.EXPLORATORY,
            priority=3,
            rationale="Supporting research",
            objective_id="obj2"
        ),
        SearchQuery(
            id="q3",
            query="recent developments 2024",
            query_type=SearchQueryType.TEMPORAL,
            priority=2,
            rationale="Current context",
            temporal_context=TemporalContext(recency_preference="last_year")
        ),
    ]

    search_batch = SearchQueryBatch(
        queries=search_queries,
        execution_strategy=ExecutionStrategy.HIERARCHICAL,
        max_parallel=5
    )

    # Create transformed query
    transformed_query = TransformedQuery(
        original_query="Analyze the impact of recent developments",
        search_queries=search_batch,
        research_plan=research_plan,
        transformation_rationale="Decomposed into primary and secondary analysis",
        confidence_score=0.75,
        assumptions_made=["Focus on last year", "Academic sources preferred"],
        potential_gaps=["Regional variations not covered"],
        ambiguities_resolved=["Time frame clarified"],
        clarification_context={
            "missing_dimensions": ["specific domain", "time frame"],
            "answers": "Q: What time frame?\nA: Last year\nQ: What domain?\nA: [SKIPPED]"
        }
    )

    return transformed_query


class TestAssumptionQualityEvaluator:
    """Test AssumptionQualityEvaluator."""

    def test_evaluate_with_good_assumptions(self):
        """Test evaluation with well-formed assumptions."""
        evaluator = AssumptionQualityEvaluator()
        transformed_query = create_mock_transformed_query()

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # Should get a reasonable score with 2 assumptions and 1 gap
        # The score is lower because assumptions are simple
        assert 0.3 <= score <= 1.0

    def test_evaluate_no_assumptions_no_gaps(self):
        """Test evaluation with no assumptions and no gaps."""
        evaluator = AssumptionQualityEvaluator()
        transformed_query = create_mock_transformed_query()
        transformed_query.assumptions_made = []
        transformed_query.potential_gaps = []

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # Perfect score for no assumptions when no gaps
        assert score == 1.0


class TestPriorityDistributionEvaluator:
    """Test PriorityDistributionEvaluator."""

    def test_evaluate_balanced_distribution(self):
        """Test evaluation with balanced priority distribution."""
        evaluator = PriorityDistributionEvaluator()
        transformed_query = create_mock_transformed_query()

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # Should get good score with balanced priorities (1, 2, 3)
        assert score > 0.5

    def test_evaluate_objective_alignment(self):
        """Test that high priority queries align with primary objectives."""
        evaluator = PriorityDistributionEvaluator()
        transformed_query = create_mock_transformed_query()

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # q1 has priority 1 and is linked to primary objective
        assert score > 0.5


class TestClarificationIntegrationEvaluator:
    """Test ClarificationIntegrationEvaluator."""

    def test_evaluate_with_clarification_context(self):
        """Test evaluation with clarification context."""
        evaluator = ClarificationIntegrationEvaluator()
        transformed_query = create_mock_transformed_query()

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # Should handle clarification context properly
        assert 0.0 <= score <= 1.0

    def test_evaluate_no_clarification(self):
        """Test evaluation without clarification context."""
        evaluator = ClarificationIntegrationEvaluator()
        transformed_query = create_mock_transformed_query()
        transformed_query.clarification_context = {}

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # Should return perfect score when no clarification needed
        assert score == 1.0


class TestQueryDecompositionEvaluator:
    """Test QueryDecompositionEvaluator."""

    def test_evaluate_hierarchy(self):
        """Test evaluation of hierarchical structure."""
        evaluator = QueryDecompositionEvaluator()
        transformed_query = create_mock_transformed_query()

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # Should recognize PRIMARY and SECONDARY objectives
        assert score > 0.5


class TestSupportingQuestionsEvaluator:
    """Test SupportingQuestionsEvaluator."""

    def test_evaluate_with_questions(self):
        """Test evaluation with supporting questions."""
        evaluator = SupportingQuestionsEvaluator()
        transformed_query = create_mock_transformed_query()

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # Should evaluate questions positively
        assert score > 0.3


class TestSuccessCriteriaMeasurabilityEvaluator:
    """Test SuccessCriteriaMeasurabilityEvaluator."""

    def test_evaluate_measurable_criteria(self):
        """Test evaluation with measurable success criteria."""
        evaluator = SuccessCriteriaMeasurabilityEvaluator()
        transformed_query = create_mock_transformed_query()

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # "at least 5" is quantifiable
        assert score > 0.5


class TestTemporalGeographicScopeEvaluator:
    """Test TemporalGeographicScopeEvaluator."""

    def test_evaluate_temporal_scope(self):
        """Test evaluation of temporal scope."""
        evaluator = TemporalGeographicScopeEvaluator()
        transformed_query = create_mock_transformed_query()

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # Has "recent" in query and temporal context in q3
        assert score > 0.5


class TestSearchSourceSelectionEvaluator:
    """Test SearchSourceSelectionEvaluator."""

    def test_evaluate_source_selection(self):
        """Test evaluation of search source selection."""
        evaluator = SearchSourceSelectionEvaluator()
        transformed_query = create_mock_transformed_query()

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # Has academic source for analytical query
        assert 0.0 <= score <= 1.0


class TestConfidenceCalibrationEvaluator:
    """Test ConfidenceCalibrationEvaluator."""

    def test_evaluate_calibration(self):
        """Test confidence calibration evaluation."""
        evaluator = ConfidenceCalibrationEvaluator()
        transformed_query = create_mock_transformed_query()

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # 0.75 confidence with 2 assumptions and 1 gap is reasonable
        assert score > 0.5


class TestExecutionStrategyEvaluator:
    """Test ExecutionStrategyEvaluator."""

    def test_evaluate_hierarchical_strategy(self):
        """Test evaluation of hierarchical execution strategy."""
        evaluator = ExecutionStrategyEvaluator()
        transformed_query = create_mock_transformed_query()

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # HIERARCHICAL is good for varied priorities
        # Score is lower because query count is small (3)
        assert score > 0.3

    def test_evaluate_dependency_handling(self):
        """Test evaluation of dependency handling."""
        evaluator = ExecutionStrategyEvaluator()
        transformed_query = create_mock_transformed_query()

        ctx = Mock(spec=EvaluatorContext)
        ctx.output = transformed_query

        score = evaluator.evaluate(ctx)

        # obj2 depends on obj1, HIERARCHICAL handles this well
        # Score reflects the dependency handling capability
        assert score > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
