#!/usr/bin/env python3
"""
Test the query transformation evaluation framework with mock data.
"""

import sys
from pathlib import Path
import uuid

# Change to project root directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.evals.query_transformation_evals import (
    QueryTransformationInput,
    QueryTransformationExpectedOutput,
    SearchQueryRelevanceEvaluator,
    ObjectiveCoverageEvaluator,
    TransformationAccuracyEvaluator
)
from src.models.research_plan_models import (
    ResearchObjective,
    ResearchPlan,
    ResearchMethodology,
    TransformedQuery,
)
from src.models.search_query_models import (
    SearchQuery,
    SearchQueryBatch,
    SearchQueryType,
)
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import EvaluatorContext


def create_mock_transformed_query() -> TransformedQuery:
    """Create a mock TransformedQuery for testing."""

    # Create objectives with IDs
    obj1_id = str(uuid.uuid4())
    obj2_id = str(uuid.uuid4())

    objectives = [
        ResearchObjective(
            id=obj1_id,
            objective="Analyze machine learning applications in healthcare diagnostics",
            priority="PRIMARY",
            success_criteria="Identify key ML algorithms and their accuracy rates"
        ),
        ResearchObjective(
            id=obj2_id,
            objective="Evaluate regulatory challenges for AI in medical devices",
            priority="SECONDARY",
            success_criteria="Document FDA approval processes and requirements"
        )
    ]

    # Create search queries linked to objectives
    queries = [
        SearchQuery(
            id=str(uuid.uuid4()),
            query="machine learning healthcare diagnostic algorithms",
            query_type=SearchQueryType.ANALYTICAL,
            priority=5,
            max_results=10,
            rationale="Core ML algorithms in healthcare",
            objective_id=obj1_id
        ),
        SearchQuery(
            id=str(uuid.uuid4()),
            query="FDA approval AI medical devices 2024",
            query_type=SearchQueryType.FACTUAL,
            priority=4,
            max_results=10,
            rationale="Regulatory landscape",
            objective_id=obj2_id
        ),
        SearchQuery(
            id=str(uuid.uuid4()),
            query="healthcare ML accuracy comparison studies",
            query_type=SearchQueryType.COMPARATIVE,
            priority=3,
            max_results=10,
            rationale="Performance evaluation",
            objective_id=obj1_id
        )
    ]

    # Create methodology
    methodology = ResearchMethodology(
        approach="Systematic literature review and regulatory analysis",
        data_sources=["PubMed", "IEEE", "FDA guidelines", "Industry reports"],
        analysis_methods=["Comparative analysis", "Meta-analysis", "Regulatory mapping"]
    )

    # Create research plan
    plan = ResearchPlan(
        objectives=objectives,
        methodology=methodology,
        expected_deliverables=[
            "Comprehensive report on ML in healthcare",
            "Regulatory compliance guide",
            "Algorithm performance comparison table"
        ]
    )

    return TransformedQuery(
        original_query="How does machine learning work in healthcare?",
        search_queries=SearchQueryBatch(queries=queries),
        research_plan=plan,
        confidence_score=0.85
    )


def test_evaluators():
    """Test individual evaluators."""
    print("Testing individual evaluators...")

    # Create mock data
    inputs = QueryTransformationInput(
        query="How does machine learning work in healthcare?",
        complexity="medium",
        domain="technical"
    )

    expected = QueryTransformationExpectedOutput(
        min_search_queries=3,
        max_search_queries=8,
        expected_search_themes=["machine learning", "healthcare", "algorithms"]
    )

    output = create_mock_transformed_query()

    # Create mock context
    ctx = EvaluatorContext(
        name="test_case",
        inputs=inputs,
        metadata=None,
        expected_output=expected,
        output=output,
        duration=1.0,
        _span_tree=None,
        attributes={},
        metrics={}
    )

    # Test evaluators
    evaluators = [
        ("SearchQueryRelevanceEvaluator", SearchQueryRelevanceEvaluator()),
        ("ObjectiveCoverageEvaluator", ObjectiveCoverageEvaluator()),
        ("TransformationAccuracyEvaluator", TransformationAccuracyEvaluator())
    ]

    results = {}
    for name, evaluator in evaluators:
        try:
            score = evaluator.evaluate(ctx)
            results[name] = score
            print(f"✓ {name}: {score:.3f}")
        except Exception as e:
            print(f"✗ {name}: ERROR - {e}")
            results[name] = None

    return results


def test_dataset_structure():
    """Test dataset creation."""
    print("\nTesting dataset structure...")

    # Create a simple case
    case = Case(
        name="test_case",
        inputs=QueryTransformationInput(
            query="Test query",
            complexity="simple"
        ),
        expected_output=QueryTransformationExpectedOutput(
            min_search_queries=2,
            max_search_queries=5
        ),
        evaluators=[
            SearchQueryRelevanceEvaluator()
        ]
    )

    # Create dataset
    dataset = Dataset(cases=[case])

    print(f"✓ Dataset created with {len(dataset.cases)} case(s)")
    return True


def main():
    """Run framework tests."""
    print("=" * 60)
    print("QUERY TRANSFORMATION EVALUATION FRAMEWORK TEST")
    print("=" * 60)

    try:
        # Test evaluators
        eval_results = test_evaluators()

        # Test dataset structure
        dataset_ok = test_dataset_structure()

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        success_count = sum(1 for r in eval_results.values() if r is not None)
        total_count = len(eval_results)

        print(f"Evaluators tested: {success_count}/{total_count}")
        print(f"Dataset structure: {'✓ OK' if dataset_ok else '✗ FAILED'}")

        if success_count == total_count and dataset_ok:
            print("\n🎉 All tests passed! Framework is ready.")
            return True
        else:
            print("\n❌ Some tests failed.")
            return False

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
