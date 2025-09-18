#!/usr/bin/env python3
"""Test dataset loading for evaluation framework."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from tests.evals.run_clarification_eval import MultiQuestionClarificationEvaluator  # noqa: E402
from tests.evals.run_query_transformation_eval import QueryTransformationEvaluator  # noqa: E402


def test_clarification_dataset_loading():
    """Test loading clarification dataset."""
    print("Testing clarification dataset loading...")

    evaluator = MultiQuestionClarificationEvaluator()

    # Try loading with a valid category
    dataset = evaluator.load_dataset_from_yaml(categories=["multi_question_cases"])

    assert hasattr(dataset, "samples"), "Dataset should have samples attribute"
    assert len(dataset.samples) > 0, "Should load at least one test case"

    sample = dataset.samples[0]
    assert hasattr(sample, "inputs"), "Sample should have inputs"
    assert hasattr(sample, "expected"), "Sample should have expected"

    # Check we can access the query
    query = (
        sample.inputs.get("query")
        if hasattr(sample.inputs, "get")
        else getattr(sample.inputs, "query", None)
    )
    assert query is not None, "Should be able to access query from inputs"

    print(f"✓ Loaded {len(dataset.samples)} clarification test cases")
    print(f"✓ First test case query: {query[:50]}...")

    return True


def test_query_transformation_dataset_loading():
    """Test loading query transformation dataset."""
    print("\nTesting query transformation dataset loading...")

    evaluator = QueryTransformationEvaluator()

    # Try loading with a valid category
    dataset = evaluator.load_dataset_from_yaml(categories=["golden_standard_cases"])

    assert hasattr(dataset, "cases"), "Dataset should have cases attribute"
    assert len(dataset.cases) > 0, "Should load at least one test case"

    case = dataset.cases[0]
    assert hasattr(case, "inputs"), "Case should have inputs"

    # Check for expected_output attribute (pydantic_evals uses expected_output)
    assert hasattr(case, "expected_output"), "Case should have expected_output"

    # Check we can access the query
    query = (
        case.inputs.get("query")
        if hasattr(case.inputs, "get")
        else getattr(case.inputs, "query", None)
    )
    assert query is not None, "Should be able to access query from inputs"

    print(f"✓ Loaded {len(dataset.cases)} query transformation test cases")
    print(f"✓ First test case query: {query[:50]}...")

    return True


def main():
    """Run dataset loading tests."""
    print("=" * 60)
    print("Dataset Loading Tests")
    print("=" * 60)

    try:
        test_clarification_dataset_loading()
        test_query_transformation_dataset_loading()

        print("\n✅ All dataset loading tests passed!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
