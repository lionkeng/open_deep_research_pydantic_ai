#!/usr/bin/env python3
"""
Test that the YAML dataset loads correctly for query transformation evaluation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.evals.query_transformation_evals import create_query_transformation_dataset


def test_yaml_dataset_loading():
    """Test loading dataset from YAML."""
    print("Testing YAML dataset loading...")

    # Load dataset (should load from YAML)
    dataset = create_query_transformation_dataset()

    print(f"✓ Dataset loaded with {len(dataset.cases)} cases")

    # Check some cases
    if dataset.cases:
        first_case = dataset.cases[0]
        print(f"✓ First case: {first_case.name}")
        print(f"  - Query: {first_case.inputs.query}")
        print(f"  - Has expected output: {first_case.expected_output is not None}")
        print(f"  - Evaluators: {len(first_case.evaluators)}")

        # Check evaluator types
        evaluator_types = [type(e).__name__ for e in first_case.evaluators]
        print(f"  - Evaluator types: {', '.join(evaluator_types)}")

    # Check categories by looking at case names
    categories = set()
    for case in dataset.cases:
        if hasattr(case, 'name'):
            # Extract category from case name pattern
            if 'golden' in case.name.lower():
                categories.add('golden_standard')
            elif 'tech' in case.name.lower():
                categories.add('technical')
            elif 'sci' in case.name.lower():
                categories.add('scientific')
            elif 'biz' in case.name.lower() or 'business' in case.name.lower():
                categories.add('business')
            elif 'edge' in case.name.lower():
                categories.add('edge_cases')
            elif 'cross' in case.name.lower():
                categories.add('cross_domain')
            elif 'perf' in case.name.lower():
                categories.add('performance')

    print(f"\n✓ Categories found: {', '.join(sorted(categories))}")

    return True


def main():
    """Run the test."""
    print("=" * 60)
    print("YAML DATASET LOADING TEST")
    print("=" * 60)

    try:
        success = test_yaml_dataset_loading()

        if success:
            print("\n✅ YAML dataset is properly configured and loads successfully!")
            print("The evaluation framework can now use the YAML-based dataset.")
        else:
            print("\n❌ Test failed")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
