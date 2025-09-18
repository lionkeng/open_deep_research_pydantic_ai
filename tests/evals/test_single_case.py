#!/usr/bin/env python
"""Test a single clarification case by name."""

import asyncio
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

from tests.evals.run_clarification_eval import MultiQuestionClarificationEvaluator


async def test_single_case(case_name: str, category: str = None):
    """Test a single case from the dataset."""

    # Load test cases from YAML
    yaml_path = Path(__file__).parent / "evaluation_datasets" / "clarification_dataset.yaml"

    if not yaml_path.exists():
        print(f"❌ Dataset file not found: {yaml_path}")
        return

    with open(yaml_path) as f:
        dataset = yaml.safe_load(f)

    # Find the specific case
    found_case = None
    for cat, cases in dataset["cases"].items():
        if category and cat != category:
            continue
        for case in cases:
            if case["name"] == case_name:
                found_case = case
                found_category = cat
                break
        if found_case:
            break

    if not found_case:
        print(f"❌ Case '{case_name}' not found in dataset")
        print("\nAvailable cases:")
        for cat, cases in dataset["cases"].items():
            print(f"\n{cat}:")
            for case in cases:
                print(f"  - {case['name']}")
        return

    print(f"\n🔍 Testing single case: {case_name} ({found_category})")
    print(f"   Query: {found_case['input']['query']}")
    print(
        f"   Expected: {'Clarification' if found_case['expected']['needs_clarification'] else 'No clarification'}"
    )

    # Create evaluator and run single test
    evaluator = MultiQuestionClarificationEvaluator()

    test_case = {
        "name": found_case["name"],
        "query": found_case["input"]["query"],
        "expected": found_case["expected"]["needs_clarification"],
        "category": found_category,
    }

    # Add multi-question expectations if present
    if "expected_questions" in found_case["expected"]:
        test_case["expected_question_count"] = found_case["expected"].get("expected_questions")
    if "question_types" in found_case["expected"]:
        test_case["expected_question_types"] = found_case["expected"].get("question_types")

    # Add context if present
    context = None
    if "context" in found_case["input"]:
        context = found_case["input"]["context"]

    # Run evaluation
    result = await evaluator.evaluate_query(
        test_case["query"],
        test_case["expected"],
        test_case.get("expected_question_count"),
        test_case.get("expected_question_types"),
        context=context,
    )

    # Display results
    print("\n📊 Results:")
    print(
        f"   {'✅' if result['correct'] else '❌'} Prediction: {'Clarification' if result['predicted'] else 'No clarification'}"
    )
    print(
        f"   Response time: {result['response_time']:.2f}s"
        if result["response_time"]
        else "   Response time: N/A"
    )

    if result.get("reasoning"):
        print("\n📝 Reasoning:")
        print(f"   {result['reasoning'][:200]}...")

    if result.get("question_analysis") and result["question_analysis"]:
        qa = result["question_analysis"]
        print("\n❓ Questions Analysis:")
        print(f"   Total questions: {qa.get('num_questions', 0)}")
        print(
            f"   Required: {qa.get('required_count', 0)}, Optional: {qa.get('optional_count', 0)}"
        )
        if qa.get("unique_types"):
            print(f"   Types: {', '.join(qa['unique_types'])}")

        if qa.get("questions"):
            print("\n   Questions:")
            for i, q in enumerate(qa["questions"][:3], 1):
                print(f"     {i}. {q['text'][:80]}...")
                print(f"        Type: {q['type']}, Required: {q['required']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test a single clarification case")
    parser.add_argument("case_name", help="Name of the test case to run")
    parser.add_argument("--category", help="Category to search in (optional)", default=None)

    args = parser.parse_args()

    asyncio.run(test_single_case(args.case_name, args.category))
