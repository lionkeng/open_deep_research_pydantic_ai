#!/usr/bin/env python
"""
Simple test runner for clarification agent evaluation.

This script demonstrates how to run the evaluation suite and generate reports.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import json
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Suppress logfire prompts
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

import httpx
from src.agents.clarification import ClarificationAgent, ClarifyWithUser
from src.agents.base import ResearchDependencies
from src.models.api_models import APIKeys, ResearchMetadata
from src.models.core import ResearchState


class SimpleClarificationEvaluator:
    """Simple evaluator for clarification agent without external dependencies."""

    def __init__(self):
        self.agent = ClarificationAgent()
        self.results = []

    async def evaluate_query(self, query: str, expected_clarification: bool) -> Dict[str, Any]:
        """Evaluate a single query."""
        async with httpx.AsyncClient() as http_client:
            state = ResearchState(
                request_id=f"eval-{abs(hash(query))}",
                user_query=query
            )
            deps = ResearchDependencies(
                http_client=http_client,
                api_keys=APIKeys(
                    openai=os.getenv("OPENAI_API_KEY"),
                    anthropic=os.getenv("ANTHROPIC_API_KEY")
                ),
                research_state=state
            )

            try:
                result = await self.agent.agent.run(query, deps=deps)
                output: ClarifyWithUser = result.data

                # Evaluate
                correct = output.need_clarification == expected_clarification

                evaluation = {
                    "query": query,
                    "expected": expected_clarification,
                    "predicted": output.need_clarification,
                    "correct": correct,
                    "question": output.question if output.need_clarification else None,
                    "missing_dimensions": output.missing_dimensions,
                    "reasoning": output.assessment_reasoning,
                    "verification": output.verification if not output.need_clarification else None
                }

                return evaluation

            except Exception as e:
                return {
                    "query": query,
                    "error": str(e),
                    "correct": False
                }

    async def run_evaluation_suite(self):
        """Run complete evaluation suite."""

        # Load test cases from YAML if it exists
        yaml_path = Path(__file__).parent / "clarification_dataset.yaml"
        test_cases = []

        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                dataset = yaml.safe_load(f)

                # Extract test cases from YAML
                for category, cases in dataset['cases'].items():
                    for case in cases:
                        test_cases.append({
                            "name": case['name'],
                            "query": case['input']['query'],
                            "expected": case['expected']['need_clarification'],
                            "category": category
                        })
        else:
            # Fallback to hardcoded test cases
            test_cases = [
                # Clear queries
                {"name": "bitcoin", "query": "What is the current Bitcoin price?", "expected": False, "category": "clear"},
                {"name": "code", "query": "How to implement quicksort in Python?", "expected": False, "category": "clear"},

                # Ambiguous queries
                {"name": "ai", "query": "What is AI?", "expected": True, "category": "ambiguous"},
                {"name": "python", "query": "Tell me about Python", "expected": True, "category": "ambiguous"},
                {"name": "research", "query": "Research climate change", "expected": True, "category": "ambiguous"},

                # Edge cases
                {"name": "minimal", "query": "?", "expected": True, "category": "edge"},
            ]

        print("=" * 60)
        print("CLARIFICATION AGENT EVALUATION")
        print("=" * 60)
        print(f"Running {len(test_cases)} test cases...\n")

        # Run evaluations
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] Evaluating: {test_case['name']} ({test_case['category']})")
            result = await self.evaluate_query(test_case['query'], test_case['expected'])
            result['name'] = test_case['name']
            result['category'] = test_case['category']
            self.results.append(result)

            # Print immediate feedback
            if 'error' in result:
                print(f"  ❌ Error: {result['error']}")
            elif result['correct']:
                print(f"  ✅ Correct prediction")
            else:
                print(f"  ❌ Incorrect: Expected {result['expected']}, got {result['predicted']}")

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate evaluation report."""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)

        # Calculate metrics
        total = len(self.results)
        correct = sum(1 for r in self.results if r.get('correct', False))
        errors = sum(1 for r in self.results if 'error' in r)

        # Binary metrics
        true_positives = sum(1 for r in self.results
                           if r.get('expected') and r.get('predicted') and r.get('correct'))
        false_positives = sum(1 for r in self.results
                            if not r.get('expected') and r.get('predicted'))
        false_negatives = sum(1 for r in self.results
                            if r.get('expected') and not r.get('predicted'))
        true_negatives = sum(1 for r in self.results
                           if not r.get('expected') and not r.get('predicted') and r.get('correct'))

        accuracy = correct / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nOVERALL METRICS:")
        print(f"  Total Cases: {total}")
        print(f"  Correct: {correct} ({accuracy:.1%})")
        print(f"  Errors: {errors}")
        print(f"\nBINARY CLASSIFICATION:")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Precision: {precision:.1%}")
        print(f"  Recall: {recall:.1%}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"\nCONFUSION MATRIX:")
        print(f"  True Positives:  {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  False Negatives: {false_negatives}")
        print(f"  True Negatives:  {true_negatives}")

        # Per-category analysis
        print(f"\nPER-CATEGORY PERFORMANCE:")
        categories = {}
        for result in self.results:
            cat = result.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = {'total': 0, 'correct': 0}
            categories[cat]['total'] += 1
            if result.get('correct', False):
                categories[cat]['correct'] += 1

        for cat, stats in categories.items():
            cat_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {cat}: {stats['correct']}/{stats['total']} ({cat_accuracy:.1%})")

        # Dimension analysis
        print(f"\nDIMENSION ANALYSIS:")
        all_dimensions = []
        for result in self.results:
            if result.get('predicted') and result.get('missing_dimensions'):
                all_dimensions.extend(result['missing_dimensions'])

        if all_dimensions:
            from collections import Counter
            dim_counts = Counter(all_dimensions)
            print("  Most common missing dimensions:")
            for dim, count in dim_counts.most_common(5):
                print(f"    - {dim}: {count} occurrences")

        # Example outputs
        print(f"\nEXAMPLE OUTPUTS:")

        # Show one correct clarification
        for result in self.results:
            if result.get('correct') and result.get('predicted'):
                print(f"\n  Correct Clarification Example:")
                print(f"    Query: {result['query']}")
                print(f"    Question: {result.get('question', 'N/A')}")
                print(f"    Dimensions: {', '.join(result.get('missing_dimensions', []))}")
                break

        # Show one correct non-clarification
        for result in self.results:
            if result.get('correct') and not result.get('predicted'):
                print(f"\n  Correct Non-Clarification Example:")
                print(f"    Query: {result['query']}")
                print(f"    Verification: {result.get('verification', 'N/A')[:100]}...")
                break

        # Save detailed results
        output_path = Path(__file__).parent / "evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump({
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                },
                "results": self.results
            }, f, indent=2)

        print(f"\n📊 Detailed results saved to: {output_path}")


async def main():
    """Main function."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: No OPENAI_API_KEY found. Tests will fail.")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
        return

    evaluator = SimpleClarificationEvaluator()
    await evaluator.run_evaluation_suite()


if __name__ == "__main__":
    asyncio.run(main())
