#!/usr/bin/env python
"""
Evaluation runner for multi-question clarification agent.

This script evaluates the clarification agent's performance with the new
multi-question format using real LLM calls.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import yaml
from collections import Counter
from statistics import mean, stdev

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Suppress logfire prompts
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

# Note: .env is loaded automatically when importing from src

import httpx
from src.agents.clarification import ClarificationAgent, ClarifyWithUser
from src.agents.base import ResearchDependencies
from src.models.metadata import ResearchMetadata
from src.models.core import ResearchState, ResearchStage
from src.models.clarification import ClarificationQuestion, ClarificationRequest


class MultiQuestionClarificationEvaluator:
    """Evaluator for multi-question clarification agent."""

    def __init__(self):
        self.agent = ClarificationAgent()
        self.results = []
        self.timing_data = []
        self.question_metrics = []

    async def evaluate_query(
        self,
        query: str,
        expected_clarification: bool,
        expected_question_count: Optional[int] = None,
        expected_question_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate a single query with multi-question support.

        Args:
            query: The research query to evaluate
            expected_clarification: Whether clarification is expected
            expected_question_count: Expected number of questions (if clarification needed)
            expected_question_types: Expected types of questions
        """
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            state = ResearchState(
                request_id=f"eval-{abs(hash(query))}",
                user_query=query,
                current_stage=ResearchStage.CLARIFICATION,
                metadata=ResearchMetadata()
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
                # Time the agent response
                start_time = time.time()
                run_result = await self.agent.agent.run(query, deps=deps)
                end_time = time.time()
                response_time = end_time - start_time

                output: ClarifyWithUser = run_result.output

                # Evaluate basic prediction
                correct_prediction = output.needs_clarification == expected_clarification

                # Evaluate multi-question aspects
                question_analysis = {}
                if output.needs_clarification and output.request:
                    questions = output.request.questions
                    question_analysis = {
                        "num_questions": len(questions),
                        "question_types": [q.question_type for q in questions],
                        "required_count": sum(1 for q in questions if q.is_required),
                        "optional_count": sum(1 for q in questions if not q.is_required),
                        "questions": [
                            {
                                "text": q.question,
                                "type": q.question_type,
                                "required": q.is_required,
                                "has_context": q.context is not None,
                                "has_choices": q.choices is not None,
                                "num_choices": len(q.choices) if q.choices else 0
                            }
                            for q in questions
                        ],
                        "average_question_length": mean([len(q.question) for q in questions]) if questions else 0,
                        "unique_types": list(set(q.question_type for q in questions))
                    }

                    # Check question count expectation
                    if expected_question_count is not None:
                        question_analysis["expected_count_match"] = len(questions) == expected_question_count

                    # Check question types expectation
                    if expected_question_types is not None:
                        actual_types = set(q.question_type for q in questions)
                        expected_types = set(expected_question_types)
                        question_analysis["type_coverage"] = len(actual_types.intersection(expected_types)) / len(expected_types) if expected_types else 1.0

                evaluation = {
                    "query": query,
                    "expected": expected_clarification,
                    "predicted": output.needs_clarification,
                    "correct": correct_prediction,
                    "response_time": response_time,
                    "reasoning": output.reasoning,
                    "missing_dimensions": output.missing_dimensions if hasattr(output, 'missing_dimensions') else [],
                    "question_analysis": question_analysis
                }

                # Store timing data
                self.timing_data.append(response_time)

                # Store question metrics
                if question_analysis:
                    self.question_metrics.append(question_analysis)

                return evaluation

            except Exception as e:
                return {
                    "query": query,
                    "error": str(e),
                    "correct": False,
                    "response_time": None
                }

    async def run_evaluation_suite(self):
        """Run complete evaluation suite with multi-question support."""

        # Load test cases from YAML
        yaml_path = Path(__file__).parent / "clarification_dataset.yaml"
        test_cases = []

        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                dataset = yaml.safe_load(f)

                # Extract test cases from YAML
                for category, cases in dataset['cases'].items():
                    for case in cases:
                        test_case = {
                            "name": case['name'],
                            "query": case['input']['query'],
                            "expected": case['expected']['needs_clarification'],
                            "category": category
                        }
                        # Add multi-question expectations if present
                        if 'expected_questions' in case['expected']:
                            test_case['expected_question_count'] = case['expected'].get('expected_questions')
                        if 'question_types' in case['expected']:
                            test_case['expected_question_types'] = case['expected'].get('question_types')
                        test_cases.append(test_case)
        else:
            # Enhanced test cases with multi-question expectations
            test_cases = [
                # Clear queries (no clarification needed)
                {
                    "name": "bitcoin_price",
                    "query": "What is the current Bitcoin price in USD?",
                    "expected": False,
                    "category": "clear_specific"
                },
                {
                    "name": "resnet_comparison",
                    "query": "Compare ResNet-50 vs VGG-16 for ImageNet classification accuracy",
                    "expected": False,
                    "category": "clear_specific"
                },

                # Ambiguous queries (multiple questions expected)
                {
                    "name": "vague_reference",
                    "query": "Tell me about it",
                    "expected": True,
                    "expected_question_count": 2,
                    "expected_question_types": ["text"],
                    "category": "ambiguous"
                },
                {
                    "name": "incomplete_comparison",
                    "query": "Compare them",
                    "expected": True,
                    "expected_question_count": 3,
                    "expected_question_types": ["text", "choice"],
                    "category": "ambiguous"
                },
                {
                    "name": "broad_research",
                    "query": "Research machine learning",
                    "expected": True,
                    "expected_question_count": 4,
                    "expected_question_types": ["text", "choice", "multi_choice"],
                    "category": "ambiguous"
                },

                # Partial context (focused questions expected)
                {
                    "name": "partial_project",
                    "query": "I need help with my Python project",
                    "expected": True,
                    "expected_question_count": 3,
                    "category": "partial_context"
                },
                {
                    "name": "vague_database",
                    "query": "What's the best database?",
                    "expected": True,
                    "expected_question_count": 4,
                    "expected_question_types": ["text", "choice"],
                    "category": "partial_context"
                },

                # Edge cases
                {
                    "name": "minimal",
                    "query": "?",
                    "expected": True,
                    "category": "edge"
                },
                {
                    "name": "single_word",
                    "query": "Python",
                    "expected": True,
                    "expected_question_count": 2,
                    "category": "edge"
                }
            ]

        print("=" * 80)
        print("MULTI-QUESTION CLARIFICATION AGENT EVALUATION")
        print("=" * 80)
        print(f"Running {len(test_cases)} test cases...\n")

        # Run evaluations
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] Evaluating: {test_case['name']} ({test_case['category']})")

            result = await self.evaluate_query(
                test_case['query'],
                test_case['expected'],
                test_case.get('expected_question_count'),
                test_case.get('expected_question_types')
            )

            result['name'] = test_case['name']
            result['category'] = test_case['category']
            self.results.append(result)

            # Print immediate feedback
            if 'error' in result:
                print(f"  ❌ Error: {result['error']}")
            elif result['correct']:
                print(f"  ✅ Correct prediction")
                if result.get('question_analysis'):
                    qa = result['question_analysis']
                    print(f"     Questions: {qa['num_questions']} ({qa['required_count']} required, {qa['optional_count']} optional)")
                    print(f"     Types: {', '.join(qa['unique_types'])}")
            else:
                print(f"  ❌ Incorrect: Expected {result['expected']}, got {result['predicted']}")

            if result.get('response_time'):
                print(f"     Response time: {result['response_time']:.2f}s")

        # Generate comprehensive report
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive evaluation report for multi-question system."""
        print("\n" + "=" * 80)
        print("EVALUATION REPORT")
        print("=" * 80)

        # Basic metrics
        total = len(self.results)
        correct = sum(1 for r in self.results if r.get('correct', False))
        errors = sum(1 for r in self.results if 'error' in r)

        # Binary classification metrics
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

        print(f"\n📊 OVERALL METRICS:")
        print(f"  Total Cases: {total}")
        print(f"  Correct: {correct} ({accuracy:.1%})")
        print(f"  Errors: {errors}")
        print(f"\n🎯 BINARY CLASSIFICATION:")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Precision: {precision:.1%}")
        print(f"  Recall: {recall:.1%}")
        print(f"  F1 Score: {f1:.3f}")

        print(f"\n📈 CONFUSION MATRIX:")
        print(f"  True Positives:  {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  False Negatives: {false_negatives}")
        print(f"  True Negatives:  {true_negatives}")

        # Performance metrics
        if self.timing_data:
            print(f"\n⏱️  PERFORMANCE METRICS:")
            print(f"  Average Response Time: {mean(self.timing_data):.2f}s")
            print(f"  Min Response Time: {min(self.timing_data):.2f}s")
            print(f"  Max Response Time: {max(self.timing_data):.2f}s")
            if len(self.timing_data) > 1:
                print(f"  Std Dev: {stdev(self.timing_data):.2f}s")

        # Multi-question analysis
        if self.question_metrics:
            print(f"\n❓ MULTI-QUESTION ANALYSIS:")
            all_counts = [m['num_questions'] for m in self.question_metrics]
            print(f"  Average Questions per Query: {mean(all_counts):.1f}")
            print(f"  Min Questions: {min(all_counts)}")
            print(f"  Max Questions: {max(all_counts)}")

            # Question type distribution
            all_types = []
            for m in self.question_metrics:
                all_types.extend(m['question_types'])
            type_counts = Counter(all_types)
            print(f"  Question Type Distribution:")
            for qtype, count in type_counts.most_common():
                percentage = (count / len(all_types) * 100) if all_types else 0
                print(f"    - {qtype}: {count} ({percentage:.1f}%)")

            # Required vs Optional
            total_required = sum(m['required_count'] for m in self.question_metrics)
            total_optional = sum(m['optional_count'] for m in self.question_metrics)
            total_questions = total_required + total_optional
            if total_questions > 0:
                print(f"  Required vs Optional:")
                print(f"    - Required: {total_required} ({total_required/total_questions*100:.1f}%)")
                print(f"    - Optional: {total_optional} ({total_optional/total_questions*100:.1f}%)")

            # Average question length
            avg_lengths = [m['average_question_length'] for m in self.question_metrics if m['average_question_length'] > 0]
            if avg_lengths:
                print(f"  Average Question Length: {mean(avg_lengths):.0f} characters")

        # Per-category analysis
        print(f"\n📂 PER-CATEGORY PERFORMANCE:")
        categories = {}
        for result in self.results:
            cat = result.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = {
                    'total': 0,
                    'correct': 0,
                    'question_counts': []
                }
            categories[cat]['total'] += 1
            if result.get('correct', False):
                categories[cat]['correct'] += 1
            if result.get('question_analysis'):
                categories[cat]['question_counts'].append(result['question_analysis']['num_questions'])

        for cat, stats in categories.items():
            cat_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            avg_questions = mean(stats['question_counts']) if stats['question_counts'] else 0
            print(f"  {cat}:")
            print(f"    - Accuracy: {stats['correct']}/{stats['total']} ({cat_accuracy:.1%})")
            if stats['question_counts']:
                print(f"    - Avg Questions: {avg_questions:.1f}")

        # Missing dimensions analysis
        print(f"\n🔍 DIMENSION ANALYSIS:")
        all_dimensions = []
        for result in self.results:
            if result.get('predicted') and result.get('missing_dimensions'):
                all_dimensions.extend(result['missing_dimensions'])

        if all_dimensions:
            dim_counts = Counter(all_dimensions)
            print("  Most common missing dimensions:")
            for dim, count in dim_counts.most_common(5):
                print(f"    - {dim}: {count} occurrences")

        # Example outputs
        print(f"\n📝 EXAMPLE OUTPUTS:")

        # Show example with multiple questions
        for result in self.results:
            if result.get('correct') and result.get('predicted') and result.get('question_analysis'):
                qa = result['question_analysis']
                if qa['num_questions'] > 1:
                    print(f"\n  Multi-Question Clarification Example:")
                    print(f"    Query: '{result['query']}'")
                    print(f"    Generated {qa['num_questions']} questions:")
                    for i, q in enumerate(qa['questions'][:3], 1):  # Show up to 3 questions
                        req_tag = "[Required]" if q['required'] else "[Optional]"
                        print(f"      {i}. {req_tag} {q['text'][:80]}...")
                        print(f"         Type: {q['type']}, Choices: {q['num_choices']}")
                    break

        # Show correct non-clarification example
        for result in self.results:
            if result.get('correct') and not result.get('predicted'):
                print(f"\n  Correct Non-Clarification Example:")
                print(f"    Query: '{result['query']}'")
                print(f"    Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
                break

        # Save detailed results
        output_path = Path(__file__).parent / "evaluation_results_multi.json"
        with open(output_path, 'w') as f:
            json.dump({
                "summary": {
                    "total_cases": total,
                    "correct": correct,
                    "errors": errors
                },
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                },
                "performance": {
                    "avg_response_time": mean(self.timing_data) if self.timing_data else None,
                    "min_response_time": min(self.timing_data) if self.timing_data else None,
                    "max_response_time": max(self.timing_data) if self.timing_data else None
                },
                "multi_question_stats": {
                    "avg_questions_per_query": mean([m['num_questions'] for m in self.question_metrics]) if self.question_metrics else 0,
                    "question_type_distribution": dict(Counter(sum([m['question_types'] for m in self.question_metrics], [])))
                },
                "results": self.results
            }, f, indent=2)

        print(f"\n💾 Detailed results saved to: {output_path}")
        print("=" * 80)


async def main():
    """Main function."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: No OPENAI_API_KEY found.")
        print("   Please set: export OPENAI_API_KEY='your-key-here'")
        print("   Or run: source .env")
        return

    print("\n🚀 Starting Multi-Question Clarification Agent Evaluation...")
    print(f"   Using model: {os.getenv('MODEL_NAME', 'gpt-4o-mini')}")

    evaluator = MultiQuestionClarificationEvaluator()
    await evaluator.run_evaluation_suite()


if __name__ == "__main__":
    asyncio.run(main())
