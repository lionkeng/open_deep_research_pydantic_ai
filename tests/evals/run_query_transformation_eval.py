#!/usr/bin/env python3
"""
Run query transformation evaluation using pydantic-ai evaluation framework.

This script runs comprehensive evaluation of the query transformation agent
following the same patterns as the clarification agent evaluation.
"""

import asyncio
import os
import sys
import yaml
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from statistics import mean, stdev
from collections import Counter

# Change to project root directory
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Suppress logfire prompts
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

import httpx
from pydantic import SecretStr
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

from src.agents.query_transformation import QueryTransformationAgent
from src.agents.base import ResearchDependencies
from src.models.metadata import ResearchMetadata
from src.models.core import ResearchState, ResearchStage
from src.models.api_models import APIKeys
from src.models.research_plan_models import TransformedQuery

from tests.evals.query_transformation_evals import (
    QueryTransformationInput,
    QueryTransformationExpectedOutput,
    SearchQueryRelevanceEvaluator,
    ObjectiveCoverageEvaluator,
    PlanCoherenceEvaluator,
    QueryDiversityEvaluator,
    TransformationAccuracyEvaluator,
)


# Map evaluator names to classes
EVALUATOR_MAP = {
    "SearchQueryRelevanceEvaluator": SearchQueryRelevanceEvaluator,
    "ObjectiveCoverageEvaluator": ObjectiveCoverageEvaluator,
    "PlanCoherenceEvaluator": PlanCoherenceEvaluator,
    "QueryDiversityEvaluator": QueryDiversityEvaluator,
    "TransformationAccuracyEvaluator": TransformationAccuracyEvaluator,
}


class QueryTransformationEvaluator:
    """Evaluator for query transformation agent."""

    def __init__(self):
        self.agent = QueryTransformationAgent()
        self.results = []
        self.timing_data = []

    def load_dataset_from_yaml(
        self,
        file_path: Optional[Path] = None,
        categories: Optional[List[str]] = None
    ) -> Dataset:
        """
        Load dataset from YAML file, similar to clarification eval.

        Args:
            file_path: Path to YAML file. If None, uses default location.
            categories: Optional list of categories to include.

        Returns:
            Dataset object ready for evaluation
        """
        # Default file path
        if file_path is None:
            file_path = Path(__file__).parent / "evaluation_datasets" / "query_transformation_dataset.yaml"

        # Load YAML data
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        # Collect all cases
        all_cases = []

        # Define category keys to process
        category_keys = [
            "golden_standard_cases",
            "technical_cases",
            "scientific_cases",
            "business_cases",
            "edge_cases",
            "cross_domain_cases",
            "performance_cases"
        ]

        # Filter categories if specified
        if categories:
            category_keys = [k for k in category_keys if any(cat in k for cat in categories)]

        # Process each category
        for category_key in category_keys:
            if category_key in data:
                category_cases = data[category_key]
                for case_data in category_cases:
                    try:
                        # Create input
                        inputs = QueryTransformationInput(**case_data.get("inputs", {}))

                        # Create expected output
                        expected_data = case_data.get("expected_output", {})
                        expected_output = QueryTransformationExpectedOutput(**expected_data) if expected_data else None

                        # Create evaluators
                        evaluator_names = case_data.get("evaluators", ["SearchQueryRelevanceEvaluator"])
                        evaluators = []
                        for evaluator_name in evaluator_names:
                            if evaluator_name in EVALUATOR_MAP:
                                evaluators.append(EVALUATOR_MAP[evaluator_name]())
                            else:
                                print(f"Warning: Unknown evaluator '{evaluator_name}', skipping")

                        # Create case
                        case_name = case_data.get("name", case_data.get("id", "unnamed_case"))

                        case = Case(
                            name=case_name,
                            inputs=inputs,
                            expected_output=expected_output,
                            evaluators=evaluators
                        )
                        all_cases.append(case)

                    except Exception as e:
                        print(f"Error creating case {case_data.get('id', 'unknown')}: {e}")

        return Dataset(cases=all_cases)

    async def evaluate_query(
        self,
        query: str,
        complexity: str = "medium",
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single query transformation.

        Args:
            query: The query to transform
            complexity: Query complexity level
            domain: Query domain

        Returns:
            Evaluation results
        """
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            state = ResearchState(
                request_id=f"eval-{abs(hash(query))}",
                user_query=query,
                current_stage=ResearchStage.RESEARCH_EXECUTION,
                metadata=ResearchMetadata()
            )
            deps = ResearchDependencies(
                http_client=http_client,
                api_keys=APIKeys(
                    openai=SecretStr(openai_key) if (openai_key := os.getenv("OPENAI_API_KEY")) else None,
                    anthropic=SecretStr(anthropic_key) if (anthropic_key := os.getenv("ANTHROPIC_API_KEY")) else None
                ),
                research_state=state
            )

            try:
                # Time the agent response
                start_time = time.time()
                run_result = await self.agent.agent.run(query, deps=deps)
                end_time = time.time()
                response_time = end_time - start_time

                output: TransformedQuery = run_result.output

                evaluation = {
                    "query": query,
                    "complexity": complexity,
                    "domain": domain,
                    "response_time": response_time,
                    "num_objectives": len(output.research_plan.objectives),
                    "num_search_queries": len(output.search_queries.queries),
                    "confidence_score": output.confidence_score,
                    "has_methodology": output.research_plan.methodology is not None,
                    "output": output
                }

                # Store timing data
                self.timing_data.append(response_time)

                return evaluation

            except Exception as e:
                return {
                    "query": query,
                    "error": str(e),
                    "response_time": None
                }

    async def run_evaluation_suite(self, categories: Optional[List[str]] = None):
        """
        Run evaluation suite using YAML dataset.

        Args:
            categories: Optional list of categories to evaluate
        """
        print("=" * 80)
        print("QUERY TRANSFORMATION AGENT EVALUATION")
        print("=" * 80)

        # Load dataset from YAML
        yaml_path = Path(__file__).parent / "evaluation_datasets" / "query_transformation_dataset.yaml"

        if not yaml_path.exists():
            print(f"Warning: YAML dataset not found at {yaml_path}")
            print("Using minimal test set...")

            # Minimal fallback test cases
            test_cases = [
                {
                    "name": "simple_factual",
                    "query": "What is machine learning?",
                    "complexity": "simple",
                    "domain": "technical"
                },
                {
                    "name": "complex_research",
                    "query": "Analyze the impact of climate change on global agriculture",
                    "complexity": "complex",
                    "domain": "scientific"
                }
            ]
        else:
            # Load from YAML and extract test cases
            dataset = self.load_dataset_from_yaml(yaml_path, categories)
            test_cases = []

            for case in dataset.cases:
                test_case = {
                    "name": case.name,
                    "query": case.inputs.query,
                    "complexity": getattr(case.inputs, 'complexity', 'medium'),
                    "domain": getattr(case.inputs, 'domain', None),
                    "expected": case.expected_output
                }
                test_cases.append(test_case)

        print(f"Running {len(test_cases)} test cases...\n")

        # Run evaluations
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] Evaluating: {test_case['name']}")

            result = await self.evaluate_query(
                test_case['query'],
                test_case['complexity'],
                test_case.get('domain')
            )

            result['name'] = test_case['name']
            self.results.append(result)

            # Print immediate feedback
            if 'error' in result:
                print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  ✅ Transformed successfully")
                print(f"     Objectives: {result['num_objectives']}")
                print(f"     Search queries: {result['num_search_queries']}")
                print(f"     Confidence: {result['confidence_score']:.2f}")

            if result.get('response_time'):
                print(f"     Response time: {result['response_time']:.2f}s")

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive evaluation report."""
        print("\n" + "=" * 80)
        print("EVALUATION REPORT")
        print("=" * 80)

        # Basic metrics
        total = len(self.results)
        successful = sum(1 for r in self.results if 'error' not in r)
        errors = total - successful

        print(f"\n📊 OVERALL METRICS:")
        print(f"  Total Cases: {total}")
        print(f"  Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"  Errors: {errors}")

        # Transformation metrics for successful cases
        successful_results = [r for r in self.results if 'error' not in r]

        if successful_results:
            print(f"\n📈 TRANSFORMATION METRICS:")

            # Objectives
            obj_counts = [r['num_objectives'] for r in successful_results]
            print(f"  Objectives per query:")
            print(f"    - Average: {mean(obj_counts):.1f}")
            print(f"    - Min: {min(obj_counts)}")
            print(f"    - Max: {max(obj_counts)}")

            # Search queries
            query_counts = [r['num_search_queries'] for r in successful_results]
            print(f"  Search queries per query:")
            print(f"    - Average: {mean(query_counts):.1f}")
            print(f"    - Min: {min(query_counts)}")
            print(f"    - Max: {max(query_counts)}")

            # Confidence scores
            confidence_scores = [r['confidence_score'] for r in successful_results]
            print(f"  Confidence scores:")
            print(f"    - Average: {mean(confidence_scores):.2f}")
            print(f"    - Min: {min(confidence_scores):.2f}")
            print(f"    - Max: {max(confidence_scores):.2f}")

            # Methodology presence
            has_methodology = sum(1 for r in successful_results if r['has_methodology'])
            print(f"  Has methodology: {has_methodology}/{len(successful_results)} ({has_methodology/len(successful_results)*100:.1f}%)")

        # Performance metrics
        if self.timing_data:
            print(f"\n⏱️  PERFORMANCE METRICS:")
            print(f"  Average Response Time: {mean(self.timing_data):.2f}s")
            print(f"  Min Response Time: {min(self.timing_data):.2f}s")
            print(f"  Max Response Time: {max(self.timing_data):.2f}s")
            if len(self.timing_data) > 1:
                print(f"  Std Dev: {stdev(self.timing_data):.2f}s")

        # Complexity analysis
        print(f"\n📂 PER-COMPLEXITY PERFORMANCE:")
        complexity_groups = {}
        for result in successful_results:
            complexity = result.get('complexity', 'unknown')
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(result)

        for complexity, results in complexity_groups.items():
            print(f"  {complexity}:")
            print(f"    - Count: {len(results)}")
            print(f"    - Avg objectives: {mean([r['num_objectives'] for r in results]):.1f}")
            print(f"    - Avg queries: {mean([r['num_search_queries'] for r in results]):.1f}")
            print(f"    - Avg confidence: {mean([r['confidence_score'] for r in results]):.2f}")

        # Example output
        if successful_results:
            print(f"\n📝 EXAMPLE TRANSFORMATION:")
            example = successful_results[0]
            print(f"  Query: '{example['query']}'")
            if 'output' in example:
                output = example['output']
                print(f"  Generated {len(output.research_plan.objectives)} objectives:")
                for i, obj in enumerate(output.research_plan.objectives[:3], 1):
                    print(f"    {i}. {obj.objective[:80]}...")
                print(f"  Generated {len(output.search_queries.queries)} search queries:")
                for i, q in enumerate(output.search_queries.queries[:3], 1):
                    print(f"    {i}. {q.query[:80]}...")

        # Save results
        results_dir = Path('./eval_results')
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / "query_transformation_results.json"

        # Prepare serializable results
        serializable_results = []
        for r in self.results:
            result_copy = r.copy()
            if 'output' in result_copy:
                # Remove non-serializable output object
                del result_copy['output']
            serializable_results.append(result_copy)

        with open(output_path, 'w') as f:
            json.dump({
                "summary": {
                    "total_cases": total,
                    "successful": successful,
                    "errors": errors
                },
                "metrics": {
                    "avg_objectives": mean(obj_counts) if successful_results else 0,
                    "avg_search_queries": mean(query_counts) if successful_results else 0,
                    "avg_confidence": mean(confidence_scores) if successful_results else 0
                },
                "performance": {
                    "avg_response_time": mean(self.timing_data) if self.timing_data else None,
                    "min_response_time": min(self.timing_data) if self.timing_data else None,
                    "max_response_time": max(self.timing_data) if self.timing_data else None
                },
                "results": serializable_results
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

    print("\n🚀 Starting Query Transformation Agent Evaluation...")
    print(f"   Using model: {os.getenv('MODEL_NAME', 'gpt-4o-mini')}")

    evaluator = QueryTransformationEvaluator()

    # You can specify categories to test specific subsets
    # e.g., categories=["golden_standard", "technical"]
    await evaluator.run_evaluation_suite()


if __name__ == "__main__":
    asyncio.run(main())
