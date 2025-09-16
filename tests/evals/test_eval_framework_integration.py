#!/usr/bin/env python3
"""Integration test for the complete evaluation framework.

This test verifies that all components work together:
- YAML dataset loading
- Multi-judge evaluation
- Regression tracking
- Both clarification and query transformation agents
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Suppress logfire prompts
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

import httpx
from pydantic import SecretStr

# Import evaluation components
from tests.evals.run_clarification_eval import MultiQuestionClarificationEvaluator
from tests.evals.run_query_transformation_eval import QueryTransformationEvaluator
from tests.evals.base_multi_judge import BaseMultiJudgeEvaluator, VotingMethod
from tests.evals.clarification_multi_judge_adapter import ClarificationMultiJudgeAdapter
from tests.evals.query_transformation_multi_judge_adapter import QueryTransformationMultiJudgeAdapter
from tests.evals.regression_tracker_fixed import PerformanceTracker

# Import agents and dependencies
from agents.clarification import ClarificationAgent
from agents.query_transformation import QueryTransformationAgent
from agents.base import ResearchDependencies
from models.core import ResearchState, ResearchStage
from models.metadata import ResearchMetadata
from models.api_models import APIKeys


async def test_clarification_evaluation():
    """Test clarification agent evaluation pipeline."""
    print("\n" + "="*60)
    print("Testing Clarification Agent Evaluation")
    print("="*60)

    # Initialize evaluator
    evaluator = MultiQuestionClarificationEvaluator()

    # Load dataset (testing embedded loader)
    print("Loading YAML dataset...")
    dataset = evaluator.load_dataset_from_yaml(categories=["golden_standard"])
    print(f"✓ Loaded {len(dataset.samples)} test cases")

    # Test multi-judge evaluation
    adapter = ClarificationMultiJudgeAdapter()
    multi_judge = BaseMultiJudgeEvaluator(
        adapter=adapter,
        voting_method=VotingMethod.CONFIDENCE_WEIGHTED,
        consensus_threshold=0.7
    )

    # Get first test case
    if dataset.samples:
        sample = dataset.samples[0]
        print(f"\nEvaluating: '{sample.inputs.get('query', 'N/A')}'")

        # Run agent
        agent = ClarificationAgent()
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            state = ResearchState(
                request_id="test-clarification",
                user_query=sample.inputs.get("query", ""),
                current_stage=ResearchStage.CLARIFICATION,
                metadata=ResearchMetadata()
            )
            deps = ResearchDependencies(
                http_client=http_client,
                api_keys=APIKeys(
                    openai=SecretStr(key) if (key := os.getenv("OPENAI_API_KEY")) else None
                ),
                research_state=state
            )

            try:
                result = await agent.agent.run(sample.inputs.get("query", ""), deps=deps)
                print(f"✓ Agent output: needs_clarification={result.output.needs_clarification}")

                # Multi-judge evaluation
                consensus = await multi_judge.evaluate(
                    input=sample.inputs.get("query", ""),
                    output=result.output,
                    context={"test": True}
                )
                print(f"✓ Multi-judge consensus: score={consensus.final_score:.3f}")

                return True
            except Exception as e:
                print(f"✗ Error: {e}")
                return False

    return False


async def test_query_transformation_evaluation():
    """Test query transformation agent evaluation pipeline."""
    print("\n" + "="*60)
    print("Testing Query Transformation Agent Evaluation")
    print("="*60)

    # Initialize evaluator
    evaluator = QueryTransformationEvaluator()

    # Load dataset (testing embedded loader)
    print("Loading YAML dataset...")
    dataset = evaluator.load_dataset_from_yaml(categories=["golden_standard"])
    print(f"✓ Loaded {len(dataset.samples)} test cases")

    # Test multi-judge evaluation
    adapter = QueryTransformationMultiJudgeAdapter()
    multi_judge = BaseMultiJudgeEvaluator(
        adapter=adapter,
        voting_method=VotingMethod.EXPERT_WEIGHTED,
        consensus_threshold=0.75
    )

    # Get first test case
    if dataset.samples:
        sample = dataset.samples[0]
        print(f"\nEvaluating: '{sample.inputs.get('query', 'N/A')}'")

        # Run agent
        agent = QueryTransformationAgent()
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            state = ResearchState(
                request_id="test-transformation",
                user_query=sample.inputs.get("query", ""),
                current_stage=ResearchStage.RESEARCH_EXECUTION,
                metadata=ResearchMetadata()
            )
            deps = ResearchDependencies(
                http_client=http_client,
                api_keys=APIKeys(
                    openai=SecretStr(key) if (key := os.getenv("OPENAI_API_KEY")) else None
                ),
                research_state=state
            )

            try:
                result = await agent.agent.run(sample.inputs.get("query", ""), deps=deps)
                output = result.output
                print(f"✓ Agent output: {len(output.search_queries.queries)} queries")

                # Multi-judge evaluation
                consensus = await multi_judge.evaluate(
                    input=sample.inputs.get("query", ""),
                    output=output,
                    context={"test": True}
                )
                print(f"✓ Multi-judge consensus: score={consensus.final_score:.3f}")

                return True
            except Exception as e:
                print(f"✗ Error: {e}")
                return False

    return False


async def test_regression_tracking():
    """Test regression tracking system."""
    print("\n" + "="*60)
    print("Testing Regression Tracking System")
    print("="*60)

    # Initialize tracker with test database
    tracker = PerformanceTracker(db_path="test_regression.db")

    print("Testing performance tracking...")
    try:
        # Run a minimal evaluation
        metrics, alerts = await tracker.run_performance_evaluation(
            git_commit="test-commit-123",
            model_version="test-model-v1"
        )

        print(f"✓ Tracked metrics: accuracy={metrics.accuracy:.3f}")
        print(f"✓ Regression alerts: {len(alerts)} found")

        # Test historical analysis
        historical = tracker.get_historical_performance(limit=5)
        print(f"✓ Historical records: {len(historical)} entries")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        # Cleanup test database
        import os
        if os.path.exists("test_regression.db"):
            os.remove("test_regression.db")


async def test_framework_integration():
    """Test full framework integration."""
    print("\n" + "="*60)
    print("Testing Full Framework Integration")
    print("="*60)

    results = {}

    # Check API key availability
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No OPENAI_API_KEY found - skipping agent tests")
        print("   Set: export OPENAI_API_KEY='your-key-here'")
        results["clarification"] = False
        results["transformation"] = False
    else:
        # Test clarification evaluation
        results["clarification"] = await test_clarification_evaluation()

        # Test query transformation evaluation
        results["transformation"] = await test_query_transformation_evaluation()

    # Test regression tracking (doesn't need API key)
    results["regression"] = await test_regression_tracking()

    # Summary
    print("\n" + "="*60)
    print("Integration Test Summary")
    print("="*60)

    for component, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{component.capitalize():20} {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✅ All integration tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check output above.")

    return all_passed


def main():
    """Main entry point."""
    print("\n🚀 Evaluation Framework Integration Test")
    print("="*60)
    print("This test verifies all evaluation components work together:")
    print("- YAML dataset loading")
    print("- Multi-judge consensus evaluation")
    print("- Regression tracking")
    print("- Both agent evaluations")
    print("="*60)

    success = asyncio.run(test_framework_integration())

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
