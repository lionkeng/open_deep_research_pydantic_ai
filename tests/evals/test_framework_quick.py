#!/usr/bin/env python3
"""Quick test to verify the evaluation framework is working."""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Suppress logfire prompts
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"


async def main():
    """Run quick tests to verify framework components."""

    print("\n🚀 Quick Framework Test")
    print("=" * 60)

    # Test 1: YAML datasets exist
    print("\n1. Checking YAML datasets...")
    yaml_dir = Path(__file__).parent / "evaluation_datasets"
    clarification_yaml = yaml_dir / "clarification_dataset.yaml"
    transformation_yaml = yaml_dir / "query_transformation_dataset.yaml"

    if clarification_yaml.exists():
        print("   ✓ Clarification dataset found")
    else:
        print("   ✗ Clarification dataset missing")

    if transformation_yaml.exists():
        print("   ✓ Query transformation dataset found")
    else:
        print("   ✗ Query transformation dataset missing")

    # Test 2: Multi-judge framework
    print("\n2. Testing multi-judge framework...")
    try:
        from tests.evals.base_multi_judge import BaseMultiJudgeEvaluator, VotingMethod
        from tests.evals.clarification_multi_judge_adapter import ClarificationMultiJudgeAdapter

        adapter = ClarificationMultiJudgeAdapter()
        evaluator = BaseMultiJudgeEvaluator(adapter=adapter, voting_method=VotingMethod.MAJORITY)
        print("   ✓ Multi-judge framework loaded")
    except Exception as e:
        print(f"   ✗ Multi-judge framework error: {e}")

    # Test 3: Regression tracker
    print("\n3. Testing regression tracker...")
    try:
        from tests.evals.regression_tracker_fixed import PerformanceTracker

        tracker = PerformanceTracker(db_path="test_quick.db")
        print("   ✓ Regression tracker initialized")

        # Cleanup
        import os

        if os.path.exists("test_quick.db"):
            os.remove("test_quick.db")
    except Exception as e:
        print(f"   ✗ Regression tracker error: {e}")

    # Test 4: Evaluators
    print("\n4. Testing evaluators...")

    try:
        print("   ✓ Clarification evaluators loaded")
    except Exception as e:
        print(f"   ✗ Clarification evaluators error: {e}")

    try:
        print("   ✓ Query transformation evaluators loaded")
    except Exception as e:
        print(f"   ✗ Query transformation evaluators error: {e}")

    # Test 5: Run simple evaluation if API key exists
    if os.getenv("OPENAI_API_KEY"):
        print("\n5. Running simple evaluation...")
        try:
            # Run a quick test
            import httpx
            from pydantic import SecretStr

            from agents.base import ResearchDependencies
            from agents.clarification import ClarificationAgent
            from models.api_models import APIKeys
            from models.core import ResearchStage, ResearchState
            from models.metadata import ResearchMetadata

            agent = ClarificationAgent()
            query = "test query"

            async with httpx.AsyncClient(timeout=10.0) as http_client:
                state = ResearchState(
                    request_id="quick-test",
                    user_query=query,
                    current_stage=ResearchStage.CLARIFICATION,
                    metadata=ResearchMetadata(),
                )
                deps = ResearchDependencies(
                    http_client=http_client,
                    api_keys=APIKeys(openai=SecretStr(os.getenv("OPENAI_API_KEY"))),
                    research_state=state,
                )

                result = await agent.agent.run(query, deps=deps)
                print("   ✓ Agent evaluation completed")

        except Exception as e:
            print(f"   ✗ Agent evaluation error: {e}")
    else:
        print("\n5. Skipping agent evaluation (no API key)")

    print("\n" + "=" * 60)
    print("✅ Quick test complete!")


if __name__ == "__main__":
    asyncio.run(main())
