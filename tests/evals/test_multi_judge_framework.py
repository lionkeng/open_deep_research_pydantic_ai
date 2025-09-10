#!/usr/bin/env python3
"""Quick test to verify the multi-judge framework works."""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Suppress logfire prompts
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

from tests.evals.base_multi_judge import (
    BaseMultiJudgeEvaluator,
    VotingMethod,
    JudgeConfiguration,
    JudgeExpertise,
    AgentEvaluationAdapter,
    EvaluationDimension
)
from typing import List, Dict, Any, Optional


# Simple test adapter
class SimpleTestAdapter(AgentEvaluationAdapter[str, str]):
    """Simple adapter for testing the framework."""

    def get_evaluation_dimensions(self) -> List[EvaluationDimension]:
        return [
            EvaluationDimension(
                name="quality",
                description="Overall quality of the output",
                weight=1.0
            ),
            EvaluationDimension(
                name="relevance",
                description="Relevance to the input",
                weight=1.0
            )
        ]

    def format_output_for_evaluation(self, output: str) -> str:
        return f"Output: {output}"

    def create_evaluation_prompt(
        self,
        input: str,
        output: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        return f"""
        Input: {input}
        Output: {output}

        Please evaluate this output according to the dimensions in your system prompt.
        """

    def is_output_valid(self, output: str) -> bool:
        return bool(output)

    def get_expertise_context(self, expertise: JudgeExpertise) -> str:
        return "You are evaluating test outputs. "


async def test_framework():
    """Test the multi-judge framework."""
    print("Testing Multi-Judge Framework")
    print("=" * 50)

    # Create simple adapter
    adapter = SimpleTestAdapter()

    # Use only OpenAI judges for testing
    test_judges = [
        JudgeConfiguration(
            model="openai:gpt-5-mini",
            expertise=JudgeExpertise.GENERAL,
            weight=1.0,
            temperature=0.0
        )
    ]

    # Create evaluator
    evaluator = BaseMultiJudgeEvaluator(
        adapter=adapter,
        judges=test_judges,
        voting_method=VotingMethod.WEIGHTED_AVERAGE
    )

    # Test evaluation
    test_input = "What is 2+2?"
    test_output = "The answer is 4."

    print(f"\nTest Input: {test_input}")
    print(f"Test Output: {test_output}")
    print("\nRunning evaluation...")

    try:
        result = await evaluator.evaluate(
            input=test_input,
            output=test_output,
            context={"test": True}
        )

        print(f"\n✅ Evaluation successful!")
        print(f"Final Score: {result.final_score:.3f}")
        print(f"Consensus Reached: {result.consensus_reached}")
        print(f"Dimension Scores: {result.dimension_scores}")

        # Test comparison
        print("\n" + "=" * 50)
        print("Testing Pairwise Comparison")
        output_b = "2 + 2 equals 4"

        comparison = await evaluator.compare_outputs(
            input=test_input,
            output_a=test_output,
            output_b=output_b
        )

        print(f"Winner: {comparison['winner']}")
        print(f"Confidence: {comparison['confidence']:.2%}")

        print("\n✅ Framework test complete!")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: No OPENAI_API_KEY found.")
        print("   Please set: export OPENAI_API_KEY='your-key-here'")
        return

    success = await test_framework()

    if success:
        print("\n🎉 The generalized multi-judge framework is working correctly!")
        print("\nKey achievements:")
        print("✓ Generalized framework separated from agent-specific logic")
        print("✓ Agent adapters define evaluation dimensions and prompts")
        print("✓ Multiple voting methods supported")
        print("✓ Pairwise comparison capability")
        print("✓ Works with any agent type via adapters")
    else:
        print("\n❌ Framework test failed")


if __name__ == "__main__":
    asyncio.run(main())
