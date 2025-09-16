#!/usr/bin/env python3
"""Demonstration of the Generalized Multi-Judge Evaluation Framework.

This script shows how the multi-judge framework can be used to evaluate
different agent types using the same consensus mechanisms.
"""

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

import httpx
from pydantic import SecretStr

from tests.evals.base_multi_judge import (
    BaseMultiJudgeEvaluator,
    VotingMethod,
    JudgeConfiguration,
    JudgeExpertise
)
from tests.evals.clarification_multi_judge_adapter import ClarificationMultiJudgeAdapter
from tests.evals.query_transformation_multi_judge_adapter import QueryTransformationMultiJudgeAdapter

from agents.clarification import ClarificationAgent, ClarifyWithUser
from agents.query_transformation import QueryTransformationAgent
from agents.base import ResearchDependencies
from models.core import ResearchState, ResearchStage
from models.metadata import ResearchMetadata
from models.api_models import APIKeys
from models.research_plan_models import TransformedQuery


async def evaluate_clarification_agent():
    """Demonstrate multi-judge evaluation of ClarificationAgent."""
    print("\n" + "=" * 80)
    print("CLARIFICATION AGENT - MULTI-JUDGE EVALUATION")
    print("=" * 80)

    # Create adapter and evaluator
    adapter = ClarificationMultiJudgeAdapter()
    evaluator = BaseMultiJudgeEvaluator(
        adapter=adapter,
        voting_method=VotingMethod.CONFIDENCE_WEIGHTED,
        consensus_threshold=0.7
    )

    # Initialize agent
    agent = ClarificationAgent()

    # Test queries
    test_queries = [
        "Tell me about machine learning",
        "How can I optimize my website for better performance?",
        "What are the best practices for remote team management?"
    ]

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for query in test_queries:
            print(f"\n📝 Query: '{query}'")
            print("-" * 60)

            # Create dependencies
            state = ResearchState(
                request_id=f"multi-judge-demo-{abs(hash(query))}",
                user_query=query,
                current_stage=ResearchStage.CLARIFICATION,
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
                # Get agent output
                run_result = await agent.agent.run(query, deps=deps)
                output = run_result.output

                # Evaluate with multi-judge consensus
                consensus = await evaluator.evaluate(
                    input=query,
                    output=output,
                    context={"demo": True, "agent": "clarification"}
                )

                # Display results
                print(f"✅ Needs Clarification: {output.needs_clarification}")
                if output.request and output.request.questions:
                    print(f"   Generated {len(output.request.questions)} questions")

                print(f"\n🎯 Consensus Results:")
                print(f"   Final Score: {consensus.final_score:.3f}")
                print(f"   Consensus Reached: {consensus.consensus_reached}")
                print(f"   Agreement Score: {consensus.agreement_score:.3f}")
                print(f"   Voting Method: {consensus.voting_method.value}")

                print(f"\n📊 Dimension Scores:")
                for dim, score in consensus.dimension_scores.items():
                    print(f"   {dim}: {score:.2f}")

                print(f"\n👥 Judge Participation:")
                successful = sum(1 for j in consensus.judge_results if j.success)
                print(f"   {successful}/{len(consensus.judge_results)} judges successful")

            except Exception as e:
                print(f"❌ Error: {e}")


async def evaluate_query_transformation_agent():
    """Demonstrate multi-judge evaluation of QueryTransformationAgent."""
    print("\n" + "=" * 80)
    print("QUERY TRANSFORMATION AGENT - MULTI-JUDGE EVALUATION")
    print("=" * 80)

    # Create adapter and evaluator with custom judges
    adapter = QueryTransformationMultiJudgeAdapter()

    # Custom judge configuration for query transformation
    custom_judges = [
        JudgeConfiguration(
            model="openai:gpt-5",
            expertise=JudgeExpertise.SCIENTIFIC,  # Good for research methodology
            weight=1.3,
            temperature=0.1
        ),
        JudgeConfiguration(
            model="openai:gpt-5-mini",
            expertise=JudgeExpertise.TECHNICAL,
            weight=1.0,
            temperature=0.0
        )
    ]

    # Add Claude judge if API key is available
    if os.getenv("ANTHROPIC_API_KEY"):
        custom_judges.append(
            JudgeConfiguration(
                model="anthropic:claude-3-sonnet-20240229",
                expertise=JudgeExpertise.GENERAL,
                weight=1.2,
                temperature=0.1
            )
        )

    evaluator = BaseMultiJudgeEvaluator(
        adapter=adapter,
        judges=custom_judges,
        voting_method=VotingMethod.EXPERT_WEIGHTED,  # Use expert weighting
        consensus_threshold=0.75
    )

    # Initialize agent
    agent = QueryTransformationAgent()

    # Test queries
    test_queries = [
        "What are the latest advances in quantum computing?",
        "How does climate change affect global agriculture?",
        "Compare different JavaScript frameworks for web development"
    ]

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for query in test_queries:
            print(f"\n📝 Query: '{query}'")
            print("-" * 60)

            # Create dependencies
            state = ResearchState(
                request_id=f"multi-judge-demo-{abs(hash(query))}",
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
                # Get agent output
                run_result = await agent.agent.run(query, deps=deps)
                output: TransformedQuery = run_result.output

                # Evaluate with multi-judge consensus
                consensus = await evaluator.evaluate(
                    input=query,
                    output=output,
                    context={"demo": True, "agent": "query_transformation"}
                )

                # Display results
                print(f"✅ Transformation Complete:")
                print(f"   {len(output.research_plan.objectives)} objectives")
                print(f"   {len(output.search_queries.queries)} search queries")
                print(f"   Confidence: {output.confidence_score:.2f}")

                print(f"\n🎯 Consensus Results:")
                print(f"   Final Score: {consensus.final_score:.3f}")
                print(f"   Consensus Reached: {consensus.consensus_reached}")
                print(f"   Agreement Score: {consensus.agreement_score:.3f}")
                print(f"   Voting Method: {consensus.voting_method.value}")

                print(f"\n📊 Dimension Scores:")
                for dim, score in consensus.dimension_scores.items():
                    print(f"   {dim}: {score:.2f}")

                print(f"\n👥 Judge Participation:")
                successful = sum(1 for j in consensus.judge_results if j.success)
                print(f"   {successful}/{len(consensus.judge_results)} judges successful")

                # Show disagreement analysis
                if consensus.disagreement_analysis.get("dimension_variances"):
                    print(f"\n📈 Dimension Agreement Levels:")
                    for dim, agreement in consensus.disagreement_analysis.get("dimension_agreements", {}).items():
                        print(f"   {dim}: {agreement:.2%} agreement")

            except Exception as e:
                print(f"❌ Error: {e}")


async def compare_agent_outputs():
    """Demonstrate pairwise comparison between two outputs."""
    print("\n" + "=" * 80)
    print("PAIRWISE COMPARISON DEMO")
    print("=" * 80)

    query = "How can I improve my website's SEO?"

    # Create query transformation evaluator
    adapter = QueryTransformationMultiJudgeAdapter()
    evaluator = BaseMultiJudgeEvaluator(
        adapter=adapter,
        voting_method=VotingMethod.MAJORITY
    )

    # Initialize agent
    agent = QueryTransformationAgent()

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        # Create dependencies
        state = ResearchState(
            request_id="comparison-demo",
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
            print(f"📝 Query: '{query}'")
            print("\nGenerating two different transformation outputs...")

            # Get two outputs with different temperatures
            original_temp = agent.agent.model.temperature

            # First output (lower temperature)
            agent.agent.model.temperature = 0.3
            result_a = await agent.agent.run(query, deps=deps)
            output_a = result_a.output

            # Second output (higher temperature)
            agent.agent.model.temperature = 0.7
            result_b = await agent.agent.run(query, deps=deps)
            output_b = result_b.output

            # Restore original temperature
            agent.agent.model.temperature = original_temp

            print(f"\n📊 Output A: {len(output_a.search_queries.queries)} queries, confidence {output_a.confidence_score:.2f}")
            print(f"📊 Output B: {len(output_b.search_queries.queries)} queries, confidence {output_b.confidence_score:.2f}")

            # Compare outputs
            print("\n🔄 Running pairwise comparison...")
            comparison = await evaluator.compare_outputs(
                input=query,
                output_a=output_a,
                output_b=output_b,
                context={"comparison_type": "temperature_variation"}
            )

            print(f"\n🏆 Winner: {comparison['winner']}")
            print(f"   Confidence: {comparison['confidence']:.2%}")
            print(f"   Vote Breakdown: {comparison['vote_breakdown']}")

            print("\n👥 Individual Judge Decisions:")
            for judge_comp in comparison['judge_comparisons']:
                print(f"   {judge_comp['judge']}: {judge_comp['winner']} (confidence: {judge_comp['confidence']})")

        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Main function to run all demonstrations."""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: No OPENAI_API_KEY found.")
        print("   The multi-judge evaluation requires API access.")
        print("   Please set: export OPENAI_API_KEY='your-key-here'")
        return

    print("\n🚀 Starting Multi-Judge Evaluation Framework Demo")
    print("=" * 80)
    print("This demo shows how the generalized multi-judge framework")
    print("can evaluate different agent types with the same consensus system.")
    print("=" * 80)

    # Run evaluations for both agents
    await evaluate_clarification_agent()
    await evaluate_query_transformation_agent()

    # Run pairwise comparison
    await compare_agent_outputs()

    print("\n" + "=" * 80)
    print("✅ Multi-Judge Evaluation Demo Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. The same multi-judge framework works for different agent types")
    print("2. Agent-specific adapters define evaluation dimensions and prompts")
    print("3. Multiple voting methods provide flexibility in consensus")
    print("4. Pairwise comparison enables A/B testing of outputs")
    print("5. Disagreement analysis helps identify evaluation reliability")


if __name__ == "__main__":
    asyncio.run(main())
