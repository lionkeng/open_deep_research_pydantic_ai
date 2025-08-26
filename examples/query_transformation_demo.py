#!/usr/bin/env python3
"""
Demonstration of the Query Transformation functionality.

This example shows how the QueryTransformationAgent and ClarificationAgent
work together to transform broad queries into specific, actionable research questions.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import httpx
from pydantic_ai.usage import RunUsage

from open_deep_research_with_pydantic_ai.agents.base import ResearchDependencies
from open_deep_research_with_pydantic_ai.agents.clarification import ClarificationAgent
from open_deep_research_with_pydantic_ai.agents.query_transformation import QueryTransformationAgent
from open_deep_research_with_pydantic_ai.models.api_models import APIKeys, ResearchMetadata
from open_deep_research_with_pydantic_ai.models.research import ResearchState, TransformedQuery


async def demonstrate_transformation_workflow():
    """Demonstrate the complete transformation workflow."""

    print("=" * 60)
    print("QUERY TRANSFORMATION DEMONSTRATION")
    print("=" * 60)
    print()

    # Example 1: Direct transformation
    print("📋 Example 1: Direct Query Transformation")
    print("-" * 40)

    transformation_agent = QueryTransformationAgent()

    original_query = "climate change effects"
    print(f"Original Query: '{original_query}'")
    print()

    # Simulate clarification responses
    clarification_responses = {
        "What time period are you interested in?": "The last 20 years, 2004-2024",
        "Are you focusing on a specific region?": "Arctic regions, particularly Greenland",
        "What type of effects are you most interested in?": "Ice sheet melting and sea level rise",
        "What's your background level with this topic?": "Undergraduate environmental science student",
    }

    print("Clarification Responses:")
    for i, (question, answer) in enumerate(clarification_responses.items(), 1):
        print(f"  {i}. Q: {question}")
        print(f"     A: {answer}")
    print()

    # Transform the query (without actual LLM call, using fallback)
    try:
        transformed_query = await transformation_agent.transform_query(
            original_query=original_query,
            clarification_responses=clarification_responses,
            conversation_context=["Previous discussion about environmental impacts"],
        )

        print("🎯 Transformation Results:")
        print(f"Primary Question: {transformed_query.transformed_query}")
        print(f"Specificity Score: {transformed_query.specificity_score:.2f}/1.0")

        if transformed_query.supporting_questions:
            print("Supporting Questions:")
            for i, question in enumerate(transformed_query.supporting_questions, 1):
                print(f"  {i}. {question}")

        print(f"Rationale: {transformed_query.transformation_rationale}")

        if transformed_query.missing_dimensions:
            print(f"Missing Dimensions: {', '.join(transformed_query.missing_dimensions)}")

        print()

    except Exception as e:
        print(f"❌ Error in transformation: {e}")
        print(
            "Note: This demo runs without API keys - actual transformation requires valid credentials"
        )
        print()


async def demonstrate_clarification_integration():
    """Demonstrate integration with clarification agent."""

    print("📋 Example 2: Clarification + Transformation Integration")
    print("-" * 40)

    # Create mock dependencies
    async with httpx.AsyncClient() as http_client:
        deps = ResearchDependencies(
            http_client=http_client,
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="demo-123",
                user_id="demo-user",
                session_id="demo-session",
                user_query="artificial intelligence impact on jobs",
            ),
            metadata=ResearchMetadata(),
            usage=RunUsage(),
        )

        clarification_agent = ClarificationAgent()

        original_query = "artificial intelligence impact on jobs"
        print(f"Original Query: '{original_query}'")
        print()

        # Simulate clarification responses
        clarification_responses = {
            "What time period interests you?": "Next 10 years (2024-2034)",
            "Which industries or job sectors?": "Healthcare, finance, and manufacturing",
            "What type of impact analysis?": "Both job displacement and creation",
            "Any specific geographic focus?": "United States and European Union",
        }

        print("Processing Clarification Responses:")
        for i, (question, answer) in enumerate(clarification_responses.items(), 1):
            print(f"  {i}. {question}")
            print(f"     → {answer}")
        print()

        # Process with integrated workflow
        try:
            result = await clarification_agent.process_clarification_responses_with_transformation(
                original_query=original_query,
                clarification_responses=clarification_responses,
                deps=deps,
            )

            print("🔄 Integrated Transformation Results:")
            print(f"Transformed Query: {result.transformed_query}")
            print(f"Specificity Score: {result.specificity_score:.2f}")
            print(f"Method Used: {result.transformation_metadata.get('method', 'unknown')}")
            print()

            # Show research state updates
            print("📊 Research State Updates:")
            print(f"Clarified Query: {deps.research_state.clarified_query}")
            if "transformed_query" in deps.research_state.metadata:
                tq_data = deps.research_state.metadata["transformed_query"]
                print(f"Stored Specificity: {tq_data.get('specificity_score', 'N/A')}")
            print()

        except Exception as e:
            print(f"❌ Integration error: {e}")
            print("Note: This demo uses fallback transformation without API calls")
            print()


async def demonstrate_validation():
    """Demonstrate transformation validation."""

    print("📋 Example 3: Transformation Validation")
    print("-" * 40)

    transformation_agent = QueryTransformationAgent()

    # Create example transformations to validate
    transformations = [
        TransformedQuery(
            original_query="renewable energy",
            transformed_query="What are the economic impacts of solar and wind energy adoption on electricity costs in California from 2020-2024?",
            supporting_questions=[
                "How do renewable energy subsidies affect market prices?",
                "What are the job creation impacts in the renewable sector?",
            ],
            transformation_rationale="Added specificity for technology types, geographic region, timeframe, and impact metrics",
            specificity_score=0.9,
            clarification_responses={
                "What type of renewable energy?": "Solar and wind",
                "What location?": "California",
                "What timeframe?": "2020-2024",
                "What impact?": "Economic impacts on electricity costs",
            },
        ),
        TransformedQuery(
            original_query="machine learning",
            transformed_query="machine learning applications",  # Poor transformation
            supporting_questions=[],
            transformation_rationale="Basic expansion",
            specificity_score=0.3,
            clarification_responses={},
        ),
    ]

    print("Validating Transformations:")
    print()

    for i, transformation in enumerate(transformations, 1):
        print(f"Transformation #{i}:")
        print(f"  Original: {transformation.original_query}")
        print(f"  Transformed: {transformation.transformed_query}")
        print()

        # Validate quality
        validation = transformation_agent._basic_transformation_validation(transformation)

        print("  📊 Validation Scores:")
        for metric, score in validation["scores"].items():
            print(f"    {metric.replace('_', ' ').title()}: {score:.1f}/10")
        print(f"    Overall Score: {validation['overall_score']:.1f}/10")
        print()

        # Interpret results
        overall = validation["overall_score"]
        if overall >= 8:
            quality = "Excellent ✅"
        elif overall >= 6:
            quality = "Good 👍"
        elif overall >= 4:
            quality = "Fair ⚠️"
        else:
            quality = "Poor ❌"

        print(f"  Quality Assessment: {quality}")
        print("-" * 40)


async def demonstrate_edge_cases():
    """Demonstrate handling of edge cases."""

    print("📋 Example 4: Edge Cases and Error Handling")
    print("-" * 40)

    transformation_agent = QueryTransformationAgent()

    edge_cases = [
        {"name": "Empty clarifications", "query": "blockchain technology", "responses": {}},
        {
            "name": "Irrelevant responses",
            "query": "quantum computing",
            "responses": {
                "Do you like this topic?": "yes",
                "Any other questions?": "no",
                "Is this interesting?": "maybe",
            },
        },
        {
            "name": "Very specific query",
            "query": "How do LSTM neural networks with attention mechanisms perform on named entity recognition tasks using the CoNLL-2003 dataset?",
            "responses": {
                "Any additional constraints?": "None",
                "What evaluation metrics?": "F1 score and precision",
            },
        },
    ]

    for case in edge_cases:
        print(f"Case: {case['name']}")
        print(f"Query: {case['query']}")
        print("Responses:", case["responses"] if case["responses"] else "None")

        result = transformation_agent._create_fallback_transformation(
            case["query"], case["responses"]
        )

        print(f"Result: {result.transformed_query}")
        print(f"Score: {result.specificity_score:.2f}")
        print(f"Method: {result.transformation_metadata.get('method')}")
        print("-" * 30)


async def main():
    """Run all demonstrations."""
    try:
        await demonstrate_transformation_workflow()
        await demonstrate_clarification_integration()
        await demonstrate_validation()
        await demonstrate_edge_cases()

        print("✅ All demonstrations completed successfully!")
        print()
        print("💡 Key Takeaways:")
        print("   • Query transformation makes research more focused and actionable")
        print("   • Integration with clarification creates smooth workflow")
        print("   • Validation ensures transformation quality")
        print("   • Robust error handling provides reliable fallbacks")
        print()
        print("Note: This demo uses fallback transformations.")
        print("For full AI-powered transformation, provide valid API keys in the environment.")

    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("This is expected when running without proper API configuration.")


if __name__ == "__main__":
    print("Starting Query Transformation Demo...")
    print("Note: This demo will use fallback methods since no API keys are configured.")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("This is normal when running without API configuration.")
