#!/usr/bin/env python
"""Quick test to verify Research Executor evaluation framework works."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.evals.research_executor_evals import (
    FindingsRelevanceEvaluator,
    ComprehensiveEvaluator,
    MultiJudgeConsensusEvaluator,
    ResearchExecutorExpectedOutput
)
from models.research_executor import ResearchResults, ResearchFinding, ResearchSource
from pydantic_evals.evaluators import EvaluatorContext
from datetime import datetime, timezone


async def test_basic_evaluation():
    """Test that basic evaluation works."""

    # Create a sample research result
    result = ResearchResults(
        query="What is the capital of France?",
        findings=[
            ResearchFinding(
                finding="Paris is the capital of France",
                supporting_evidence=["According to official sources, Paris has been the capital since 987 AD"],
                confidence_level=0.95,
                category="geography"
            )
        ],
        sources=[
            ResearchSource(
                title="Geography Encyclopedia",
                url="https://example.com/geo",
                relevance_score=0.9,
                date_accessed=datetime.now(timezone.utc).isoformat()
            )
        ],
        key_insights=["Paris is a major European capital"],
        data_gaps=[],
        quality_score=0.85,
        execution_time=datetime.now(timezone.utc)
    )

    # Create evaluator context
    ctx = EvaluatorContext(
        name="test_basic",
        inputs={"query": "What is the capital of France?"},
        output=result,
        expected_output=ResearchExecutorExpectedOutput(
            min_findings=1,
            min_sources=1
        ),
        metadata=None,
        duration=1.0,
        _span_tree=None,
        attributes={},
        metrics={}
    )

    # Test with FindingsRelevanceEvaluator
    evaluator = FindingsRelevanceEvaluator()
    score = evaluator.evaluate(ctx)

    print(f"✓ Basic evaluation working - Score: {score:.2f}")
    return score


async def test_multi_judge():
    """Test multi-judge consensus evaluation."""

    result = ResearchResults(
        query="Explain quantum computing",
        findings=[
            ResearchFinding(
                finding="Quantum computing uses quantum bits (qubits) that can exist in superposition",
                supporting_evidence=["IBM Research: Qubits leverage quantum mechanical phenomena"],
                confidence_level=0.9,
                category="technology"
            ),
            ResearchFinding(
                finding="Quantum computers can solve certain problems exponentially faster than classical computers",
                supporting_evidence=["Shor's algorithm demonstrates exponential speedup for factoring"],
                confidence_level=0.85,
                category="technology"
            )
        ],
        sources=[
            ResearchSource(
                title="IBM Quantum Network",
                url="https://quantum.ibm.com",
                relevance_score=0.95,
                date_accessed=datetime.now(timezone.utc).isoformat()
            )
        ],
        key_insights=[
            "Quantum computing represents a paradigm shift in computation",
            "Current quantum computers are in the NISQ era"
        ],
        data_gaps=["Error correction methods still being developed"],
        quality_score=0.88,
        execution_time=datetime.now(timezone.utc)
    )

    # Create evaluator context
    ctx = EvaluatorContext(
        name="test_multi_judge",
        inputs={"query": "Explain quantum computing"},
        output=result,
        expected_output=ResearchExecutorExpectedOutput(
            min_findings=2,
            min_sources=1,
            expected_categories=["technology"]
        ),
        metadata=None,
        duration=1.0,
        _span_tree=None,
        attributes={},
        metrics={}
    )

    evaluator = MultiJudgeConsensusEvaluator()
    score = evaluator.evaluate(ctx)

    print(f"✓ Multi-judge evaluation working - Score: {score:.2f}")
    return score


async def test_empty_result():
    """Test evaluation with empty/minimal results."""

    result = ResearchResults(
        query="Test query",
        findings=[],
        sources=[],
        key_insights=[],
        data_gaps=["No data available"],
        quality_score=0.1,
        execution_time=datetime.now(timezone.utc)
    )

    # Create evaluator context
    ctx = EvaluatorContext(
        name="test_empty",
        inputs={"query": "Test query"},
        output=result,
        expected_output=ResearchExecutorExpectedOutput(),
        metadata=None,
        duration=1.0,
        _span_tree=None,
        attributes={},
        metrics={}
    )

    evaluator = ComprehensiveEvaluator()
    score = evaluator.evaluate(ctx)

    print(f"✓ Empty result evaluation working - Score: {score:.2f}")
    return score


async def main():
    """Run all quick tests."""
    print("Testing Research Executor Evaluation Framework...")
    print("-" * 50)

    try:
        # Test basic evaluation
        score1 = await test_basic_evaluation()

        # Test multi-judge (skip if no API key)
        import os
        if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
            score2 = await test_multi_judge()
        else:
            print("⚠ Skipping multi-judge test (no API keys found)")
            score2 = None

        # Test edge case
        score3 = await test_empty_result()

        print("-" * 50)
        print("✅ All tests passed!")
        score2_str = f"{score2:.2f}" if score2 is not None else "N/A"
        print(f"Scores: Basic={score1:.2f}, Multi-judge={score2_str}, Empty={score3:.2f}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
