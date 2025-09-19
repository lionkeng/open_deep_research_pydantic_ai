#!/usr/bin/env python
"""Quick tests to verify the Research Executor evaluation framework."""

import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic_evals.evaluators import EvaluatorContext

from models.research_executor import HierarchicalFinding, ResearchResults, ResearchSource
from tests.evals.research_executor_evals import (
    ComprehensiveEvaluator,
    FindingsRelevanceEvaluator,
    MultiJudgeConsensusEvaluator,
    ResearchExecutorExpectedOutput,
)


@pytest.mark.asyncio
async def test_basic_evaluation() -> float:
    """Test that the findings relevance evaluator produces a score."""

    result = ResearchResults(
        query="What is the capital of France?",
        findings=[
            HierarchicalFinding(
                finding="Paris is the capital of France",
                supporting_evidence=[
                    "According to official sources, Paris has been the capital since 987 AD"
                ],
                confidence_score=0.95,
                category="geography",
            )
        ],
        sources=[
            ResearchSource(
                title="Geography Encyclopedia",
                url="https://example.com/geo",
                relevance_score=0.9,
                date_accessed=datetime.now(UTC).isoformat(),
            )
        ],
        key_insights=["Paris is a major European capital"],
        data_gaps=[],
        overall_quality_score=0.85,
        execution_time=datetime.now(UTC),
    )

    ctx = EvaluatorContext(
        name="test_basic",
        inputs={"query": "What is the capital of France?"},
        output=result,
        expected_output=ResearchExecutorExpectedOutput(min_findings=1, min_sources=1),
        metadata=None,
        duration=1.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )

    evaluator = FindingsRelevanceEvaluator()
    score = evaluator.evaluate(ctx)
    assert score >= 0.0
    return score


@pytest.mark.asyncio
async def test_multi_judge() -> float:
    """Test multi-judge consensus evaluation."""

    result = ResearchResults(
        query="Explain quantum computing",
        findings=[
            HierarchicalFinding(
                finding="Quantum computing uses quantum bits (qubits) that can exist in superposition",
                supporting_evidence=["IBM Research: Qubits leverage quantum mechanical phenomena"],
                confidence_score=0.9,
                category="technology",
            ),
            HierarchicalFinding(
                finding="Quantum computers can solve certain problems exponentially faster than classical computers",
                supporting_evidence=[
                    "Shor's algorithm demonstrates exponential speedup for factoring"
                ],
                confidence_score=0.85,
                category="technology",
            ),
        ],
        sources=[
            ResearchSource(
                title="IBM Quantum Network",
                url="https://quantum.ibm.com",
                relevance_score=0.95,
                date_accessed=datetime.now(UTC).isoformat(),
            )
        ],
        key_insights=[
            "Quantum computing represents a paradigm shift in computation",
            "Current quantum computers are in the NISQ era",
        ],
        data_gaps=["Error correction methods still being developed"],
        overall_quality_score=0.88,
        execution_time=datetime.now(UTC),
    )

    ctx = EvaluatorContext(
        name="test_multi_judge",
        inputs={"query": "Explain quantum computing"},
        output=result,
        expected_output=ResearchExecutorExpectedOutput(
            min_findings=2, min_sources=1, expected_categories=["technology"]
        ),
        metadata=None,
        duration=1.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )

    evaluator = MultiJudgeConsensusEvaluator()
    score = evaluator.evaluate(ctx)
    assert score >= 0.0
    return score


@pytest.mark.asyncio
async def test_empty_result() -> float:
    """Test evaluation with empty/minimal results."""

    result = ResearchResults(
        query="Test query",
        findings=[],
        sources=[],
        key_insights=[],
        data_gaps=["No data available"],
        overall_quality_score=0.1,
        execution_time=datetime.now(UTC),
    )

    ctx = EvaluatorContext(
        name="test_empty",
        inputs={"query": "Test query"},
        output=result,
        expected_output=ResearchExecutorExpectedOutput(),
        metadata=None,
        duration=1.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )

    evaluator = ComprehensiveEvaluator()
    score = evaluator.evaluate(ctx)
    assert score >= 0.0
    return score


async def main() -> None:
    """Allow running the quick checks as a script."""
    print("Testing Research Executor Evaluation Framework...")
    print("-" * 50)

    score1 = await test_basic_evaluation()

    if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        score2 = await test_multi_judge()
    else:
        print("⚠ Skipping multi-judge test (no API keys found)")
        score2 = None

    score3 = await test_empty_result()

    print("-" * 50)
    print("✅ Quick evaluation checks completed")
    score2_str = f"{score2:.2f}" if score2 is not None else "N/A"
    print(f"Scores: Basic={score1:.2f}, Multi-judge={score2_str}, Empty={score3:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
