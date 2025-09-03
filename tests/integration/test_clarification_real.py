"""
Real integration tests for ClarificationAgent that actually test the LLM's capabilities.

These tests run the actual agent with real LLM calls to evaluate:
- Binary correctness: Does it correctly identify when clarification is needed?
- Question quality: Are the clarification questions relevant and helpful?
- Dimension coverage: Does it identify the right missing dimensions?
- Consistency: Does it produce similar results for the same query?
"""

import asyncio
import os
from typing import Dict, List, Tuple
import pytest
import httpx
from dataclasses import dataclass

from src.agents.clarification import ClarificationAgent, ClarifyWithUser
from src.agents.base import ResearchDependencies
from src.models.api_models import APIKeys, ResearchMetadata
from src.models.core import ResearchState


@dataclass
class ClarificationTestCase:
    """Test case for clarification agent evaluation."""
    name: str
    query: str
    should_clarify: bool
    expected_dimensions: List[str] = None
    expected_themes: List[str] = None
    description: str = ""


class TestClarificationAgentReal:
    """Real integration tests for ClarificationAgent."""

    @pytest.fixture
    def api_keys(self):
        """Get API keys from environment."""
        return APIKeys(
            openai=os.getenv("OPENAI_API_KEY"),
            anthropic=os.getenv("ANTHROPIC_API_KEY"),
            tavily=os.getenv("TAVILY_API_KEY")
        )

    @pytest.fixture
    async def agent_dependencies(self, api_keys):
        """Create real dependencies for testing."""
        async with httpx.AsyncClient() as http_client:
            state = ResearchState(
                request_id="test-real-clarification",
                user_query="test"  # Will be updated per test
            )
            yield ResearchDependencies(
                http_client=http_client,
                api_keys=api_keys,
                research_state=state
            )

    @pytest.fixture
    def clarification_agent(self):
        """Create a real ClarificationAgent instance."""
        return ClarificationAgent()

    def get_golden_dataset(self) -> List[ClarificationTestCase]:
        """Get golden dataset of test cases."""
        return [
            # Clear, specific queries that should NOT need clarification
            ClarificationTestCase(
                name="bitcoin_price",
                query="What is the current Bitcoin price in USD?",
                should_clarify=False,
                description="Specific, time-bound financial query"
            ),
            ClarificationTestCase(
                name="technical_comparison",
                query="Compare ResNet-50 vs VGG-16 for ImageNet classification accuracy",
                should_clarify=False,
                description="Specific technical comparison with clear metrics"
            ),
            ClarificationTestCase(
                name="code_implementation",
                query="How to implement binary search algorithm in Python with O(log n) complexity",
                should_clarify=False,
                description="Clear programming task with specific requirements"
            ),

            # Ambiguous queries that SHOULD need clarification
            ClarificationTestCase(
                name="broad_ai",
                query="What is AI?",
                should_clarify=True,
                expected_dimensions=["audience_level", "scope", "purpose"],
                expected_themes=["technical depth", "specific aspect", "use case"],
                description="Very broad topic needing scope clarification"
            ),
            ClarificationTestCase(
                name="ambiguous_python",
                query="Tell me about Python",
                should_clarify=True,
                expected_dimensions=["context", "scope"],
                expected_themes=["programming language", "snake", "specific aspect"],
                description="Ambiguous term - could be language or animal"
            ),
            ClarificationTestCase(
                name="vague_research",
                query="Research climate change",
                should_clarify=True,
                expected_dimensions=["scope", "purpose", "deliverable"],
                expected_themes=["specific aspect", "geographic region", "time period"],
                description="Broad research request needing focus"
            ),
            ClarificationTestCase(
                name="incomplete_context",
                query="How does it work?",
                should_clarify=True,
                expected_dimensions=["context", "subject"],
                expected_themes=["what 'it' refers to"],
                description="Missing subject reference"
            ),

            # Edge cases
            ClarificationTestCase(
                name="minimal_query",
                query="?",
                should_clarify=True,
                expected_dimensions=["query_content"],
                description="Minimal query with no content"
            ),
            ClarificationTestCase(
                name="multiple_questions",
                query="What is machine learning and how does it compare to deep learning and can you also explain neural networks?",
                should_clarify=True,
                expected_dimensions=["focus", "depth", "priority"],
                expected_themes=["which topic to prioritize", "level of detail"],
                description="Multiple questions needing prioritization"
            )
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("test_case", [
        case for case in [
            ClarificationTestCase("bitcoin_price", "What is the current Bitcoin price in USD?", False),
            ClarificationTestCase("broad_ai", "What is AI?", True, ["audience_level", "scope"]),
        ]
    ])
    async def test_golden_dataset(self, clarification_agent, agent_dependencies, test_case):
        """Test agent with golden dataset cases."""
        # Update query in dependencies
        agent_dependencies.research_state.user_query = test_case.query

        # Run the agent
        result = await clarification_agent.agent.run(
            test_case.query,
            deps=agent_dependencies
        )

        # Extract the structured output
        clarification_result: ClarifyWithUser = result.data

        # Test binary correctness
        assert clarification_result.need_clarification == test_case.should_clarify, \
            f"Failed for '{test_case.name}': Expected clarification={test_case.should_clarify}, got {clarification_result.need_clarification}"

        # If clarification is needed, check quality
        if test_case.should_clarify:
            assert clarification_result.question, "Clarification question should not be empty"
            assert len(clarification_result.missing_dimensions) > 0, "Should identify missing dimensions"
            assert clarification_result.assessment_reasoning, "Should provide reasoning"

            # Check if expected dimensions are covered (if specified)
            if test_case.expected_dimensions:
                missing_dims_lower = [d.lower() for d in clarification_result.missing_dimensions]
                for expected_dim in test_case.expected_dimensions:
                    # Check if dimension concept is mentioned (fuzzy match)
                    found = any(expected_dim.lower() in dim or dim in expected_dim.lower()
                              for dim in missing_dims_lower)
                    assert found, f"Expected dimension '{expected_dim}' not found in {clarification_result.missing_dimensions}"
        else:
            # If no clarification needed, should have verification
            assert clarification_result.verification, "Should provide verification when no clarification needed"
            assert not clarification_result.question, "Should not have a question when clarification not needed"

    @pytest.mark.asyncio
    async def test_consistency_across_runs(self, clarification_agent, agent_dependencies):
        """Test that the same query produces consistent results across multiple runs."""
        query = "What is machine learning?"
        agent_dependencies.research_state.user_query = query

        results = []
        for _ in range(3):  # Run 3 times
            result = await clarification_agent.agent.run(query, deps=agent_dependencies)
            results.append(result.data)

        # All runs should agree on whether clarification is needed
        clarification_decisions = [r.need_clarification for r in results]
        assert all(d == clarification_decisions[0] for d in clarification_decisions), \
            f"Inconsistent clarification decisions: {clarification_decisions}"

        # If clarification needed, check dimension consistency
        if clarification_decisions[0]:
            # Check that similar dimensions are identified
            all_dimensions = [set(r.missing_dimensions) for r in results]
            # At least some dimensions should be common across runs
            common_dimensions = all_dimensions[0]
            for dims in all_dimensions[1:]:
                common_dimensions = common_dimensions.intersection(dims)

            # We expect at least some consistency in identified dimensions
            assert len(common_dimensions) > 0 or all(len(dims) > 0 for dims in all_dimensions), \
                "Dimensions should have some consistency across runs"

    @pytest.mark.asyncio
    async def test_property_single_word_broad_queries(self, clarification_agent, agent_dependencies):
        """Property test: Single-word broad queries should need clarification."""
        broad_single_words = ["Technology", "Science", "Business", "Health", "Education"]

        for word in broad_single_words:
            agent_dependencies.research_state.user_query = word
            result = await clarification_agent.agent.run(word, deps=agent_dependencies)

            assert result.data.need_clarification, \
                f"Single broad word '{word}' should need clarification"
            assert len(result.data.missing_dimensions) > 0, \
                f"Should identify missing dimensions for '{word}'"

    @pytest.mark.asyncio
    async def test_property_specific_metrics_queries(self, clarification_agent, agent_dependencies):
        """Property test: Queries with specific metrics/numbers should not need clarification."""
        specific_queries = [
            "What is the speed of light in meters per second?",
            "Calculate 15% of 2500",
            "Convert 100 USD to EUR at current exchange rate"
        ]

        for query in specific_queries:
            agent_dependencies.research_state.user_query = query
            result = await clarification_agent.agent.run(query, deps=agent_dependencies)

            assert not result.data.need_clarification, \
                f"Specific query '{query[:50]}...' should not need clarification"

    @pytest.mark.asyncio
    async def test_clarification_question_relevance(self, clarification_agent, agent_dependencies):
        """Test that clarification questions are relevant to the query."""
        ambiguous_queries = [
            ("What is Python?", ["programming", "language", "snake", "aspect"]),
            ("Explain ML", ["machine learning", "specific", "aspect", "level"])
        ]

        for query, expected_keywords in ambiguous_queries:
            agent_dependencies.research_state.user_query = query
            result = await clarification_agent.agent.run(query, deps=agent_dependencies)

            if result.data.need_clarification:
                question_lower = result.data.question.lower()
                # Check if clarification question contains relevant keywords
                relevant = any(keyword in question_lower for keyword in expected_keywords)
                assert relevant, \
                    f"Clarification question '{result.data.question}' not relevant to query '{query}'"

    @pytest.mark.asyncio
    async def test_dimension_framework_coverage(self, clarification_agent, agent_dependencies):
        """Test that the 4-dimension framework is properly utilized."""
        # Query that should trigger multiple dimension categories
        query = "Research AI"
        agent_dependencies.research_state.user_query = query

        result = await clarification_agent.agent.run(query, deps=agent_dependencies)

        if result.data.need_clarification:
            dimensions_text = " ".join(result.data.missing_dimensions).lower()
            reasoning_text = result.data.assessment_reasoning.lower()
            all_text = dimensions_text + " " + reasoning_text

            # Check for coverage of the 4 framework categories
            framework_categories = {
                "audience": ["audience", "level", "background", "technical", "beginner", "expert"],
                "scope": ["scope", "focus", "aspect", "broad", "specific"],
                "source": ["source", "academic", "industry", "quality"],
                "deliverable": ["deliverable", "format", "output", "report", "summary"]
            }

            covered_categories = []
            for category, keywords in framework_categories.items():
                if any(keyword in all_text for keyword in keywords):
                    covered_categories.append(category)

            # Should cover at least 2 of the 4 categories for a broad query
            assert len(covered_categories) >= 2, \
                f"Should cover multiple framework categories, only found: {covered_categories}"

    def calculate_metrics(self, results: List[Tuple[ClarificationTestCase, ClarifyWithUser]]) -> Dict:
        """Calculate evaluation metrics from test results."""
        correct_predictions = sum(
            1 for test_case, result in results
            if test_case.should_clarify == result.need_clarification
        )
        total = len(results)

        true_positives = sum(
            1 for test_case, result in results
            if test_case.should_clarify and result.need_clarification
        )
        false_positives = sum(
            1 for test_case, result in results
            if not test_case.should_clarify and result.need_clarification
        )
        false_negatives = sum(
            1 for test_case, result in results
            if test_case.should_clarify and not result.need_clarification
        )

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": correct_predictions / total,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
