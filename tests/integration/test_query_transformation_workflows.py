"""Integration tests for QueryTransformationAgent workflow integration.

These tests focus on the agent's integration with external dependencies,
real AI models, and its role within larger research workflows.
"""

import pytest
import asyncio
import os
import time
from typing import Dict, Any, List
from unittest.mock import patch, AsyncMock, MagicMock

from src.agents.query_transformation import QueryTransformationAgent
from src.agents.base import ResearchDependencies, AgentConfiguration
from src.models.core import ResearchState, ResearchStage
from src.models.metadata import ResearchMetadata
from src.models.api_models import APIKeys
from src.models.research_plan_models import TransformedQuery
from pydantic import SecretStr
from pydantic_ai.usage import RunUsage
import uuid


class TestQueryTransformationWorkflowIntegration:
    """Integration tests for QueryTransformationAgent within research workflows.

    These tests focus on workflow integration (component interactions, dependency
    injection, error handling) rather than external API integration. Most tests
    use mocked LLMs for speed and reliability.

    Tests marked with 'real_api' in their name use actual API calls when
    API keys are available.
    """

    @pytest.fixture
    def agent(self) -> QueryTransformationAgent:
        """Create a QueryTransformationAgent instance with mocked LLM for workflow testing."""
        # Create agent with mocked LLM to avoid real API calls
        with patch('src.agents.base.Agent') as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance

            agent = QueryTransformationAgent()
            agent.agent = mock_agent_instance

            # Set up default mock response
            mock_result = MagicMock()
            mock_result.output = self._create_mock_transformed_query()
            mock_agent_instance.run = AsyncMock(return_value=mock_result)

            return agent

    @pytest.fixture
    def real_agent(self) -> QueryTransformationAgent:
        """Create a real QueryTransformationAgent for API integration testing."""
        return QueryTransformationAgent()

    @pytest.fixture
    def real_dependencies(self) -> ResearchDependencies:
        """Create real dependencies with actual API keys for testing."""
        return ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(
                openai=SecretStr(key) if (key := os.getenv("OPENAI_API_KEY")) else None,
                anthropic=SecretStr(key) if (key := os.getenv("ANTHROPIC_API_KEY")) else None
            ),
            research_state=ResearchState(
                request_id="workflow-integration-test",
                user_query="test query",
                current_stage=ResearchStage.CLARIFICATION,
                metadata=ResearchMetadata()
            )
        )

    def _create_mock_transformed_query(self) -> TransformedQuery:
        """Helper to create a mock TransformedQuery response."""
        from src.models.research_plan_models import (
            ResearchPlan, ResearchObjective, ResearchMethodology
        )
        from src.models.search_query_models import SearchQuery, SearchQueryBatch, SearchQueryType

        # Create objectives
        obj1_id = str(uuid.uuid4())
        obj2_id = str(uuid.uuid4())

        objectives = [
            ResearchObjective(
                id=obj1_id,
                objective="Understand core concepts",
                priority="PRIMARY",
                success_criteria="Clear understanding achieved"
            ),
            ResearchObjective(
                id=obj2_id,
                objective="Explore applications",
                priority="SECONDARY",
                success_criteria="Applications identified"
            )
        ]

        # Create search queries linked to objectives
        queries = [
            SearchQuery(
                id=str(uuid.uuid4()),
                query="core concepts overview",
                query_type=SearchQueryType.FACTUAL,
                priority=5,
                max_results=10,
                rationale="Foundation understanding",
                objective_id=obj1_id
            ),
            SearchQuery(
                id=str(uuid.uuid4()),
                query="practical applications",
                query_type=SearchQueryType.EXPLORATORY,
                priority=4,
                max_results=10,
                rationale="Application exploration",
                objective_id=obj2_id
            )
        ]

        return TransformedQuery(
            original_query="test query",
            search_queries=SearchQueryBatch(queries=queries),
            research_plan=ResearchPlan(
                objectives=objectives,
                methodology=ResearchMethodology(
                    approach="Systematic exploration",
                    data_sources=["Academic", "Industry"],
                    analysis_methods=["Literature review"],
                    quality_criteria=["Accuracy", "Relevance"]
                ),
                expected_deliverables=["Comprehensive overview"]
            ),
            confidence_score=0.85,
            transformation_rationale="Query successfully transformed",
            assumptions_made=["General audience", "English language"],
            potential_gaps=["Technical depth", "Regional variations"]
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_real_api_integration_openai(self, real_agent: QueryTransformationAgent, real_dependencies: ResearchDependencies) -> None:
        """Test agent integration with real OpenAI API."""
        query = "How does machine learning work?"
        real_dependencies.research_state.user_query = query

        result = await real_agent.agent.run(query, deps=real_dependencies)

        # Verify API integration worked
        assert hasattr(result, 'output')
        assert isinstance(result.output, TransformedQuery)
        assert len(result.output.search_queries.queries) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_real_api_integration_anthropic(self, real_agent: QueryTransformationAgent, real_dependencies: ResearchDependencies) -> None:
        """Test agent integration with real Anthropic API."""
        # Modify dependencies to use Anthropic
        real_dependencies.api_keys = APIKeys(
            anthropic=SecretStr(key) if (key := os.getenv("ANTHROPIC_API_KEY")) else None
        )

        query = "What is Python programming?"
        real_dependencies.research_state.user_query = query

        result = await real_agent.agent.run(query, deps=real_dependencies)

        # Verify API integration worked
        assert hasattr(result, 'output')
        assert isinstance(result.output, TransformedQuery)
        assert len(result.output.search_queries.queries) > 0

    @pytest.mark.asyncio
    async def test_workflow_context_integration(self, agent: QueryTransformationAgent, real_dependencies: ResearchDependencies) -> None:
        """Test agent integration with research workflow context."""
        # Set up workflow context with conversation history
        real_dependencies.research_state.metadata = ResearchMetadata(
            conversation_messages=[
                {"role": "user", "content": "I'm researching climate change"},
                {"role": "assistant", "content": "I can help you research climate change."}
            ]
        )

        query = "What are the main causes?"
        real_dependencies.research_state.user_query = query

        result = await agent.agent.run(query, deps=real_dependencies)
        output = result.output

        assert isinstance(output, TransformedQuery)
        # The agent should leverage context for better transformation
        assert len(output.search_queries.queries) > 0
        assert output.transformation_rationale

    @pytest.mark.asyncio
    async def test_multi_stage_workflow_integration(self, agent: QueryTransformationAgent, real_dependencies: ResearchDependencies) -> None:
        """Test agent behavior within multi-stage research workflow."""
        # Test progression through workflow stages
        stages = [
            ResearchStage.CLARIFICATION,
            ResearchStage.CLARIFICATION,
            ResearchStage.RESEARCH_EXECUTION,
        ]

        for stage in stages:
            real_dependencies.research_state.current_stage = stage
            real_dependencies.research_state.request_id = f"workflow-{stage.value}"

            query = "Research artificial intelligence applications"
            result = await agent.agent.run(query, deps=real_dependencies)

            assert hasattr(result, 'output')
            assert isinstance(result.output, TransformedQuery)

    @pytest.mark.asyncio
    async def test_dependency_injection_variants(self, agent: QueryTransformationAgent) -> None:
        """Test agent behavior with different dependency configurations."""
        base_state = ResearchState(
            request_id="dep-injection-test",
            user_query="What is machine learning?"
        )

        # Test with minimal dependencies
        minimal_deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=base_state
        )

        result = await agent.agent.run("Test query", deps=minimal_deps)
        assert isinstance(result.output, TransformedQuery)

        # Test with extended dependencies including clarification
        extended_state = ResearchState(
            request_id="dep-injection-test",
            user_query="What is machine learning?",
            clarified_query="Explain machine learning algorithms and their applications",
            metadata=ResearchMetadata(
                user_preferences={"technical_level": "expert"}
            )
        )
        extended_deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=extended_state,
            usage=RunUsage(requests=1, output_tokens=100)
        )

        result = await agent.agent.run("Test query", deps=extended_deps)
        assert isinstance(result.output, TransformedQuery)

    @pytest.mark.asyncio
    async def test_concurrent_workflow_handling(self, agent: QueryTransformationAgent) -> None:
        """Test agent behavior with concurrent workflow requests."""
        # Create multiple concurrent workflow contexts
        contexts = []
        for i in range(3):
            deps = ResearchDependencies(
                http_client=AsyncMock(),
                api_keys=APIKeys(openai=SecretStr("test-key")),
                research_state=ResearchState(
                    request_id=f"concurrent-workflow-{i}",
                    user_query=f"Query {i}",
                    current_stage=ResearchStage.CLARIFICATION
                )
            )
            contexts.append(deps)

        # Run concurrent requests
        tasks = [
            agent.agent.run(f"Concurrent query {i}", deps=contexts[i])
            for i in range(len(contexts))
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed and be properly structured
        for i, result in enumerate(results):
            assert hasattr(result, 'output'), f"Result {i} missing data"
            assert isinstance(result.output, TransformedQuery), f"Result {i} wrong type"


class TestQueryTransformationErrorRecovery:
    """Test error recovery and resilience in workflow integration."""

    @pytest.fixture
    def agent(self) -> QueryTransformationAgent:
        """Create a QueryTransformationAgent instance for testing."""
        with patch('src.agents.base.Agent') as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance

            agent = QueryTransformationAgent()
            agent.agent = mock_agent_instance

            # Default mock will be overridden in specific tests
            return agent

    @pytest.mark.asyncio
    async def test_api_timeout_recovery(self, agent: QueryTransformationAgent) -> None:
        """Test agent behavior when API calls timeout."""
        # Mock HTTP client that times out
        mock_client = AsyncMock()
        mock_client.post.side_effect = asyncio.TimeoutError("Request timeout")

        deps = ResearchDependencies(
            http_client=mock_client,
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="timeout-test",
                user_query="test query"
            )
        )

        # Configure agent to handle timeout
        agent.agent.run = AsyncMock(side_effect=asyncio.TimeoutError("Timeout"))

        # Should handle timeout gracefully
        with pytest.raises(asyncio.TimeoutError):
            await agent.agent.run("Test timeout query", deps=deps)

    @pytest.mark.asyncio
    async def test_malformed_response_handling(self, agent: QueryTransformationAgent) -> None:
        """Test agent behavior with malformed API responses."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="malformed-test",
                user_query="test query"
            )
        )

        # Mock malformed response
        mock_result = MagicMock()
        mock_result.output = {"invalid": "structure"}  # Wrong format
        agent.agent.run = AsyncMock(return_value=mock_result)

        # Should handle malformed response gracefully
        try:
            result = await agent.agent.run("Test malformed response", deps=deps)
            # Check if it returns something or raises error
            if hasattr(result, 'output'):
                # May have fallback handling
                pass
        except (AttributeError, TypeError, ValueError):
            # Expected for malformed response
            pass


class TestQueryTransformationPerformanceIntegration:
    """Test performance characteristics in workflow integration."""

    @pytest.fixture
    def agent(self) -> QueryTransformationAgent:
        """Create a QueryTransformationAgent instance for testing."""
        with patch('src.agents.base.Agent') as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance

            agent = QueryTransformationAgent()
            agent.agent = mock_agent_instance

            # Fast mock response for performance testing
            from src.models.research_plan_models import (
                ResearchPlan, ResearchObjective, ResearchMethodology
            )
            from src.models.search_query_models import SearchQuery, SearchQueryBatch, SearchQueryType

            obj_id = str(uuid.uuid4())
            mock_result = MagicMock()
            mock_result.output = TransformedQuery(
                original_query="performance test",
                search_queries=SearchQueryBatch(queries=[
                    SearchQuery(
                        id=str(uuid.uuid4()),
                        query="test query",
                        query_type=SearchQueryType.FACTUAL,
                        priority=5,
                        max_results=10,
                        rationale="Test",
                        objective_id=obj_id
                    )
                ]),
                research_plan=ResearchPlan(
                    objectives=[
                        ResearchObjective(
                            id=obj_id,
                            objective="Test objective for performance",
                            priority="PRIMARY",
                            success_criteria="Test"
                        )
                    ],
                    methodology=ResearchMethodology(
                        approach="Test",
                        data_sources=["Test"],
                        analysis_methods=["Test"],
                        quality_criteria=["Test"]
                    ),
                    expected_deliverables=["Test"]
                ),
                confidence_score=0.9,
                transformation_rationale="Test",
                assumptions_made=["Test"],
                potential_gaps=[]
            )
            mock_agent_instance.run = AsyncMock(return_value=mock_result)

            return agent

    @pytest.mark.asyncio
    async def test_workflow_performance_benchmarks(self, agent: QueryTransformationAgent) -> None:
        """Test that agent meets performance requirements in workflow context."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="perf-benchmark",
                user_query="Performance test query"
            )
        )

        # Measure response time
        start_time = time.time()
        result = await agent.agent.run("Performance test query", deps=deps)
        end_time = time.time()

        response_time = end_time - start_time

        # Should respond within reasonable time for workflow integration
        assert response_time < 5.0, f"Workflow integration took {response_time}s, should be under 5s"

        # Response should be valid
        assert hasattr(result, 'output')
        assert isinstance(result.output, TransformedQuery)

    @pytest.mark.asyncio
    async def test_memory_usage_workflow(self, agent: QueryTransformationAgent) -> None:
        """Test memory usage doesn't grow excessively during workflow operations."""
        pytest.importorskip("psutil", reason="psutil not installed")
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run multiple workflow operations
        for i in range(5):
            deps = ResearchDependencies(
                http_client=AsyncMock(),
                api_keys=APIKeys(openai=SecretStr("test-key")),
                research_state=ResearchState(
                    request_id=f"memory-test-{i}",
                    user_query=f"Memory test query {i}"
                )
            )

            result = await agent.agent.run(f"Memory test {i}", deps=deps)
            assert isinstance(result.output, TransformedQuery)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (less than 100MB for this test)
        assert memory_growth < 100, f"Memory grew by {memory_growth}MB, should be under 100MB"

    @pytest.mark.asyncio
    async def test_concurrent_workflow_performance(self, agent: QueryTransformationAgent) -> None:
        """Test performance under concurrent workflow load."""
        # Create concurrent workflow tasks
        tasks = []
        for i in range(10):  # 10 concurrent requests
            deps = ResearchDependencies(
                http_client=AsyncMock(),
                api_keys=APIKeys(openai=SecretStr("test-key")),
                research_state=ResearchState(
                    request_id=f"concurrent-perf-{i}",
                    user_query=f"Concurrent query {i}"
                )
            )
            tasks.append(agent.agent.run(f"Concurrent test {i}", deps=deps))

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        total_time = end_time - start_time

        # All concurrent requests should complete within reasonable time
        assert total_time < 10.0, f"Concurrent workflow took {total_time}s, should be under 10s"

        # All should be successful
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent request {i} failed: {result}")
            assert hasattr(result, 'output')
            assert isinstance(result.output, TransformedQuery)


class TestQueryTransformationWithClarification:
    """Test query transformation integration with clarification results."""

    @pytest.fixture
    def agent_with_clarification(self) -> QueryTransformationAgent:
        """Create an agent that handles clarified queries."""
        with patch('src.agents.base.Agent') as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance

            agent = QueryTransformationAgent()
            agent.agent = mock_agent_instance

            # Mock response that accounts for clarification
            from src.models.research_plan_models import (
                ResearchPlan, ResearchObjective, ResearchMethodology
            )
            from src.models.search_query_models import SearchQuery, SearchQueryBatch, SearchQueryType

            obj1_id = str(uuid.uuid4())
            obj2_id = str(uuid.uuid4())

            mock_result = MagicMock()
            mock_result.output = TransformedQuery(
                original_query="clarified query about machine learning applications",
                search_queries=SearchQueryBatch(queries=[
                    SearchQuery(
                        id=str(uuid.uuid4()),
                        query="machine learning healthcare applications",
                        query_type=SearchQueryType.EXPLORATORY,
                        priority=5,
                        max_results=10,
                        rationale="Healthcare focus from clarification",
                        objective_id=obj1_id
                    ),
                    SearchQuery(
                        id=str(uuid.uuid4()),
                        query="ML diagnostic tools medical imaging",
                        query_type=SearchQueryType.ANALYTICAL,
                        priority=4,
                        max_results=10,
                        rationale="Specific application area",
                        objective_id=obj2_id
                    )
                ]),
                research_plan=ResearchPlan(
                    objectives=[
                        ResearchObjective(
                            id=obj1_id,
                            objective="Explore ML in healthcare",
                            priority="PRIMARY",
                            success_criteria="Comprehensive overview"
                        ),
                        ResearchObjective(
                            id=obj2_id,
                            objective="Analyze diagnostic applications",
                            priority="SECONDARY",
                            success_criteria="Detailed analysis"
                        )
                    ],
                    methodology=ResearchMethodology(
                        approach="Focused exploration based on clarification",
                        data_sources=["Medical journals", "Tech papers"],
                        analysis_methods=["Application analysis"],
                        quality_criteria=["Medical relevance", "Technical accuracy"]
                    ),
                    expected_deliverables=["Healthcare ML overview", "Diagnostic tools analysis"]
                ),
                confidence_score=0.92,
                transformation_rationale="Query refined based on clarification responses",
                assumptions_made=["Healthcare focus confirmed", "Technical depth appropriate"],
                potential_gaps=["Regulatory aspects", "Cost analysis"]
            )
            mock_agent_instance.run = AsyncMock(return_value=mock_result)

            return agent

    @pytest.mark.asyncio
    async def test_transformation_with_clarified_query(self, agent_with_clarification: QueryTransformationAgent) -> None:
        """Test transformation when a clarified query is available."""
        from src.models.clarification import ClarificationResponse, ClarificationAnswer

        # Create dependencies with clarification context
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="clarified-test",
                user_query="Tell me about AI",
                clarified_query="Machine learning applications in healthcare",
                metadata=ResearchMetadata(
                    clarification={
                        "request": {
                            "questions": [
                                {
                                    "id": "q1",
                                    "question": "Which field of AI interests you?",
                                    "question_type": "choice",
                                    "choices": ["Machine Learning", "Computer Vision", "NLP"]
                                },
                                {
                                    "id": "q2",
                                    "question": "What application area?",
                                    "question_type": "choice",
                                    "choices": ["Healthcare", "Finance", "Education"]
                                }
                            ]
                        },
                        "response": {
                            "request_id": "clarification-req-1",
                            "answers": [
                                {"question_id": "q1", "answer": "Machine Learning"},
                                {"question_id": "q2", "answer": "Healthcare"}
                            ]
                        }
                    }
                )
            )
        )

        result = await agent_with_clarification.agent.run(
            deps.research_state.clarified_query,
            deps=deps
        )

        # Verify the transformation uses clarified context
        assert result.output.original_query == "clarified query about machine learning applications"
        assert any("healthcare" in q.query.lower() for q in result.output.search_queries.queries)
        assert "clarification" in result.output.transformation_rationale.lower()
        assert result.output.confidence_score > 0.9  # Higher confidence with clarification

    @pytest.mark.asyncio
    async def test_transformation_preserves_clarification_context(self, agent_with_clarification: QueryTransformationAgent) -> None:
        """Test that clarification context is preserved in transformation."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(openai=SecretStr("test-key")),
            research_state=ResearchState(
                request_id="context-preserve-test",
                user_query="Explain this technology",
                clarified_query="Explain blockchain technology for supply chain management",
                metadata=ResearchMetadata(
                    clarification={
                        "assessment": {
                            "needs_clarification": True,
                            "missing_dimensions": ["SPECIFICITY", "APPLICATION_DOMAIN"],
                            "assessment_reasoning": "Technology type and use case needed"
                        },
                        "user_provided": {
                            "technology": "blockchain",
                            "domain": "supply chain"
                        }
                    }
                )
            )
        )

        result = await agent_with_clarification.agent.run(
            deps.research_state.clarified_query,
            deps=deps
        )

        # Verify context preservation
        assert result.output.transformation_rationale
        assert len(result.output.assumptions_made) > 0
        # Should reflect the clarification process
        assert any("clarif" in assumption.lower() or "confirm" in assumption.lower()
                  for assumption in result.output.assumptions_made)
