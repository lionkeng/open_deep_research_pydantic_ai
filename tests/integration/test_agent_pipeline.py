"""
Integration tests for agent pipeline data flow and interactions.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from agents.base import ResearchDependencies
from agents.factory import AgentFactory, AgentType
from models.api_models import APIKeys
from models.core import ResearchStage, ResearchState
from models.metadata import ResearchMetadata
from models.report_generator import ReportSection, ResearchReport
from models.research_executor import ResearchFinding, ResearchResults, ResearchSource


class TestAgentPipelineIntegration:
    """Test integration between agents in the research pipeline."""

    @pytest_asyncio.fixture
    async def pipeline_dependencies(self):
        """Create dependencies for pipeline testing."""
        return ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="pipeline-test",
                user_id="test-user",
                session_id="test-session",
                user_query="Research the impact of AI on healthcare",
                current_stage=ResearchStage.CLARIFICATION,
                metadata=ResearchMetadata(),
            ),
            usage=None,
        )

    @pytest.mark.asyncio
    async def test_clarification_to_transformation_flow(self, pipeline_dependencies):
        """Test data flow from clarification to query transformation."""
        # Create agents
        clarification_agent = AgentFactory.create_agent(
            AgentType.CLARIFICATION, pipeline_dependencies
        )
        transformation_agent = AgentFactory.create_agent(
            AgentType.QUERY_TRANSFORMATION, pipeline_dependencies
        )

        # Mock clarification output
        clarification_result = MagicMock()
        clarification_result.clarification_needed = False
        clarification_result.transformed_query = (
            "AI applications in healthcare diagnostics and treatment"
        )
        clarification_result.confidence_score = 0.9

        with patch.object(clarification_agent, "run") as mock_run:
            mock_run.return_value = clarification_result
            # Execute clarification
            clarified = await clarification_agent.run(pipeline_dependencies)

            # Update state with clarification result
            pipeline_dependencies.research_state.metadata.transformed_query = {
                "query": clarified.transformed_query
            }

            # Mock transformation output
            transformation_result = MagicMock()
            transformation_result.original_query = clarified.transformed_query
            transformation_result.transformed_query = (
                "Comprehensive analysis of AI in healthcare focusing on diagnostics and treatment"
            )
            transformation_result.supporting_questions = [
                "How is AI used in medical imaging?",
                "What are AI treatment recommendations?",
            ]
            transformation_result.transformation_rationale = (
                "Expanded query for comprehensive research"
            )
            transformation_result.specificity_score = 0.85

            with patch.object(transformation_agent, "run") as mock_transform_run:
                mock_transform_run.return_value = transformation_result
                # Execute transformation
                transformed = await transformation_agent.run(pipeline_dependencies)

                # Verify data flow
                assert transformed.original_query == clarified.transformed_query
                assert transformed.specificity_score > 0.8


    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, pipeline_dependencies):
        """Test complete pipeline from clarification to report."""
        agents = {
            AgentType.CLARIFICATION: AgentFactory.create_agent(
                AgentType.CLARIFICATION, pipeline_dependencies
            ),
            AgentType.QUERY_TRANSFORMATION: AgentFactory.create_agent(
                AgentType.QUERY_TRANSFORMATION, pipeline_dependencies
            ),
            AgentType.RESEARCH_EXECUTOR: AgentFactory.create_agent(
                AgentType.RESEARCH_EXECUTOR, pipeline_dependencies
            ),
            AgentType.REPORT_GENERATOR: AgentFactory.create_agent(
                AgentType.REPORT_GENERATOR, pipeline_dependencies
            ),
        }

        # Track execution order
        execution_order = []

        async def mock_run(agent_type):
            execution_order.append(agent_type)
            await asyncio.sleep(0.01)  # Simulate work
            return MagicMock()

        # Mock all agent executions
        for agent_type, agent in agents.items():

            async def run_with_tracking(_deps, at=agent_type):
                return await mock_run(at)

            agent.run = run_with_tracking

        # Execute pipeline
        for agent_type in [
            AgentType.CLARIFICATION,
            AgentType.QUERY_TRANSFORMATION,
            AgentType.RESEARCH_EXECUTOR,
            AgentType.REPORT_GENERATOR,
        ]:
            await agents[agent_type].run(pipeline_dependencies)

        # Verify execution order
        assert len(execution_order) == 4

    @pytest.mark.asyncio
    async def test_pipeline_error_propagation(self, pipeline_dependencies):
        """Test error propagation through the pipeline."""
        # Create agents
        research_agent = AgentFactory.create_agent(
            AgentType.RESEARCH_EXECUTOR, pipeline_dependencies
        )
        report_agent = AgentFactory.create_agent(
            AgentType.REPORT_GENERATOR, pipeline_dependencies
        )

        # Mock research failure
        with patch.object(research_agent, "run", side_effect=Exception("Research failed")):
            with pytest.raises(Exception, match="Research failed"):
                await research_agent.run(pipeline_dependencies)

            # Report generation should handle missing data gracefully
            report_result = MagicMock()
            report_result.title = "Fallback Report"

            with patch.object(report_agent, "run", return_value=report_result):
                result = await report_agent.run(pipeline_dependencies)
                assert result.title == "Fallback Report"

    @pytest.mark.asyncio
    async def test_context_preservation_across_agents(self, pipeline_dependencies):
        """Test that context is preserved across agent boundaries."""
        # Initial context
        initial_context = {
            "user_preference": "technical_detail",
            "domain": "healthcare",
            "urgency": "high",
        }
        pipeline_dependencies.research_state.metadata.additional_context = initial_context.copy()

        # Create agents
        agents = [
            AgentFactory.create_agent(AgentType.CLARIFICATION, pipeline_dependencies),
            AgentFactory.create_agent(AgentType.QUERY_TRANSFORMATION, pipeline_dependencies),
        ]

        # Mock executions that preserve and add to context
        for i, agent in enumerate(agents):

            async def mock_run_with_context(deps, agent_num=i):
                # Verify initial context is preserved
                additional = deps.research_state.metadata.additional_context
                assert additional["user_preference"] == "technical_detail"
                assert additional["domain"] == "healthcare"

                # Add agent-specific context
                deps.research_state.metadata.additional_context[f"agent_{agent_num}_processed"] = True
                return MagicMock()

            with patch.object(agent, "run", side_effect=mock_run_with_context):
                await agent.run(pipeline_dependencies)

        # Verify all context is preserved
        additional = pipeline_dependencies.research_state.metadata.additional_context
        assert additional["user_preference"] == "technical_detail"
        assert all(
            f"agent_{i}_processed" in additional
            for i in range(2)
        )

    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self, pipeline_dependencies):
        """Test parallel execution of independent agents."""
        # Create multiple independent agents
        agents = [
            AgentFactory.create_agent(AgentType.RESEARCH_EXECUTOR, pipeline_dependencies),
            AgentFactory.create_agent(AgentType.RESEARCH_EXECUTOR, pipeline_dependencies),
            AgentFactory.create_agent(AgentType.RESEARCH_EXECUTOR, pipeline_dependencies),
        ]

        execution_times = []

        async def mock_run_with_delay(deps):
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)  # Simulate work
            execution_times.append(asyncio.get_event_loop().time() - start)
            return MagicMock()

        # Mock all agents
        for agent in agents:
            agent.run = mock_run_with_delay

        # Execute in parallel
        start = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[agent.run(pipeline_dependencies) for agent in agents])
        total_time = asyncio.get_event_loop().time() - start

        # Verify parallel execution (should be much faster than sequential)
        assert len(results) == 3
        assert total_time < 0.3  # Should be ~0.1s for parallel, would be ~0.3s for sequential
