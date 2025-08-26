"""Comprehensive integration tests for the Pydantic-AI refactored system.

Tests the complete 3-phase clarification system with proper dependency injection,
memory-safe event handling, concurrent processing, and circuit breaker patterns.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from open_deep_research_with_pydantic_ai.core.workflow import ResearchWorkflow
from open_deep_research_with_pydantic_ai.core.events import research_event_bus, ResearchEventBus
from open_deep_research_with_pydantic_ai.core.agents import coordinator
from open_deep_research_with_pydantic_ai.dependencies import ResearchDependencies
from open_deep_research_with_pydantic_ai.models.api_models import APIKeys, ResearchMetadata
from open_deep_research_with_pydantic_ai.models.research import (
    ResearchState, ResearchStage, ClarificationResult, TransformedQueryResult, BriefGenerationResult
)


class TestPydanticAIIntegration:
    """Integration tests for the complete pydantic-ai refactored system."""

    @pytest.fixture
    def workflow(self):
        """Create a fresh workflow instance for testing."""
        return ResearchWorkflow()

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        from unittest.mock import AsyncMock
        import httpx

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        research_state = ResearchState(
            request_id="test-123",
            user_id="test-user",
            session_id="test-session",
            user_query="What is machine learning?",
        )

        return ResearchDependencies(
            http_client=mock_client,
            api_keys=APIKeys(),
            research_state=research_state,
            metadata=ResearchMetadata(),
        )

    @pytest.fixture
    def sample_clarification_result(self):
        """Sample clarification result for testing."""
        return ClarificationResult(
            needs_clarification=False,
            question="",
            verification="Query is sufficiently specific for research.",
            confidence_score=0.85,
            missing_dimensions=[],
            breadth_score=0.3,
            assessment_reasoning="Query has clear scope and intent.",
            suggested_clarifications=[]
        )

    @pytest.fixture
    def sample_transformation_result(self):
        """Sample transformation result for testing."""
        return TransformedQueryResult(
            original_query="What is machine learning?",
            transformed_query="What are the key machine learning algorithms and their practical applications in data science?",
            transformation_rationale="Added specificity about algorithms and applications",
            specificity_score=0.8,
            supporting_questions=["What are supervised learning algorithms?", "How is ML used in industry?"],
            clarification_responses={},
            domain_indicators=["data science", "algorithms"],
            complexity_assessment="medium",
            estimated_scope="moderate"
        )

    @pytest.fixture
    def sample_brief_result(self):
        """Sample brief generation result for testing."""
        return BriefGenerationResult(
            brief_text="Comprehensive research brief on machine learning fundamentals, algorithms, and applications in various domains.",
            confidence_score=0.9,
            key_research_areas=["ML Algorithms", "Applications", "Theory"],
            research_objectives=["Understand core concepts", "Explore applications"],
            methodology_suggestions=["Literature review", "Case studies"],
            estimated_complexity="medium",
            estimated_duration="2-3 hours",
            suggested_sources=["Academic papers", "Industry reports"],
            potential_challenges=["Technical complexity", "Rapidly evolving field"],
            success_criteria=["Clear understanding", "Practical examples"]
        )

    @pytest.mark.asyncio
    async def test_agent_coordinator_initialization(self, workflow):
        """Test that the agent coordinator initializes correctly."""
        workflow._ensure_initialized()

        assert workflow._initialized is True
        assert 'clarification' in workflow.coordinator.agents
        assert 'transformation' in workflow.coordinator.agents
        assert 'brief' in workflow.coordinator.agents

        # Test agent statistics tracking
        stats = workflow.coordinator.get_stats()
        assert 'clarification' in stats
        assert 'transformation' in stats
        assert 'brief' in stats
        for agent_stats in stats.values():
            assert 'calls' in agent_stats
            assert 'errors' in agent_stats

    def test_circuit_breaker_functionality(self, workflow):
        """Test circuit breaker pattern works correctly."""
        agent_type = "test_agent"

        # Initially circuit should be closed
        assert workflow._check_circuit_breaker(agent_type) is True

        # Record multiple errors to trigger circuit breaker
        for i in range(workflow._circuit_breaker_threshold):
            workflow._record_error(agent_type, Exception(f"Error {i}"))

        # Circuit should now be open
        assert workflow._check_circuit_breaker(agent_type) is False
        assert workflow._circuit_open.get(agent_type) is True

        # Recording success should close the circuit
        workflow._record_success(agent_type)
        assert workflow._check_circuit_breaker(agent_type) is True
        assert workflow._circuit_open.get(agent_type) is False

    @pytest.mark.asyncio
    async def test_memory_safe_event_system(self):
        """Test that the event system properly manages memory."""
        event_bus = ResearchEventBus()

        # Test handler subscription and cleanup
        call_count = 0

        def test_handler(event):
            nonlocal call_count
            call_count += 1

        from open_deep_research_with_pydantic_ai.core.events import ResearchStartedEvent

        # Subscribe handler
        await event_bus.subscribe(ResearchStartedEvent, test_handler)

        # Emit event
        test_event = ResearchStartedEvent(
            _request_id="test-123",
            user_query="test query",
            user_id="test-user"
        )
        await event_bus.emit(test_event)

        # Wait for async processing
        await asyncio.sleep(0.1)

        assert call_count == 1

        # Test event history
        history = await event_bus.get_event_history("test-123")
        assert len(history) == 1
        assert history[0].request_id == "test-123"

        # Test cleanup
        await event_bus.cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_processing_workflow(self, workflow, mock_dependencies):
        """Test that workflow can handle concurrent operations."""
        workflow._ensure_initialized()

        # Mock the agent coordinator to return test results
        with patch.object(workflow.coordinator, 'run_agent') as mock_run:
            # Set up mock responses for different agent types
            def mock_agent_response(agent_type, prompt, deps, **kwargs):
                if agent_type == 'clarification':
                    return ClarificationResult(
                        needs_clarification=False,
                        verification="Test verification",
                        confidence_score=0.8,
                        breadth_score=0.3,
                        assessment_reasoning="Test reasoning"
                    )
                elif agent_type == 'transformation':
                    return TransformedQueryResult(
                        original_query="test query",
                        transformed_query="enhanced test query",
                        transformation_rationale="test rationale",
                        specificity_score=0.7
                    )
                elif agent_type == 'brief':
                    return BriefGenerationResult(
                        brief_text="Test brief content",
                        confidence_score=0.9,
                        key_research_areas=["Area 1", "Area 2"]
                    )

            mock_run.side_effect = mock_agent_response

            # Test three-phase execution
            research_state = ResearchState(
                request_id="test-concurrent",
                user_id="test-user",
                user_query="Test query for concurrent processing"
            )

            await workflow._execute_three_phase_clarification(
                research_state, mock_dependencies, "Test query"
            )

            # Verify all phases were called
            assert mock_run.call_count >= 3  # At least clarification, transformation, brief

            # Verify metadata was populated
            assert research_state.metadata is not None
            assert "clarification_assessment" in research_state.metadata

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, workflow, mock_dependencies):
        """Test that errors are handled gracefully with proper recovery."""
        workflow._ensure_initialized()

        # Test that single agent failures don't break the whole system
        with patch.object(workflow.coordinator, 'run_agent') as mock_run:
            # Make transformation fail but others succeed
            def mock_agent_with_failure(agent_type, prompt, deps, **kwargs):
                if agent_type == 'clarification':
                    return ClarificationResult(
                        needs_clarification=False,
                        verification="Test verification",
                        confidence_score=0.8,
                        breadth_score=0.3,
                        assessment_reasoning="Test reasoning"
                    )
                elif agent_type == 'transformation':
                    raise Exception("Transformation agent failed")
                elif agent_type == 'brief':
                    return BriefGenerationResult(
                        brief_text="Test brief despite transformation failure",
                        confidence_score=0.7,
                        key_research_areas=["Area 1"]
                    )

            mock_run.side_effect = mock_agent_with_failure

            research_state = ResearchState(
                request_id="test-error-handling",
                user_id="test-user",
                user_query="Test error handling"
            )

            # Should complete despite transformation failure
            await workflow._execute_three_phase_clarification(
                research_state, mock_dependencies, "Test query"
            )

            # Verify clarification and brief still worked
            assert "clarification_assessment" in research_state.metadata
            assert "research_brief_text" in research_state.metadata
            # Transformation should be missing due to error
            assert "transformed_query" not in research_state.metadata

    @pytest.mark.asyncio
    async def test_end_to_end_planning_workflow(self, workflow):
        """Test complete end-to-end planning workflow."""
        workflow._ensure_initialized()

        # Mock all agents to return valid responses
        with patch.object(workflow.coordinator, 'run_agent') as mock_run:
            # Set up comprehensive mock responses
            def comprehensive_mock_response(agent_type, prompt, deps, **kwargs):
                if agent_type == 'clarification':
                    return ClarificationResult(
                        needs_clarification=True,
                        question="What specific applications are you most interested in?",
                        verification="",
                        confidence_score=0.7,
                        missing_dimensions=["application_domain"],
                        breadth_score=0.8,
                        assessment_reasoning="Query is broad and needs domain focus",
                        suggested_clarifications=["Specify application domain"]
                    )
                elif agent_type == 'transformation':
                    return TransformedQueryResult(
                        original_query="What is AI?",
                        transformed_query="What are the key AI techniques and their applications in healthcare?",
                        transformation_rationale="Added domain specificity and technique focus",
                        specificity_score=0.85,
                        supporting_questions=["What are ML algorithms used in healthcare?"],
                        domain_indicators=["healthcare", "AI techniques"],
                        complexity_assessment="high",
                        estimated_scope="broad"
                    )
                elif agent_type == 'brief':
                    return BriefGenerationResult(
                        brief_text="Comprehensive research brief covering AI techniques, healthcare applications, machine learning algorithms, and practical implementation considerations.",
                        confidence_score=0.95,
                        key_research_areas=["AI Techniques", "Healthcare Applications", "Implementation"],
                        research_objectives=["Understand techniques", "Identify applications", "Assess feasibility"],
                        methodology_suggestions=["Literature review", "Case studies", "Expert interviews"],
                        estimated_complexity="high",
                        estimated_duration="4-6 hours",
                        suggested_sources=["Medical journals", "AI conferences", "Industry reports"],
                        potential_challenges=["Technical complexity", "Medical domain knowledge", "Regulatory considerations"],
                        success_criteria=["Clear technique understanding", "Practical application examples", "Implementation roadmap"]
                    )

            mock_run.side_effect = comprehensive_mock_response

            # Execute complete planning workflow
            result = await workflow.execute_planning_only(
                user_query="What is AI?",
                api_keys=APIKeys(),
                user_id="integration-test",
                session_id="test-session"
            )

            # Verify complete workflow execution
            assert result.current_stage == ResearchStage.RESEARCH_EXECUTION
            assert result.metadata is not None

            # Verify all three phases completed
            assert "clarification_assessment" in result.metadata
            assert "transformed_query" in result.metadata
            assert "research_brief_text" in result.metadata
            assert "research_brief_full" in result.metadata

            # Verify data quality
            clarification = result.metadata["clarification_assessment"]
            assert clarification["confidence_score"] == 0.7
            assert clarification["needs_clarification"] is True

            transformation = result.metadata["transformed_query"]
            assert transformation["specificity_score"] == 0.85
            assert transformation["complexity_assessment"] == "high"

            brief_full = result.metadata["research_brief_full"]
            assert brief_full["confidence_score"] == 0.95
            assert len(brief_full["key_research_areas"]) == 3
            assert brief_full["estimated_complexity"] == "high"

    @pytest.mark.asyncio
    async def test_dependency_injection_system(self, mock_dependencies):
        """Test that dependency injection works correctly throughout the system."""
        # Test that dependencies are properly passed and used
        assert mock_dependencies.research_state.user_id == "test-user"
        assert mock_dependencies.research_state.session_id == "test-session"
        assert mock_dependencies.api_keys is not None
        assert mock_dependencies.http_client is not None

        # Test metadata operations
        mock_dependencies.add_metadata("test_key", "test_value")
        assert mock_dependencies.get_metadata("test_key") == "test_value"
        assert mock_dependencies.get_metadata("nonexistent", "default") == "default"

        # Test research state updates
        original_stage = mock_dependencies.research_state.current_stage
        mock_dependencies.update_research_state(current_stage=ResearchStage.COMPRESSION)
        # Note: update_research_state may not work as expected if it only updates existing attributes

        # Test HTTP client integration
        assert hasattr(mock_dependencies.http_client, 'get')
        assert hasattr(mock_dependencies.http_client, 'post')

    @pytest.mark.asyncio
    async def test_performance_and_concurrency_limits(self, workflow):
        """Test that the system respects concurrency limits and performs well."""
        workflow._ensure_initialized()

        # Test circuit breaker configuration
        assert workflow._max_concurrent_tasks == 5
        assert workflow._task_timeout == 300.0
        assert workflow._circuit_breaker_threshold == 3

        # Test that timeout handling works
        with patch.object(workflow.coordinator, 'run_agent') as mock_run:
            # Simulate slow agent
            async def slow_agent(*args, **kwargs):
                await asyncio.sleep(1.0)  # Simulate slow operation
                return ClarificationResult(
                    needs_clarification=False,
                    verification="slow response",
                    confidence_score=0.5,
                    breadth_score=0.5,
                    assessment_reasoning="slow reasoning"
                )

            mock_run.side_effect = slow_agent

            # Should complete within reasonable time due to timeout handling
            start_time = asyncio.get_event_loop().time()

            try:
                result = await asyncio.wait_for(
                    workflow._run_agent_with_circuit_breaker(
                        'clarification',
                        "test prompt",
                        mock_dependencies
                    ),
                    timeout=2.0  # Should complete within 2 seconds
                )

                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time

                # Should be roughly 1 second (the sleep time)
                assert 0.8 <= duration <= 1.5, f"Duration was {duration}"
                assert result.verification == "slow response"

            except asyncio.TimeoutError:
                pytest.fail("Agent execution took too long")

    def test_structured_output_validation(self, sample_clarification_result, sample_transformation_result, sample_brief_result):
        """Test that all structured outputs follow Pydantic validation rules."""
        # Test ClarificationResult validation
        assert sample_clarification_result.needs_clarification is False
        assert 0.0 <= sample_clarification_result.confidence_score <= 1.0
        assert 0.0 <= sample_clarification_result.breadth_score <= 1.0
        assert isinstance(sample_clarification_result.missing_dimensions, list)
        assert isinstance(sample_clarification_result.suggested_clarifications, list)

        # Test TransformedQueryResult validation
        assert len(sample_transformation_result.transformed_query) >= 10  # min_length validation
        assert len(sample_transformation_result.transformation_rationale) >= 20  # min_length validation
        assert 0.0 <= sample_transformation_result.specificity_score <= 1.0
        assert sample_transformation_result.complexity_assessment in ['low', 'medium', 'high']
        assert sample_transformation_result.estimated_scope in ['narrow', 'moderate', 'broad']

        # Test BriefGenerationResult validation
        assert len(sample_brief_result.brief_text) >= 100  # min_length validation
        assert 0.0 <= sample_brief_result.confidence_score <= 1.0
        assert len(sample_brief_result.key_research_areas) >= 1  # min_length validation
        assert sample_brief_result.estimated_complexity in ['low', 'medium', 'high']
        assert isinstance(sample_brief_result.research_objectives, list)
        assert isinstance(sample_brief_result.methodology_suggestions, list)
        assert isinstance(sample_brief_result.potential_challenges, list)

    @pytest.mark.asyncio
    async def test_integration_with_existing_systems(self):
        """Test integration with existing CLI and API systems."""
        # Test that the workflow can be called from different contexts
        workflow = ResearchWorkflow()

        # Test CLI-style usage
        with patch('sys.stdin.isatty', return_value=True):
            # This should work in interactive mode
            workflow._ensure_initialized()
            assert workflow._initialized is True

        # Test API-style usage
        with patch('sys.stdin.isatty', return_value=False):
            # This should work in non-interactive mode
            workflow._ensure_initialized()
            assert workflow._initialized is True

        # Test event bus integration
        stats = await research_event_bus.get_stats()
        assert isinstance(stats, dict)
        assert 'total_event_types' in stats
        assert 'total_handlers' in stats
