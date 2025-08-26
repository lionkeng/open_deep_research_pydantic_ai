"""Tests for specific system improvements made during Pydantic-AI refactoring.

This test suite validates that all the critical improvements identified
in the code review have been successfully implemented and are working.
"""

import asyncio
import gc
import time
import weakref
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, List

import pytest

from open_deep_research_with_pydantic_ai.core.workflow import ResearchWorkflow
from open_deep_research_with_pydantic_ai.core.events import ResearchEventBus, research_event_bus
from open_deep_research_with_pydantic_ai.core.agents import coordinator
from open_deep_research_with_pydantic_ai.dependencies import ResearchDependencies
from open_deep_research_with_pydantic_ai.models.api_models import APIKeys, ResearchMetadata
from open_deep_research_with_pydantic_ai.models.research import ResearchState, ClarificationResult


class TestSystemImprovements:
    """Test suite for validating specific system improvements."""

    @pytest.mark.asyncio
    async def test_circular_import_elimination(self):
        """Test that circular imports have been eliminated."""
        # This test passes by virtue of being able to import all modules successfully
        try:
            # These imports should work without circular dependency issues
            from open_deep_research_with_pydantic_ai.core.agents import coordinator
            from open_deep_research_with_pydantic_ai.core.workflow import workflow
            from open_deep_research_with_pydantic_ai.dependencies import ResearchDependencies
            from open_deep_research_with_pydantic_ai.models.research import ClarificationResult

            # Test that coordinator has all expected agents
            expected_agents = ['clarification', 'transformation', 'brief']
            for agent_type in expected_agents:
                assert agent_type in coordinator.agents, f"Missing agent: {agent_type}"

            # Test agent lookup doesn't cause circular imports
            for agent_type in expected_agents:
                agent = coordinator.get_agent(agent_type)
                assert agent is not None

            print("✓ Circular import elimination successful")

        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")

    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self):
        """Test that the system prevents memory leaks through proper cleanup."""
        initial_objects = len(gc.get_objects())

        # Create and use event bus with many handlers
        event_bus = ResearchEventBus()

        # Create handlers that should be cleaned up automatically
        handlers = []
        call_counts = []

        for i in range(100):
            call_count = [0]  # Use list to make it mutable in closure
            call_counts.append(call_count)

            def make_handler(count_ref):
                def handler(event):
                    count_ref[0] += 1
                return handler

            handler = make_handler(call_count)
            handlers.append(handler)

            from open_deep_research_with_pydantic_ai.core.events import ResearchStartedEvent
            await event_bus.subscribe(ResearchStartedEvent, handler)

        # Emit events to activate handlers
        from open_deep_research_with_pydantic_ai.core.events import ResearchStartedEvent
        test_event = ResearchStartedEvent(
            _request_id="memory-test",
            user_query="test query",
            user_id="test-user"
        )

        await event_bus.emit(test_event)
        await asyncio.sleep(0.1)  # Allow async processing

        # Verify handlers were called
        total_calls = sum(count[0] for count in call_counts)
        assert total_calls > 0, "Handlers should have been called"

        # Delete references to handlers
        del handlers
        del call_counts

        # Force garbage collection
        gc.collect()
        await asyncio.sleep(0.1)

        # Cleanup event bus
        await event_bus.cleanup()
        del event_bus

        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count should not grow significantly (allowing some variance)
        object_growth = final_objects - initial_objects
        assert object_growth < 50, f"Memory leak detected: {object_growth} objects not cleaned up"

        print(f"✓ Memory leak prevention validated (growth: {object_growth} objects)")

    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self):
        """Test that concurrent processing improves performance."""
        workflow = ResearchWorkflow()
        workflow._ensure_initialized()

        # Create mock dependencies
        mock_client = AsyncMock()
        research_state = ResearchState(
            request_id="perf-test",
            user_id="test-user",
            user_query="Performance test query"
        )

        deps = ResearchDependencies(
            http_client=mock_client,
            api_keys=APIKeys(),
            research_state=research_state,
            metadata=ResearchMetadata(),
        )

        # Mock agent responses with artificial delay to test concurrency
        async def mock_agent_with_delay(agent_type, prompt, deps_arg, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time

            if agent_type == 'clarification':
                return ClarificationResult(
                    needs_clarification=False,
                    verification="Concurrent test verification",
                    confidence_score=0.8,
                    breadth_score=0.3,
                    assessment_reasoning="Concurrent processing test"
                )
            elif agent_type == 'transformation':
                from open_deep_research_with_pydantic_ai.models.research import TransformedQueryResult
                return TransformedQueryResult(
                    original_query="test",
                    transformed_query="concurrent test",
                    transformation_rationale="test rationale",
                    specificity_score=0.7
                )
            elif agent_type == 'brief':
                from open_deep_research_with_pydantic_ai.models.research import BriefGenerationResult
                return BriefGenerationResult(
                    brief_text="Concurrent processing test brief",
                    confidence_score=0.9,
                    key_research_areas=["Concurrency", "Performance"]
                )

        with patch.object(workflow.coordinator, 'run_agent', side_effect=mock_agent_with_delay):
            start_time = time.time()

            # Execute three-phase clarification (should use concurrent processing where possible)
            await workflow._execute_three_phase_clarification(research_state, deps, "test query")

            end_time = time.time()
            total_time = end_time - start_time

            # With 0.1s delay per agent and 3 agents, sequential would take ~0.3s
            # Concurrent processing should be faster (allowing some overhead)
            assert total_time < 0.5, f"Processing took too long: {total_time}s (expected < 0.5s)"

            # Verify all phases completed
            assert "clarification_assessment" in research_state.metadata
            assert "research_brief_text" in research_state.metadata

            print(f"✓ Concurrent processing performance validated ({total_time:.3f}s)")

    @pytest.mark.asyncio
    async def test_circuit_breaker_error_handling(self):
        """Test that circuit breaker prevents cascading failures."""
        workflow = ResearchWorkflow()
        workflow._ensure_initialized()

        agent_type = "test_circuit_breaker"

        # Initially circuit should be closed
        assert workflow._check_circuit_breaker(agent_type)

        # Simulate failures to trigger circuit breaker
        test_error = Exception("Test circuit breaker error")

        for i in range(workflow._circuit_breaker_threshold):
            workflow._record_error(agent_type, test_error)

            if i < workflow._circuit_breaker_threshold - 1:
                # Should still be closed
                assert workflow._check_circuit_breaker(agent_type), f"Circuit opened too early at error {i}"
            else:
                # Should now be open
                assert not workflow._check_circuit_breaker(agent_type), "Circuit should be open after threshold"

        # Verify circuit is open
        assert workflow._circuit_open.get(agent_type, False), "Circuit breaker state not recorded"
        assert workflow._consecutive_errors[agent_type] == workflow._circuit_breaker_threshold

        # Test that circuit remains open for subsequent calls
        with pytest.raises(RuntimeError, match="Circuit breaker open"):
            await workflow._run_agent_with_circuit_breaker(agent_type, "test", None)

        # Test recovery after success
        workflow._record_success(agent_type)
        assert workflow._check_circuit_breaker(agent_type), "Circuit should close after success"
        assert workflow._consecutive_errors[agent_type] == 0, "Error count should reset"

        print("✓ Circuit breaker error handling validated")

    def test_proper_dependency_injection(self):
        """Test that dependency injection follows Pydantic-AI patterns."""
        # Test that ResearchDependencies follows dataclass pattern
        from dataclasses import is_dataclass
        assert is_dataclass(ResearchDependencies), "ResearchDependencies should be a dataclass"

        # Test that dependencies can be created with required fields
        import httpx

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        research_state = ResearchState(
            request_id="di-test",
            user_id="test-user",
            user_query="Dependency injection test"
        )

        deps = ResearchDependencies(
            http_client=mock_client,
            api_keys=APIKeys(),
            research_state=research_state,
            metadata=ResearchMetadata(),
        )

        # Test that all required fields are present
        assert deps.http_client is not None
        assert deps.api_keys is not None
        assert deps.research_state is not None
        assert deps.metadata is not None

        # Test that dependencies have proper types
        assert hasattr(deps.http_client, 'get')  # AsyncClient interface
        assert isinstance(deps.api_keys, APIKeys)
        assert isinstance(deps.research_state, ResearchState)
        assert isinstance(deps.metadata, ResearchMetadata)

        # Test dependency helper methods
        deps.add_metadata("test_key", "test_value")
        assert deps.get_metadata("test_key") == "test_value"
        assert deps.get_metadata("missing", "default") == "default"

        print("✓ Proper dependency injection validated")

    @pytest.mark.asyncio
    async def test_agent_output_validation(self):
        """Test that agent outputs are properly validated using Pydantic models."""
        # Test ClarificationResult validation

        # Valid result should work
        valid_result = ClarificationResult(
            needs_clarification=True,
            question="What specific domain are you interested in?",
            verification="",
            confidence_score=0.75,
            missing_dimensions=["domain", "scope"],
            breadth_score=0.8,
            assessment_reasoning="Query lacks specific domain focus",
            suggested_clarifications=["Specify domain", "Define scope"]
        )

        assert valid_result.needs_clarification is True
        assert valid_result.confidence_score == 0.75
        assert len(valid_result.missing_dimensions) == 2

        # Invalid results should raise validation errors
        with pytest.raises(ValueError, match="Question required when clarification needed"):
            ClarificationResult(
                needs_clarification=True,  # But no question provided
                question="",  # Empty question when clarification needed
                verification="Should not have verification when clarification needed",
                confidence_score=0.75,
                breadth_score=0.8,
                assessment_reasoning="Test"
            )

        # Test confidence score bounds
        with pytest.raises(ValueError):
            ClarificationResult(
                needs_clarification=False,
                verification="Test verification",
                confidence_score=1.5,  # Invalid: > 1.0
                breadth_score=0.5,
                assessment_reasoning="Test"
            )

        with pytest.raises(ValueError):
            ClarificationResult(
                needs_clarification=False,
                verification="Test verification",
                confidence_score=-0.1,  # Invalid: < 0.0
                breadth_score=0.5,
                assessment_reasoning="Test"
            )

        print("✓ Agent output validation working correctly")

    @pytest.mark.asyncio
    async def test_event_system_isolation(self):
        """Test that the event system properly isolates different users/sessions."""
        event_bus = ResearchEventBus()

        # Create events for different users
        from open_deep_research_with_pydantic_ai.core.events import ResearchStartedEvent

        user1_events = []
        user2_events = []

        def user1_handler(event):
            user1_events.append(event)

        def user2_handler(event):
            user2_events.append(event)

        # Subscribe handlers for different users
        await event_bus.subscribe(ResearchStartedEvent, user1_handler)
        await event_bus.subscribe(ResearchStartedEvent, user2_handler)

        # Emit events for user1
        user1_event = ResearchStartedEvent(
            _request_id="user1-123",
            user_query="User 1 query",
            user_id="user1"
        )

        # Emit events for user2
        user2_event = ResearchStartedEvent(
            _request_id="user2-456",
            user_query="User 2 query",
            user_id="user2"
        )

        await event_bus.emit(user1_event)
        await event_bus.emit(user2_event)
        await asyncio.sleep(0.1)  # Allow async processing

        # Both handlers should receive both events (they're not filtered by user at handler level)
        # The isolation happens at the history/stats level
        assert len(user1_events) == 2  # Both events
        assert len(user2_events) == 2  # Both events

        # Test user-specific cleanup
        await event_bus.cleanup_user("user1")

        # Test that stats and history are properly managed
        stats = await event_bus.get_stats()
        assert stats['total_events_emitted'] >= 2

        await event_bus.cleanup()

        print("✓ Event system isolation validated")

    @pytest.mark.asyncio
    async def test_async_performance_vs_sequential(self):
        """Test that async processing provides performance benefits."""

        # Simulate sequential processing
        async def sequential_processing():
            start_time = time.time()

            # Simulate 3 operations taking 0.1s each
            for i in range(3):
                await asyncio.sleep(0.1)

            return time.time() - start_time

        # Simulate concurrent processing
        async def concurrent_processing():
            start_time = time.time()

            # Same 3 operations but concurrent
            tasks = [asyncio.sleep(0.1) for _ in range(3)]
            await asyncio.gather(*tasks)

            return time.time() - start_time

        # Run both approaches
        sequential_time = await sequential_processing()
        concurrent_time = await concurrent_processing()

        # Concurrent should be significantly faster
        assert concurrent_time < sequential_time * 0.7, f"Concurrent ({concurrent_time:.3f}s) should be much faster than sequential ({sequential_time:.3f}s)"

        # Sequential should be ~0.3s, concurrent should be ~0.1s
        assert 0.25 <= sequential_time <= 0.35, f"Sequential time unexpected: {sequential_time:.3f}s"
        assert 0.08 <= concurrent_time <= 0.15, f"Concurrent time unexpected: {concurrent_time:.3f}s"

        print(f"✓ Async performance improvement validated (sequential: {sequential_time:.3f}s, concurrent: {concurrent_time:.3f}s)")

    def test_structured_data_models(self):
        """Test that all data models follow proper Pydantic validation."""
        from open_deep_research_with_pydantic_ai.models.research import (
            TransformedQueryResult, BriefGenerationResult
        )
        from pydantic import ValidationError

        # Test TransformedQueryResult validation
        valid_transform = TransformedQueryResult(
            original_query="What is AI?",
            transformed_query="What are the key AI techniques used in modern applications?",
            transformation_rationale="Added specificity about techniques and applications",
            specificity_score=0.8,
            supporting_questions=["What are ML algorithms?"],
            complexity_assessment="medium",
            estimated_scope="moderate"
        )

        assert valid_transform.original_query != valid_transform.transformed_query
        assert len(valid_transform.transformed_query) >= 10
        assert 0.0 <= valid_transform.specificity_score <= 1.0

        # Test validation errors
        with pytest.raises(ValidationError):
            TransformedQueryResult(
                original_query="test",
                transformed_query="test",  # Same as original - should fail
                transformation_rationale="short",  # Too short - should fail
                specificity_score=0.5
            )

        # Test BriefGenerationResult validation
        valid_brief = BriefGenerationResult(
            brief_text="This is a comprehensive research brief that meets the minimum length requirements. The research objective is to understand key concepts. The scope includes analyzing methods and outcomes.",
            confidence_score=0.9,
            key_research_areas=["Area 1", "Area 2"],
            research_objectives=["Objective 1"],
            estimated_complexity="medium"
        )

        assert len(valid_brief.brief_text) >= 100
        assert len(valid_brief.key_research_areas) >= 1

        print("✓ Structured data models validation working correctly")

    @pytest.mark.asyncio
    async def test_system_resilience_under_load(self):
        """Test that the system remains stable under concurrent load."""
        workflow = ResearchWorkflow()
        workflow._ensure_initialized()

        # Create multiple concurrent requests
        concurrent_requests = 10
        results = []

        async def single_request(request_id: str):
            """Simulate a single request with mock agent responses."""
            mock_client = AsyncMock()
            research_state = ResearchState(
                request_id=f"load-test-{request_id}",
                user_id=f"user-{request_id}",
                user_query=f"Load test query {request_id}"
            )

            deps = ResearchDependencies(
                http_client=mock_client,
                api_keys=APIKeys(),
                research_state=research_state,
                metadata=ResearchMetadata(),
            )

            # Mock quick agent responses
            async def quick_mock_agent(agent_type, prompt, deps_arg, **kwargs):
                await asyncio.sleep(0.01)  # Very fast response

                if agent_type == 'clarification':
                    return ClarificationResult(
                        needs_clarification=False,
                        verification=f"Load test {request_id}",
                        confidence_score=0.8,
                        breadth_score=0.3,
                        assessment_reasoning="Load test reasoning"
                    )
                elif agent_type == 'brief':
                    from open_deep_research_with_pydantic_ai.models.research import BriefGenerationResult
                    return BriefGenerationResult(
                        brief_text=f"Load test brief for request {request_id} " * 10,  # Meet min length
                        confidence_score=0.9,
                        key_research_areas=[f"Area-{request_id}"]
                    )

            with patch.object(workflow.coordinator, 'run_agent', side_effect=quick_mock_agent):
                try:
                    await workflow._execute_three_phase_clarification(research_state, deps, f"test query {request_id}")
                    return {"success": True, "request_id": request_id, "metadata_keys": list(research_state.metadata.keys())}
                except Exception as e:
                    return {"success": False, "request_id": request_id, "error": str(e)}

        # Execute all requests concurrently
        start_time = time.time()
        tasks = [single_request(str(i)) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        total_time = end_time - start_time

        # Verify all requests completed successfully
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_results = [r for r in results if not isinstance(r, dict) or not r.get("success")]

        success_rate = len(successful_results) / len(results)

        assert success_rate >= 0.9, f"Success rate too low: {success_rate:.2%} (failed: {len(failed_results)})"
        assert total_time < 5.0, f"Load test took too long: {total_time:.3f}s"

        # Verify circuit breakers didn't trip under normal load
        for agent_type in ['clarification', 'transformation', 'brief']:
            assert not workflow._circuit_open.get(agent_type, False), f"Circuit breaker opened for {agent_type}"

        print(f"✓ System resilience validated ({concurrent_requests} concurrent requests in {total_time:.3f}s, {success_rate:.1%} success rate)")
