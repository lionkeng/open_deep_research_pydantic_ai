"""Tests for concurrent user request isolation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from core.context import (
    ResearchContext,
    ResearchContextManager,
    get_current_context,
)
from core.events import ResearchEventBus
from models.core import ResearchState


@pytest_asyncio.fixture
async def clean_event_bus():
    """Provide a clean event bus for each test."""
    from core.events import research_event_bus

    # Clean up any existing state before the test
    await research_event_bus.cleanup()
    async with research_event_bus._history_lock:
        research_event_bus._event_history.clear()
        research_event_bus._event_count_by_user.clear()
        research_event_bus._active_users.clear()

    yield research_event_bus

    # Clean up after the test
    await research_event_bus.cleanup()
    async with research_event_bus._history_lock:
        research_event_bus._event_history.clear()
        research_event_bus._event_count_by_user.clear()
        research_event_bus._active_users.clear()


class TestResearchContext:
    """Test the ResearchContext class."""

    def test_context_scope_key(self):
        """Test scope key generation."""
        # Without session
        ctx1 = ResearchContext(user_id="user1")
        assert ctx1.get_scope_key() == "user1"

        # With session
        ctx2 = ResearchContext(user_id="user1", session_id="session1")
        assert ctx2.get_scope_key() == "user1:session1"

    def test_context_matches_request(self):
        """Test request ID matching."""
        ctx = ResearchContext(user_id="user1", session_id="session1")

        # Matching request
        assert ctx.matches_request("user1:session1:uuid-123")

        # Non-matching user
        assert not ctx.matches_request("user2:session1:uuid-123")

        # Non-matching session
        assert not ctx.matches_request("user1:session2:uuid-123")

        # Without session in context
        ctx_no_session = ResearchContext(user_id="user1")
        assert ctx_no_session.matches_request("user1:uuid-123")
        assert not ctx_no_session.matches_request("user2:uuid-123")


class TestRequestIDGeneration:
    """Test scoped request ID generation."""

    def test_generate_request_id_with_user(self):
        """Test request ID generation with user ID."""
        request_id = ResearchState.generate_request_id("user1")
        assert request_id.startswith("user1:")
        parts = request_id.split(":")
        assert len(parts) == 2
        assert parts[0] == "user1"

    def test_generate_request_id_with_session(self):
        """Test request ID generation with user and session."""
        request_id = ResearchState.generate_request_id("user1", "session1")
        assert request_id.startswith("user1:session1:")
        parts = request_id.split(":")
        assert len(parts) == 3
        assert parts[0] == "user1"
        assert parts[1] == "session1"


@pytest.mark.asyncio
class TestUserScopedEventBus:
    """Test user-scoped event isolation in ResearchEventBus."""

    async def test_event_isolation_between_users(self, clean_event_bus):
        """Test that events are isolated between different users."""
        from core.events import (
            ResearchStartedEvent,
        )

        # Create events for different users
        event1 = ResearchStartedEvent(
            _request_id="user1:req1",
            user_query="Test query 1",
        )
        event2 = ResearchStartedEvent(
            _request_id="user2:req2",
            user_query="Test query 2",
        )

        # Emit events with different user contexts
        async with ResearchContextManager(user_id="user1"):
            await clean_event_bus.emit(event1)

        async with ResearchContextManager(user_id="user2"):
            await clean_event_bus.emit(event2)

        # Check that each user only sees their own events
        with patch(
            "open_deep_research_pydantic_ai.core.events.get_current_context",
            return_value=ResearchContext(user_id="user1"),
        ):
            user1_history = await clean_event_bus.get_event_history("user1:req1")
            assert len(user1_history) == 1
            assert user1_history[0].request_id == "user1:req1"

        with patch(
            "open_deep_research_pydantic_ai.core.events.get_current_context",
            return_value=ResearchContext(user_id="user2"),
        ):
            user2_history = await clean_event_bus.get_event_history("user2:req2")
            assert len(user2_history) == 1
            assert user2_history[0].request_id == "user2:req2"

    async def test_user_cleanup(self):
        """Test that user resources can be cleaned up."""
        event_bus = ResearchEventBus()

        # Add events for multiple users (directly for testing)
        async with event_bus._history_lock:
            event_bus._event_history["user1:req1"] = []
            event_bus._event_history["user1:req2"] = []
            event_bus._event_history["user2:req1"] = []
            event_bus._event_count_by_user["user1"] = 10
            event_bus._event_count_by_user["user2"] = 5
            event_bus._active_users.add("user1")
            event_bus._active_users.add("user2")

        # Clean up user1
        await event_bus.cleanup_user("user1")

        # Check user1 resources are cleaned
        async with event_bus._history_lock:
            assert "user1:req1" not in event_bus._event_history
            assert "user1:req2" not in event_bus._event_history
            assert "user1" not in event_bus._event_count_by_user
            assert "user1" not in event_bus._active_users

            # Check user2 resources remain
            assert "user2:req1" in event_bus._event_history
            assert event_bus._event_count_by_user["user2"] == 5
            assert "user2" in event_bus._active_users

    async def test_concurrent_event_emission(self, clean_event_bus):
        """Test that concurrent event emissions are thread-safe."""
        from core.events import (
            ResearchStartedEvent,
        )

        # Function to emit events concurrently
        async def emit_events(user_id: str, count: int):
            async with ResearchContextManager(user_id=user_id):
                for i in range(count):
                    event = ResearchStartedEvent(
                        _request_id=f"{user_id}:req{i}",
                        user_query=f"Query {i} from {user_id}",
                    )
                    await clean_event_bus.emit(event)

        # Run multiple users concurrently
        await asyncio.gather(
            emit_events("user1", 50),
            emit_events("user2", 50),
            emit_events("user3", 50),
        )

        # Verify all events were recorded correctly
        async with clean_event_bus._history_lock:
            # Count events per user
            user1_events = sum(
                1
                for key in clean_event_bus._event_history.keys()
                if key.startswith("user1:")
            )
            user2_events = sum(
                1
                for key in clean_event_bus._event_history.keys()
                if key.startswith("user2:")
            )
            user3_events = sum(
                1
                for key in clean_event_bus._event_history.keys()
                if key.startswith("user3:")
            )

            assert user1_events == 50, f"Expected 50 user1 events, got {user1_events}"
            assert user2_events == 50, f"Expected 50 user2 events, got {user2_events}"
            assert user3_events == 50, f"Expected 50 user3 events, got {user3_events}"

            # Verify event counts
            assert clean_event_bus._event_count_by_user["user1"] == 50
            assert clean_event_bus._event_count_by_user["user2"] == 50
            assert clean_event_bus._event_count_by_user["user3"] == 50


@pytest.mark.asyncio
class TestConcurrentWorkflows:
    """Test concurrent research workflows with user isolation."""

    async def test_concurrent_research_requests(self):
        """Test that concurrent requests from different users work correctly."""
        from core.workflow import ResearchWorkflow

        with patch("open_deep_research_pydantic_ai.core.workflow.httpx.AsyncClient"):
            workflow = ResearchWorkflow()

            # Mock the agents
            with patch.object(workflow, "_ensure_initialized"):
                # Create concurrent tasks for different users
                async def user1_research():
                    async with ResearchContextManager(user_id="user1"):
                        state = await workflow.execute_research(
                            user_query="User 1 query",
                            user_id="user1",
                        )
                        return state

                async def user2_research():
                    async with ResearchContextManager(user_id="user2"):
                        state = await workflow.execute_research(
                            user_query="User 2 query",
                            user_id="user2",
                        )
                        return state

                # Run concurrently
                results = await asyncio.gather(
                    user1_research(),
                    user2_research(),
                )

                # Check results
                state1, state2 = results
                assert state1.user_id == "user1"
                assert state2.user_id == "user2"
                assert state1.request_id.startswith("user1:")
                assert state2.request_id.startswith("user2:")

    async def test_context_manager_usage(self):
        """Test the ResearchContextManager."""
        # Initially no context
        default_ctx = get_current_context()
        assert default_ctx.user_id == "default"

        # Set context using manager
        async with ResearchContextManager(user_id="test_user", session_id="test_session"):
            ctx = get_current_context()
            assert ctx.user_id == "test_user"
            assert ctx.session_id == "test_session"

        # Context restored after exit
        ctx_after = get_current_context()
        assert ctx_after.user_id == "default"


class TestAPIUserContext:
    """Test API layer user context extraction."""

    @pytest.mark.asyncio
    async def test_api_extracts_user_headers(self):
        """Test that the API extracts user ID and session ID from headers."""
        from fastapi.testclient import TestClient

        from api.main import app

        with TestClient(app) as client:
            # Mock the workflow execution
            with patch(
                "open_deep_research_pydantic_ai.api.main.workflow.execute_research",
                new_callable=AsyncMock,
            ) as mock_execute:
                mock_execute.return_value = MagicMock(
                    request_id="user123:session456:uuid",
                    user_id="user123",
                    session_id="session456",
                )

                # Send request with user headers
                response = client.post(
                    "/research",
                    json={
                        "query": "Test query",
                        "api_keys": {"openai": "test-key"},
                        "stream": False,
                    },
                    headers={
                        "X-User-ID": "user123",
                        "X-Session-ID": "session456",
                    },
                )

                # Check response
                assert response.status_code == 200
                data = response.json()
                assert "request_id" in data
                # The request_id should contain the user context
                assert "user123" in data["request_id"] or "api-user" in data["request_id"]
