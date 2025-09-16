"""Basic tests for the deep research system."""

import pytest

from core.events import (
    ResearchStartedEvent,
    emit_research_started,
    research_event_bus,
)
from models.core import ResearchStage, ResearchState
from models.research_executor import ResearchFinding, ResearchSource


def test_research_state():
    """Test research state creation and advancement."""
    state = ResearchState(
        request_id="test-123",
        user_id="test-user",
        session_id="test-session",
        user_query="What is quantum computing?",
    )

    assert state.request_id == "test-123"
    assert state.user_query == "What is quantum computing?"
    assert state.current_stage == ResearchStage.PENDING
    assert not state.is_completed()

    # Test stage advancement
    state.advance_stage()
    assert state.current_stage == ResearchStage.CLARIFICATION

    # Test error setting
    state.set_error("Test error")
    assert state.error_message == "Test error"
    assert state.is_completed()


def test_research_finding():
    """Test research finding creation."""
    source = ResearchSource(
        url="https://example.com",
        title="Quantum Computing Research",
        relevance_score=0.9
    )

    finding = ResearchFinding(
        finding="Quantum computing uses quantum bits",
        confidence_level=0.85,
        source=source,
        supporting_evidence=["Evidence 1", "Evidence 2"]
    )

    assert finding.finding == "Quantum computing uses quantum bits"
    assert finding.confidence_level == 0.85
    assert finding.source.relevance_score == 0.9


@pytest.mark.asyncio
async def test_event_bus():
    """Test event bus functionality."""
    # Clean up any existing state first
    await research_event_bus.cleanup()
    async with research_event_bus._history_lock:
        research_event_bus._event_history.clear()
        research_event_bus._event_count_by_user.clear()
        research_event_bus._active_users.clear()

    received_events: list[ResearchStartedEvent] = []

    async def handler(event: ResearchStartedEvent):
        received_events.append(event)

    # Subscribe to events
    await research_event_bus.subscribe(ResearchStartedEvent, handler)

    # Emit event
    await emit_research_started("test-456", "Test query")

    # Wait a bit for async processing
    import asyncio

    await asyncio.sleep(0.1)

    # Check event was received
    assert len(received_events) == 1
    assert received_events[0].request_id == "test-456"
    assert received_events[0].user_query == "Test query"

    # Cleanup
    await research_event_bus.cleanup()
    async with research_event_bus._history_lock:
        research_event_bus._event_history.clear()
        research_event_bus._event_count_by_user.clear()
        research_event_bus._active_users.clear()
