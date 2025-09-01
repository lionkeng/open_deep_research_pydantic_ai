"""Basic tests for the deep research system."""

import pytest

from core.events import (
    ResearchStartedEvent,
    emit_research_started,
    research_event_bus,
)
from models.brief_generator import ResearchBrief
from models.core import ResearchStage, ResearchState
from models.research_executor import ResearchFinding


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


def test_research_brief():
    """Test research brief creation."""
    brief = ResearchBrief(
        topic="Quantum Computing",
        objectives=["Understand basics", "Learn applications"],
        key_questions=["What is it?", "How does it work?"],
        scope="Basic introduction",
    )

    assert brief.topic == "Quantum Computing"
    assert len(brief.objectives) == 2
    assert len(brief.key_questions) == 2


def test_research_finding():
    """Test research finding creation."""
    finding = ResearchFinding(
        content="Quantum computing uses quantum bits",
        source="https://example.com",
        relevance_score=0.9,
        confidence=0.85,
    )

    assert finding.content == "Quantum computing uses quantum bits"
    assert finding.relevance_score == 0.9
    assert finding.confidence == 0.85


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
