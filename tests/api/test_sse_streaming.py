"""Phase 4.1 SSE streaming infrastructure tests."""

from __future__ import annotations

import asyncio

import pytest

from api.sse_handler import SSEEventType, SSEHandler
from core.events import (
    ErrorEvent,
    ResearchCompletedEvent,
    ResearchStage,
    StreamingUpdateEvent,
)
from models.core import ResearchState

from .sse_test_utils import decode_sse_event


class _StaticRequest:
    """Simple request stub that never disconnects."""

    async def is_disconnected(self) -> bool:  # pragma: no cover - trivial
        return False


@pytest.mark.asyncio
async def test_sse_stream_emits_updates_and_completion(mock_research_event_bus):
    """SSE handler should emit connection, update, and completion messages."""

    handler = SSEHandler(request_id="req-1", request=_StaticRequest())
    events = handler.event_generator(active_sessions={})

    connection = await asyncio.wait_for(anext(events), timeout=0.5)
    assert decode_sse_event(connection)["event"] == SSEEventType.CONNECTION

    await mock_research_event_bus.emit(
        StreamingUpdateEvent(
            _request_id="req-1",
            content="Collecting sources",
            stage=ResearchStage.RESEARCH_EXECUTION,
            is_partial=True,
        )
    )

    update = await asyncio.wait_for(anext(events), timeout=0.5)
    decoded_update = decode_sse_event(update)
    assert decoded_update["event"] == SSEEventType.UPDATE
    assert decoded_update["data"]["content"] == "Collecting sources"

    await mock_research_event_bus.emit(
        ResearchCompletedEvent(
            _request_id="req-1",
            report=None,
            success=True,
            duration_seconds=1.2,
            error_message=None,
        )
    )

    completed = await asyncio.wait_for(anext(events), timeout=0.5)
    decoded_completed = decode_sse_event(completed)
    assert decoded_completed["event"] == SSEEventType.COMPLETE
    assert decoded_completed["data"]["success"] is True

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(events), timeout=0.5)


@pytest.mark.asyncio
async def test_sse_stream_emits_error_event(mock_research_event_bus):
    """Error events should be surfaced over the SSE stream."""

    request_id = "req-error"
    handler = SSEHandler(request_id=request_id, request=_StaticRequest())
    state = ResearchState(
        request_id=request_id,
        user_id="user",
        session_id=None,
        user_query="test",
    )
    events = handler.event_generator(active_sessions={request_id: state})

    await asyncio.wait_for(anext(events), timeout=0.5)  # Connection event

    await mock_research_event_bus.emit(
        ErrorEvent(
            _request_id=request_id,
            stage=ResearchStage.RESEARCH_EXECUTION,
            error_type="ValidationError",
            error_message="search failed",
            recoverable=True,
        )
    )

    error_event = await asyncio.wait_for(anext(events), timeout=0.5)
    decoded_error = decode_sse_event(error_event)
    assert decoded_error["event"] == SSEEventType.ERROR
    assert decoded_error["data"]["error_type"] == "ValidationError"
    assert decoded_error["data"]["recoverable"] is True

    # Close generator to avoid leaks
    await events.aclose()
