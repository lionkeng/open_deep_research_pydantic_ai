"""Server-Sent Events handler using sse-starlette."""

import asyncio
from collections.abc import AsyncGenerator

import logfire
from fastapi import Request
from sse_starlette import EventSourceResponse
from sse_starlette.event import ServerSentEvent

from core.events import (
    ErrorEvent,
    ResearchCompletedEvent,
    ResearchEvent,
    StageCompletedEvent,
    StreamingUpdateEvent,
    research_event_bus,
)
from core.sse_models import (
    CompletedMessage,
    ConnectionMessage,
    ErrorMessage,
    HeartbeatMessage,
    PingMessage,
    SSEEventType,
    StageCompletedMessage,
    StreamErrorMessage,
    UpdateMessage,
)
from models.core import ResearchState


class SSEHandler:
    """Handler for Server-Sent Events with proper implementation."""

    def __init__(self, request_id: str, request: Request):
        """Initialize SSE handler.

        Args:
            request_id: Research request ID to track
            request: FastAPI request object for disconnection detection
        """
        self.request_id = request_id
        self.request = request
        self.event_queue: asyncio.Queue[ResearchEvent] = asyncio.Queue()
        self.event_id = 0
        self._subscribed = False

    async def capture_event(self, event: ResearchEvent) -> None:
        """Capture events for this request.

        Args:
            event: Research event to capture
        """
        if event.request_id == self.request_id:
            await self.event_queue.put(event)

    async def subscribe_to_events(self) -> None:
        """Subscribe to relevant research events."""
        if not self._subscribed:
            await research_event_bus.subscribe(StreamingUpdateEvent, self.capture_event)
            await research_event_bus.subscribe(StageCompletedEvent, self.capture_event)
            await research_event_bus.subscribe(ErrorEvent, self.capture_event)
            await research_event_bus.subscribe(ResearchCompletedEvent, self.capture_event)
            self._subscribed = True
            logfire.info(f"SSE handler subscribed to events for {self.request_id}")

    async def event_generator(
        self, active_sessions: dict[str, ResearchState]
    ) -> AsyncGenerator[ServerSentEvent, None]:
        """Generate Server-Sent Events.

        Args:
            active_sessions: Dictionary of active research sessions

        Yields:
            ServerSentEvent instances
        """
        await self.subscribe_to_events()

        try:
            # Send initial connection event with retry interval
            connection_msg = ConnectionMessage(
                request_id=self.request_id, message="Connected to research stream"
            )
            yield ServerSentEvent(
                data=connection_msg.model_dump_json(),
                event=SSEEventType.CONNECTION,
                id=str(self.event_id),
                retry=5000,  # Retry after 5 seconds if connection drops
            )
            self.event_id += 1

            # Stream events until research is complete or client disconnects
            while True:
                # Check if client disconnected
                if await self.request.is_disconnected():
                    logfire.info(f"Client disconnected from SSE stream {self.request_id}")
                    break

                try:
                    # Wait for events with timeout for heartbeat
                    # Shorter timeout helps keep clients alive during long phases
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=15.0)

                    # Format event based on type
                    if isinstance(event, StreamingUpdateEvent):
                        update_msg = UpdateMessage(
                            request_id=self.request_id,
                            stage=event.stage.name,
                            content=event.content,
                            is_partial=event.is_partial,
                        )
                        yield ServerSentEvent(
                            data=update_msg.model_dump_json(),
                            event=SSEEventType.UPDATE,
                            id=str(self.event_id),
                        )
                        self.event_id += 1

                    elif isinstance(event, StageCompletedEvent):
                        stage_msg = StageCompletedMessage(
                            request_id=self.request_id,
                            stage=event.stage.name,
                            success=event.success,
                            result=event.result,
                            error=event.error_message,
                        )
                        yield ServerSentEvent(
                            data=stage_msg.model_dump_json(),
                            event=SSEEventType.STAGE_COMPLETED,
                            id=str(self.event_id),
                        )
                        self.event_id += 1

                    elif isinstance(event, ErrorEvent):
                        error_msg = ErrorMessage(
                            request_id=self.request_id,
                            stage=event.stage.name,
                            error_type=event.error_type,
                            message=event.error_message,
                            recoverable=event.recoverable,
                        )
                        yield ServerSentEvent(
                            data=error_msg.model_dump_json(),
                            event=SSEEventType.ERROR,
                            id=str(self.event_id),
                        )
                        self.event_id += 1

                    elif isinstance(event, ResearchCompletedEvent):
                        # Send completion event
                        completed_msg = CompletedMessage(
                            request_id=self.request_id,
                            success=event.success,
                            duration=event.duration_seconds,
                            error=event.error_message,
                            has_report=event.report is not None,
                        )
                        yield ServerSentEvent(
                            data=completed_msg.model_dump_json(),
                            event=SSEEventType.COMPLETE,
                            id=str(self.event_id),
                        )
                        self.event_id += 1

                        # Research is complete, exit loop
                        logfire.info(f"Research completed for {self.request_id}")
                        break

                    # Check if research is complete via session state
                    if self.request_id in active_sessions:
                        state = active_sessions[self.request_id]
                        if state.is_completed() and not isinstance(event, ResearchCompletedEvent):
                            # Send completion event if not already sent
                            # Calculate duration if timing information is available
                            duration = None
                            if state.started_at and state.completed_at:
                                duration = (state.completed_at - state.started_at).total_seconds()

                            completed_msg = CompletedMessage(
                                request_id=self.request_id,
                                success=state.error_message is None,
                                duration=duration,
                                error=state.error_message,
                                has_report=state.final_report is not None,
                            )
                            yield ServerSentEvent(
                                data=completed_msg.model_dump_json(),
                                event=SSEEventType.COMPLETE,
                                id=str(self.event_id),
                            )
                            self.event_id += 1
                            break

                except TimeoutError:
                    # Send heartbeat to keep connection alive
                    heartbeat_msg = HeartbeatMessage()
                    yield ServerSentEvent(
                        data=heartbeat_msg.model_dump_json(),
                        event=SSEEventType.PING,
                        id=str(self.event_id),
                    )
                    self.event_id += 1

        except asyncio.CancelledError:
            logfire.info(f"SSE stream cancelled for {self.request_id}")
            raise
        except Exception as e:
            logfire.error(f"SSE error for {self.request_id}: {str(e)}", exc_info=True)
            # Send error event before closing
            stream_error_msg = StreamErrorMessage(message=str(e))
            yield ServerSentEvent(
                data=stream_error_msg.model_dump_json(),
                event=SSEEventType.ERROR,
                id=str(self.event_id),
            )
        finally:
            logfire.info(f"SSE stream closed for {self.request_id}")


def create_sse_response(
    request_id: str, request: Request, active_sessions: dict[str, ResearchState]
) -> EventSourceResponse:
    """Create a properly configured SSE response.

    Args:
        request_id: Research request ID
        request: FastAPI request object
        active_sessions: Dictionary of active research sessions

    Returns:
        EventSourceResponse configured for the request
    """
    handler = SSEHandler(request_id, request)

    return EventSourceResponse(
        handler.event_generator(active_sessions),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
            "Connection": "keep-alive",
        },
        media_type="text/event-stream",
        ping=15,  # Send ping every 15 seconds to keep connection alive
        ping_message_factory=lambda: ServerSentEvent(
            data=PingMessage().model_dump_json(), event=SSEEventType.PING
        ),
    )
