"""Event-driven architecture for research workflow coordination.

This module provides a clean event bus system for coordinating multi-agent
research workflows. Events are immutable and processed asynchronously,
eliminating deadlock possibilities while maintaining proper coordination.
"""

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

import logfire

from open_deep_research_with_pydantic_ai.core.context import get_current_context
from open_deep_research_with_pydantic_ai.models.research import (
    ResearchFinding,
    ResearchReport,
    ResearchStage,
)


class ResearchEvent(ABC):
    """Base class for all research-related events."""

    @property
    @abstractmethod
    def request_id(self) -> str:
        """Research request ID this event relates to."""
        pass

    @property
    def timestamp(self) -> datetime:
        """When this event occurred."""
        return datetime.now(UTC)


@dataclass(frozen=True)
class ResearchStartedEvent(ResearchEvent):
    """Emitted when a new research request is initiated."""

    _request_id: str
    user_query: str
    user_id: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def request_id(self) -> str:
        return self._request_id


@dataclass(frozen=True)
class StageCompletedEvent(ResearchEvent):
    """Emitted when a research stage is completed."""

    _request_id: str
    stage: ResearchStage
    success: bool
    result: Any = None
    error_message: str | None = None

    @property
    def request_id(self) -> str:
        return self._request_id


@dataclass(frozen=True)
class AgentDelegationEvent(ResearchEvent):
    """Emitted when one agent delegates to another."""

    _request_id: str
    from_agent: str
    to_agent: str
    task_description: str
    context: dict[str, Any] | None = None

    @property
    def request_id(self) -> str:
        return self._request_id


@dataclass(frozen=True)
class FindingDiscoveredEvent(ResearchEvent):
    """Emitted when a new research finding is discovered."""

    _request_id: str
    finding: ResearchFinding
    agent: str

    @property
    def request_id(self) -> str:
        return self._request_id


@dataclass(frozen=True)
class ResearchCompletedEvent(ResearchEvent):
    """Emitted when research is fully completed."""

    _request_id: str
    report: ResearchReport | None
    success: bool
    duration_seconds: float
    error_message: str | None = None

    @property
    def request_id(self) -> str:
        return self._request_id


@dataclass(frozen=True)
class StreamingUpdateEvent(ResearchEvent):
    """Emitted for streaming updates to clients."""

    _request_id: str
    content: str
    stage: ResearchStage
    is_partial: bool = True

    @property
    def request_id(self) -> str:
        return self._request_id


@dataclass(frozen=True)
class ErrorEvent(ResearchEvent):
    """Emitted when an error occurs during research."""

    _request_id: str
    stage: ResearchStage
    error_type: str
    error_message: str
    recoverable: bool = False

    @property
    def request_id(self) -> str:
        return self._request_id


# Clarification Events
@dataclass(frozen=True)
class ClarificationRequestedEvent(ResearchEvent):
    """Emitted when interactive clarification is requested for a query."""

    _request_id: str
    original_query: str
    ambiguity_reasons: list[str]
    interaction_mode: str  # 'cli' or 'http'

    @property
    def request_id(self) -> str:
        return self._request_id


@dataclass(frozen=True)
class ClarificationSessionCreatedEvent(ResearchEvent):
    """Emitted when a clarification session is created."""

    _request_id: str
    session_id: str
    num_questions: int
    questions_preview: list[str]  # First few words of each question

    @property
    def request_id(self) -> str:
        return self._request_id


@dataclass(frozen=True)
class ClarificationQuestionAnsweredEvent(ResearchEvent):
    """Emitted when a user answers a clarification question."""

    _request_id: str
    session_id: str
    question_id: str
    question_text: str
    selected_choice: str
    free_text_response: str | None = None
    completion_progress: float = 0.0  # 0.0 to 1.0

    @property
    def request_id(self) -> str:
        return self._request_id


@dataclass(frozen=True)
class ClarificationCompletedEvent(ResearchEvent):
    """Emitted when clarification session is completed successfully."""

    _request_id: str
    session_id: str
    original_query: str
    refined_query: str
    confidence_score: float
    num_responses: int
    completion_ratio: float
    enhancements_applied: list[str]

    @property
    def request_id(self) -> str:
        return self._request_id


@dataclass(frozen=True)
class ClarificationCancelledEvent(ResearchEvent):
    """Emitted when clarification session is cancelled by user."""

    _request_id: str
    session_id: str
    reason: str  # 'user_cancelled', 'timeout', 'error'
    partial_responses: int = 0  # Number of questions answered before cancellation

    @property
    def request_id(self) -> str:
        return self._request_id


@dataclass(frozen=True)
class QueryRefinedEvent(ResearchEvent):
    """Emitted when a query is successfully refined based on clarification."""

    _request_id: str
    original_query: str
    refined_query: str
    refinement_context: dict[str, str]  # Context extracted from responses
    confidence_score: float
    enhancement_summary: str

    @property
    def request_id(self) -> str:
        return self._request_id


T = TypeVar("T", bound=ResearchEvent)
EventHandler = Callable[[T], Any]


class ResearchEventBus:
    """Central event dispatcher for research workflow events.

    Provides a clean, lock-free way to coordinate multi-agent research
    operations through immutable events. Handlers are called asynchronously,
    preventing deadlocks and improving performance.
    """

    def __init__(self):
        # Handlers are global (not user-scoped) for system-wide monitoring
        self._handlers: dict[type[ResearchEvent], list[Callable[[Any], Any]]] = {}
        # Event counts and history are scoped by user for isolation
        self._event_count_by_user: dict[str, int] = defaultdict(int)
        self._background_tasks: set[asyncio.Task] = set()
        # Event history keyed by user_scope:request_id for isolation
        self._event_history: dict[str, list[ResearchEvent]] = {}
        # Track active users for cleanup
        self._active_users: set[str] = set()

        # Locks for thread-safe access to shared state
        self._handlers_lock = asyncio.Lock()  # Protects handler registration
        self._history_lock = asyncio.Lock()  # Protects event history and counts
        self._tasks_lock = asyncio.Lock()  # Protects background tasks set

    async def subscribe(self, event_type: type[T], handler: EventHandler[T]) -> None:
        """Subscribe to events of a specific type.

        Args:
            event_type: The type of event to subscribe to
            handler: Callable that takes the event as parameter
        """
        async with self._handlers_lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []

            self._handlers[event_type].append(handler)

            logfire.debug(
                f"Handler subscribed to {event_type.__name__}",
                handler_name=handler.__name__ if hasattr(handler, "__name__") else str(handler),
                total_handlers=len(self._handlers[event_type]),
            )

    async def emit(self, event: ResearchEvent) -> None:
        """Emit an event to all registered handlers.

        Events are processed asynchronously without blocking the caller.
        Failed handlers are logged but don't affect other handlers.

        Args:
            event: The event to emit
        """
        # Get user scope from context
        context = get_current_context()
        user_scope = context.get_scope_key()
        event_type = type(event)

        # Get handlers snapshot to avoid holding lock during handler execution
        async with self._handlers_lock:
            handlers = list(self._handlers.get(event_type, []))

        # Update history and counts with lock
        async with self._history_lock:
            # Update user-scoped event count
            self._event_count_by_user[user_scope] += 1
            event_count = self._event_count_by_user[user_scope]

            # Store in user-scoped history
            history_key = f"{user_scope}:{event.request_id}"
            if history_key not in self._event_history:
                self._event_history[history_key] = []
            self._event_history[history_key].append(event)

            # Track user activity
            self._active_users.add(user_scope)

        if not handlers:
            logfire.debug(
                f"No handlers for {event_type.__name__}",
                request_id=event.request_id,
                event_count=event_count,
                user_scope=user_scope,
            )
            return

        logfire.debug(
            f"Emitting {event_type.__name__}",
            request_id=event.request_id,
            handler_count=len(handlers),
            event_count=event_count,
        )

        # Process handlers concurrently without blocking
        for handler in handlers:
            task = asyncio.create_task(self._safe_call_handler(handler, event))
            async with self._tasks_lock:
                self._background_tasks.add(task)
            # Clean up task when it's done to prevent memory leaks
            task.add_done_callback(lambda t: asyncio.create_task(self._remove_task(t)))

    async def _remove_task(self, task: asyncio.Task) -> None:
        """Remove a task from the background tasks set safely."""
        async with self._tasks_lock:
            self._background_tasks.discard(task)

    async def _safe_call_handler(self, handler: Callable[[Any], Any], event: ResearchEvent) -> None:
        """Safely call a handler, catching and logging any exceptions."""
        try:
            result = handler(event)
            # Handle both sync and async handlers
            if asyncio.iscoroutine(result):
                await result

        except Exception as e:
            logfire.error(
                f"Event handler failed for {type(event).__name__}",
                handler_name=handler.__name__ if hasattr(handler, "__name__") else str(handler),
                request_id=event.request_id,
                error=str(e),
                exc_info=True,
            )

    async def cleanup(self) -> None:
        """Clean up all background tasks. Should be called before event loop closes."""
        async with self._tasks_lock:
            if self._background_tasks:
                logfire.debug(f"Cleaning up {len(self._background_tasks)} background event tasks")
                # Cancel all remaining tasks
                tasks_to_cancel = list(self._background_tasks)
                for task in tasks_to_cancel:
                    if not task.done():
                        task.cancel()

                # Wait for all tasks to finish or be cancelled
                if tasks_to_cancel:
                    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

                self._background_tasks.clear()

    async def get_event_history(self, request_id: str) -> list[ResearchEvent]:
        """Get event history for a specific request.

        Only returns events that belong to the current user context.
        """
        context = get_current_context()
        user_scope = context.get_scope_key()
        history_key = f"{user_scope}:{request_id}"
        async with self._history_lock:
            return list(self._event_history.get(history_key, []))

    async def clear_history(self, request_id: str) -> None:
        """Clear event history for a specific request.

        Only clears events that belong to the current user context.
        """
        context = get_current_context()
        user_scope = context.get_scope_key()
        history_key = f"{user_scope}:{request_id}"
        async with self._history_lock:
            if history_key in self._event_history:
                del self._event_history[history_key]

    async def get_stats(self) -> dict[str, Any]:
        """Get event bus statistics for monitoring.

        Returns both global and user-scoped statistics.
        """
        context = get_current_context()
        user_scope = context.get_scope_key()

        # Gather stats with appropriate locks
        async with self._handlers_lock:
            total_event_types = len(self._handlers)
            total_handlers = sum(len(handlers) for handlers in self._handlers.values())
            handlers_by_type = {
                event_type.__name__: len(handlers)
                for event_type, handlers in self._handlers.items()
            }

        async with self._history_lock:
            # Count user-specific requests
            user_requests = sum(
                1 for key in self._event_history.keys() if key.startswith(f"{user_scope}:")
            )
            total_events = sum(self._event_count_by_user.values())
            user_events = self._event_count_by_user.get(user_scope, 0)
            total_requests = len(self._event_history)
            active_users = len(self._active_users)

        async with self._tasks_lock:
            background_tasks = len(self._background_tasks)

        return {
            "total_event_types": total_event_types,
            "total_handlers": total_handlers,
            "total_events_emitted": total_events,
            "user_events_emitted": user_events,
            "background_tasks": background_tasks,
            "total_requests_tracked": total_requests,
            "user_requests_tracked": user_requests,
            "active_users": active_users,
            "handlers_by_type": handlers_by_type,
        }

    async def cleanup_user(self, user_id: str, session_id: str | None = None) -> None:
        """Clean up resources for a specific user/session.

        Args:
            user_id: User identifier
            session_id: Optional session identifier
        """
        # Build scope key
        if session_id:
            user_scope = f"{user_id}:{session_id}"
        else:
            user_scope = user_id

        # Remove user's event history with lock
        async with self._history_lock:
            keys_to_remove = [
                key for key in self._event_history.keys() if key.startswith(f"{user_scope}:")
            ]
            for key in keys_to_remove:
                del self._event_history[key]

            # Reset user's event count
            if user_scope in self._event_count_by_user:
                del self._event_count_by_user[user_scope]

            # Remove from active users
            self._active_users.discard(user_scope)

        logfire.info(
            f"Cleaned up resources for user: {user_scope}",
            removed_requests=len(keys_to_remove),
        )


# Global event bus instance - shared across the application
research_event_bus = ResearchEventBus()


# Convenience functions for common event emissions
async def emit_research_started(
    request_id: str,
    user_query: str,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit a research started event.

    Args:
        request_id: Unique request identifier
        user_query: The user's research query
        user_id: Optional user identifier
        metadata: Optional metadata dictionary
    """
    await research_event_bus.emit(
        ResearchStartedEvent(
            _request_id=request_id,
            user_query=user_query,
            user_id=user_id,
            metadata=metadata,
        )
    )


async def emit_stage_completed(
    request_id: str,
    stage: ResearchStage,
    success: bool,
    result: Any = None,
    error_message: str | None = None,
) -> None:
    """Emit a stage completed event.

    Args:
        request_id: Unique request identifier
        stage: The research stage that was completed
        success: Whether the stage completed successfully
        result: Optional result data from the stage
        error_message: Optional error message if stage failed
    """
    await research_event_bus.emit(
        StageCompletedEvent(
            _request_id=request_id,
            stage=stage,
            success=success,
            result=result,
            error_message=error_message,
        )
    )


async def emit_error(
    request_id: str,
    stage: ResearchStage,
    error_type: str,
    error_message: str,
    recoverable: bool = False,
) -> None:
    """Emit an error event.

    Args:
        request_id: Unique request identifier
        stage: The research stage where error occurred
        error_type: Type of error that occurred
        error_message: Detailed error message
        recoverable: Whether the error is recoverable
    """
    await research_event_bus.emit(
        ErrorEvent(
            _request_id=request_id,
            stage=stage,
            error_type=error_type,
            error_message=error_message,
            recoverable=recoverable,
        )
    )


# Clarification convenience functions
async def emit_clarification_requested(
    request_id: str,
    original_query: str,
    ambiguity_reasons: list[str],
    interaction_mode: str = "cli",
) -> None:
    """Emit a clarification requested event.

    Args:
        request_id: Unique request identifier
        original_query: The original user query
        ambiguity_reasons: List of reasons why clarification is needed
        interaction_mode: Mode of interaction ('cli' or 'http')
    """
    await research_event_bus.emit(
        ClarificationRequestedEvent(
            _request_id=request_id,
            original_query=original_query,
            ambiguity_reasons=ambiguity_reasons,
            interaction_mode=interaction_mode,
        )
    )


async def emit_clarification_session_created(
    request_id: str, session_id: str, num_questions: int, questions_preview: list[str]
) -> None:
    """Emit a clarification session created event.

    Args:
        request_id: Unique request identifier
        session_id: Clarification session identifier
        num_questions: Number of questions in the session
        questions_preview: Preview of question texts (first few words)
    """
    await research_event_bus.emit(
        ClarificationSessionCreatedEvent(
            _request_id=request_id,
            session_id=session_id,
            num_questions=num_questions,
            questions_preview=questions_preview,
        )
    )


async def emit_clarification_question_answered(
    request_id: str,
    session_id: str,
    question_id: str,
    question_text: str,
    selected_choice: str,
    free_text_response: str | None = None,
    completion_progress: float = 0.0,
) -> None:
    """Emit a clarification question answered event.

    Args:
        request_id: Unique request identifier
        session_id: Clarification session identifier
        question_id: Question identifier
        question_text: Full question text
        selected_choice: User's selected choice key
        free_text_response: Optional free text response for 'other'
        completion_progress: Progress ratio (0.0 to 1.0)
    """
    await research_event_bus.emit(
        ClarificationQuestionAnsweredEvent(
            _request_id=request_id,
            session_id=session_id,
            question_id=question_id,
            question_text=question_text,
            selected_choice=selected_choice,
            free_text_response=free_text_response,
            completion_progress=completion_progress,
        )
    )


async def emit_clarification_completed(
    request_id: str,
    session_id: str,
    original_query: str,
    refined_query: str,
    confidence_score: float,
    num_responses: int,
    completion_ratio: float,
    enhancements_applied: list[str],
) -> None:
    """Emit a clarification completed event.

    Args:
        request_id: Unique request identifier
        session_id: Clarification session identifier
        original_query: Original user query
        refined_query: Refined query after clarification
        confidence_score: Confidence score of the refinement
        num_responses: Number of responses received
        completion_ratio: Completion ratio of required questions
        enhancements_applied: List of enhancements applied to query
    """
    await research_event_bus.emit(
        ClarificationCompletedEvent(
            _request_id=request_id,
            session_id=session_id,
            original_query=original_query,
            refined_query=refined_query,
            confidence_score=confidence_score,
            num_responses=num_responses,
            completion_ratio=completion_ratio,
            enhancements_applied=enhancements_applied,
        )
    )


async def emit_clarification_cancelled(
    request_id: str, session_id: str, reason: str = "user_cancelled", partial_responses: int = 0
) -> None:
    """Emit a clarification cancelled event.

    Args:
        request_id: Unique request identifier
        session_id: Clarification session identifier
        reason: Reason for cancellation
        partial_responses: Number of questions answered before cancellation
    """
    await research_event_bus.emit(
        ClarificationCancelledEvent(
            _request_id=request_id,
            session_id=session_id,
            reason=reason,
            partial_responses=partial_responses,
        )
    )


async def emit_query_refined(
    request_id: str,
    original_query: str,
    refined_query: str,
    refinement_context: dict[str, str],
    confidence_score: float,
    enhancement_summary: str,
) -> None:
    """Emit a query refined event.

    Args:
        request_id: Unique request identifier
        original_query: Original user query
        refined_query: Refined query
        refinement_context: Context extracted from clarification responses
        confidence_score: Confidence score of the refinement
        enhancement_summary: Summary of enhancements applied
    """
    await research_event_bus.emit(
        QueryRefinedEvent(
            _request_id=request_id,
            original_query=original_query,
            refined_query=refined_query,
            refinement_context=refinement_context,
            confidence_score=confidence_score,
            enhancement_summary=enhancement_summary,
        )
    )
