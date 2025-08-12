"""Request context management for concurrent research operations."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import logfire


@dataclass
class ResearchContext:
    """Context for a research request, providing user and session isolation.

    This context is used to:
    - Scope operations to specific users
    - Prevent cross-user data leakage
    - Enable per-user resource management
    - Support concurrent requests safely
    """

    user_id: str = "default"
    session_id: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] | None = None

    def get_scope_key(self) -> str:
        """Get the scope key for this context.

        Returns a unique key that can be used to namespace operations.
        Format: {user_id}:{session_id} or {user_id}
        """
        if self.session_id:
            return f"{self.user_id}:{self.session_id}"
        return self.user_id

    def matches_request(self, request_id: str) -> bool:
        """Check if a request ID belongs to this context.

        Args:
            request_id: Request ID to check

        Returns:
            True if the request belongs to this user/session
        """
        # Request IDs are formatted as user_id:session_id:uuid or user_id:uuid
        parts = request_id.split(":")
        if len(parts) < 2:
            return False

        # Check user_id match
        if parts[0] != self.user_id:
            return False

        # If we have a session_id, check it matches
        if self.session_id and len(parts) >= 3:
            return parts[1] == self.session_id

        return True

    def __str__(self) -> str:
        """String representation of the context."""
        if self.session_id:
            return f"ResearchContext(user={self.user_id}, session={self.session_id})"
        return f"ResearchContext(user={self.user_id})"


# Context variable for the current research context
# This uses Python's contextvars for async-safe context propagation
_current_context: ContextVar[ResearchContext | None] = ContextVar("research_context", default=None)


def get_current_context() -> ResearchContext:
    """Get the current research context.

    Returns:
        Current context or a default context if none is set
    """
    context = _current_context.get()
    if context is None:
        # Return default context for backward compatibility
        return ResearchContext()
    return context


def set_current_context(context: ResearchContext) -> None:
    """Set the current research context.

    Args:
        context: Research context to set
    """
    _current_context.set(context)
    logfire.debug(f"Set research context: {context}")


def clear_current_context() -> None:
    """Clear the current research context."""
    _current_context.set(None)
    logfire.debug("Cleared research context")


class ResearchContextManager:
    """Context manager for research context.

    Usage:
        async with ResearchContextManager(user_id="user123"):
            # All operations in this block will use this context
            await workflow.execute_research(...)
    """

    def __init__(
        self,
        user_id: str = "default",
        session_id: str | None = None,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize the context manager.

        Args:
            user_id: User identifier
            session_id: Optional session identifier
            request_id: Optional request identifier
            metadata: Optional metadata
        """
        self.context = ResearchContext(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            metadata=metadata or {},
        )
        self._token = None

    def __enter__(self):
        """Enter the context."""
        self._token = _current_context.set(self.context)
        logfire.debug(f"Entered research context: {self.context}")
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        _current_context.reset(self._token)
        logfire.debug(f"Exited research context: {self.context}")

    async def __aenter__(self):
        """Async enter the context."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit the context."""
        return self.__exit__(exc_type, exc_val, exc_tb)
