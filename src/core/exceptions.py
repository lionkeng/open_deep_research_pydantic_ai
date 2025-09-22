"""Domain-specific exception hierarchy for consistent error handling."""

from __future__ import annotations

from typing import Any


class OpenDeepResearchError(Exception):
    """Base exception for all expected application errors."""

    def __init__(
        self,
        *,
        message: str,
        error_code: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        """Serialise the error into a structured payload."""

        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class SessionError(OpenDeepResearchError):
    """Base exception for session-related failures."""

    def __init__(
        self,
        *,
        message: str,
        error_code: str,
        status_code: int = 400,
        **details: Any,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class SessionNotFoundError(SessionError):
    """Raised when a session cannot be located."""

    def __init__(self, session_id: str) -> None:
        super().__init__(
            message=f"Session {session_id} not found",
            error_code="SESSION_NOT_FOUND",
            status_code=404,
            session_id=session_id,
        )


class SessionExpiredError(SessionError):
    """Raised when accessing an expired session."""

    def __init__(self, session_id: str) -> None:
        super().__init__(
            message=f"Session {session_id} has expired",
            error_code="SESSION_EXPIRED",
            status_code=410,
            session_id=session_id,
        )


class SessionStateError(SessionError):
    """Raised when a session is in an invalid state for the requested action."""

    def __init__(self, session_id: str, current_state: str, allowed_states: list[str]) -> None:
        super().__init__(
            message=(
                f"Session {session_id} is in state '{current_state}' and cannot"
                " transition to the requested state"
            ),
            error_code="SESSION_INVALID_STATE",
            status_code=409,
            session_id=session_id,
            current_state=current_state,
            allowed_states=allowed_states,
        )


class ClarificationError(OpenDeepResearchError):
    """Base exception for clarification flow issues."""

    def __init__(
        self,
        *,
        message: str,
        error_code: str,
        status_code: int = 400,
        **details: Any,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class ClarificationLimitError(ClarificationError):
    """Raised when clarification limit is exceeded."""

    def __init__(self, session_id: str, limit: int) -> None:
        super().__init__(
            message=f"Session {session_id} exceeded clarification limit of {limit}",
            error_code="CLARIFICATION_LIMIT_EXCEEDED",
            status_code=429,
            session_id=session_id,
            limit=limit,
        )


class ClarificationStateError(ClarificationError):
    """Raised when clarification flow is not in an expected state."""

    def __init__(self, session_id: str, reason: str, status_code: int = 409) -> None:
        super().__init__(
            message=f"Clarification state error for session {session_id}: {reason}",
            error_code="CLARIFICATION_STATE_ERROR",
            status_code=status_code,
            session_id=session_id,
            reason=reason,
        )


class ClarificationTimeoutError(ClarificationError):
    """Raised when clarification flow times out."""

    def __init__(self, session_id: str, timeout_seconds: int) -> None:
        super().__init__(
            message=(
                "Clarification timed out after"
                f" {timeout_seconds} seconds for session {session_id}"
            ),
            error_code="CLARIFICATION_TIMEOUT",
            status_code=408,
            session_id=session_id,
            timeout_seconds=timeout_seconds,
        )


class ExternalServiceError(OpenDeepResearchError):
    """Raised when an external service dependency fails."""

    def __init__(
        self,
        *,
        service: str,
        message: str,
        original_error: Exception | None = None,
        status_code: int = 503,
    ) -> None:
        details = {"service": service}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(
            message=f"{service} error: {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=status_code,
            details=details,
        )


class RateLimitError(OpenDeepResearchError):
    """Raised when API rate limits are exceeded."""

    def __init__(self, *, limit: int, window_seconds: int) -> None:
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"limit": limit, "window_seconds": window_seconds},
        )


__all__ = [
    "OpenDeepResearchError",
    "SessionError",
    "SessionNotFoundError",
    "SessionExpiredError",
    "SessionStateError",
    "ClarificationError",
    "ClarificationLimitError",
    "ClarificationTimeoutError",
    "ClarificationStateError",
    "ExternalServiceError",
    "RateLimitError",
]
