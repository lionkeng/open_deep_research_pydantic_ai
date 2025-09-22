"""Session models for robust HTTP-mode state management."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from models.clarification import ClarificationRequest, ClarificationResponse


class ClarificationExchange(BaseModel):
    """Record of a clarification request/response pair."""

    request: ClarificationRequest
    response: ClarificationResponse | None = None
    requested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    responded_at: datetime | None = None
    timed_out: bool = False

    def record_response(self, response: ClarificationResponse) -> None:
        self.response = response
        self.responded_at = datetime.now(UTC)


class SessionConfig(BaseModel):
    """Configuration options for a research session."""

    ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    max_clarifications: int = Field(default=3, ge=0, le=10)
    clarification_timeout_seconds: int = Field(default=300, ge=30, le=1800)
    enable_caching: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0, le=10)


class SessionMetadata(BaseModel):
    """Lifecycle metadata for sessions."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    access_count: int = 0
    last_accessed_at: datetime | None = None
    client_ip: str | None = None
    user_agent: str | None = None

    def update_access(self) -> None:
        self.access_count += 1
        now = datetime.now(UTC)
        self.last_accessed_at = now
        self.updated_at = now

    def is_expired(self) -> bool:
        return bool(self.expires_at and datetime.now(UTC) > self.expires_at)


class SessionState(str, Enum):
    """State transitions for HTTP research sessions."""

    IDLE = "idle"
    RESEARCHING = "researching"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    CLARIFICATION_TIMEOUT = "clarification_timeout"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    ERROR = "error"
    EXPIRED = "expired"

    @classmethod
    def valid_transitions(cls) -> dict[SessionState, list[SessionState]]:
        return {
            cls.IDLE: [cls.RESEARCHING, cls.EXPIRED],
            cls.RESEARCHING: [
                cls.AWAITING_CLARIFICATION,
                cls.SYNTHESIZING,
                cls.COMPLETED,
                cls.ERROR,
            ],
            cls.AWAITING_CLARIFICATION: [
                cls.RESEARCHING,
                cls.CLARIFICATION_TIMEOUT,
                cls.ERROR,
            ],
            cls.CLARIFICATION_TIMEOUT: [cls.RESEARCHING, cls.COMPLETED, cls.ERROR],
            cls.SYNTHESIZING: [cls.COMPLETED, cls.ERROR],
            cls.COMPLETED: [cls.EXPIRED],
            cls.ERROR: [cls.IDLE, cls.EXPIRED],
            cls.EXPIRED: [],
        }

    def can_transition_to(self, target: SessionState) -> bool:
        return target in self.valid_transitions().get(self, [])


class ResearchSession(BaseModel):
    """Complete session record tracked by the HTTP API."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    state: SessionState = SessionState.IDLE
    config: SessionConfig = Field(default_factory=SessionConfig)
    metadata: SessionMetadata = Field(default_factory=SessionMetadata)

    query: str | None = None
    research_results: dict[str, Any] | None = None
    clarification_exchanges: list[ClarificationExchange] = Field(default_factory=list)
    synthesis_result: dict[str, Any] | None = None

    error_count: int = 0
    last_error: str | None = None
    error_history: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("metadata", mode="before")
    @classmethod
    def ensure_metadata(cls, value: Any) -> SessionMetadata:
        if isinstance(value, SessionMetadata):
            return value
        if isinstance(value, dict):
            return SessionMetadata(**value)
        return SessionMetadata()

    def transition_to(self, new_state: SessionState) -> bool:
        if not self.state.can_transition_to(new_state):
            return False
        self.state = new_state
        self.metadata.updated_at = datetime.now(UTC)
        return True

    def record_error(self, error: Exception | str) -> None:
        self.error_count += 1
        message = str(error)
        self.last_error = message
        self.error_history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "message": message,
                "state": self.state.value,
            }
        )
        self.metadata.updated_at = datetime.now(UTC)

    def apply_ttl(self) -> None:
        if self.config.ttl_seconds:
            self.metadata.expires_at = self.metadata.created_at + timedelta(
                seconds=self.config.ttl_seconds
            )

    def model_dump_safe(self) -> dict[str, Any]:
        """Dump the session into JSON-serialisable data."""
        return self.model_dump(mode="json")
