"""Pydantic models and enums for Server-Sent Events (SSE) messaging."""

import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SSEEventType(str, Enum):
    """SSE event field values.

    These define the 'event' field in SSE messages sent over the wire.
    """

    CONNECTION = "connection"  # Initial connection established
    PING = "ping"  # Heartbeat/keep-alive signal
    UPDATE = "update"  # Streaming content update
    STAGE_COMPLETED = "stage"  # Stage completion (note: wire format is "stage")
    ERROR = "error"  # Error occurred
    COMPLETE = "complete"  # Research completed


class SSEDataType(str, Enum):
    """SSE data type field values.

    These define the 'type' field within the JSON payload of SSE messages.
    """

    # Connection types
    CONNECTED = "connected"  # Initial connection confirmation
    HEARTBEAT = "heartbeat"  # Keep-alive heartbeat
    PING = "ping"  # Ping response

    # Update types
    UPDATE = "update"  # Content update
    STAGE_COMPLETED = "stage_completed"  # Stage completed

    # Error types
    ERROR = "error"  # General error
    STREAM_ERROR = "stream_error"  # SSE stream error

    # Completion types
    COMPLETED = "completed"  # Research completed


# Base SSE message models
class BaseSSEMessage(BaseModel):
    """Base model for all SSE message payloads."""

    pass  # type field will be defined in subclasses


class ConnectionMessage(BaseSSEMessage):
    """Initial connection confirmation message."""

    type: Literal[SSEDataType.CONNECTED] = SSEDataType.CONNECTED
    request_id: str
    message: str


class UpdateMessage(BaseSSEMessage):
    """Streaming update message during research."""

    type: Literal[SSEDataType.UPDATE] = SSEDataType.UPDATE
    request_id: str
    stage: str
    content: str
    is_partial: bool = False


class StageCompletedMessage(BaseSSEMessage):
    """Stage completion notification."""

    type: Literal[SSEDataType.STAGE_COMPLETED] = SSEDataType.STAGE_COMPLETED
    request_id: str
    stage: str
    success: bool
    result: Any = None
    error: str | None = None


class ErrorMessage(BaseSSEMessage):
    """Error notification during research."""

    type: Literal[SSEDataType.ERROR] = SSEDataType.ERROR
    request_id: str
    stage: str
    error_type: str | None = None
    message: str = Field(..., description="Error message")
    recoverable: bool = False


class CompletedMessage(BaseSSEMessage):
    """Research completion notification."""

    type: Literal[SSEDataType.COMPLETED] = SSEDataType.COMPLETED
    request_id: str
    success: bool
    duration: float | None = Field(None, description="Duration in seconds")
    error: str | None = None
    has_report: bool = False
    report: dict[str, Any] | None = None


class StreamErrorMessage(BaseSSEMessage):
    """SSE stream error notification."""

    type: Literal[SSEDataType.STREAM_ERROR] = SSEDataType.STREAM_ERROR
    message: str


class HeartbeatMessage(BaseSSEMessage):
    """Keep-alive heartbeat message."""

    type: Literal[SSEDataType.HEARTBEAT] = SSEDataType.HEARTBEAT


class PingMessage(BaseSSEMessage):
    """Ping response message."""

    type: Literal[SSEDataType.PING] = SSEDataType.PING


# Union type for all possible SSE messages
SSEMessage = (
    ConnectionMessage
    | UpdateMessage
    | StageCompletedMessage
    | ErrorMessage
    | CompletedMessage
    | StreamErrorMessage
    | HeartbeatMessage
    | PingMessage
)


def parse_sse_message(data: str) -> SSEMessage:
    """Parse JSON string into appropriate SSE message model.

    Args:
        data: JSON string containing SSE message

    Returns:
        Parsed SSE message model

    Raises:
        ValidationError: If data doesn't match any message schema
    """
    # Parse JSON first
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    # Determine message type and parse accordingly
    msg_type = parsed.get("type")

    if msg_type == SSEDataType.CONNECTED:
        return ConnectionMessage.model_validate(parsed)
    if msg_type == SSEDataType.UPDATE:
        return UpdateMessage.model_validate(parsed)
    if msg_type == SSEDataType.STAGE_COMPLETED:
        return StageCompletedMessage.model_validate(parsed)
    if msg_type == SSEDataType.ERROR:
        return ErrorMessage.model_validate(parsed)
    if msg_type == SSEDataType.COMPLETED:
        return CompletedMessage.model_validate(parsed)
    if msg_type == SSEDataType.STREAM_ERROR:
        return StreamErrorMessage.model_validate(parsed)
    if msg_type == SSEDataType.HEARTBEAT:
        return HeartbeatMessage.model_validate(parsed)
    if msg_type == SSEDataType.PING:
        return PingMessage.model_validate(parsed)
    raise ValueError(f"Unknown message type: {msg_type}")
