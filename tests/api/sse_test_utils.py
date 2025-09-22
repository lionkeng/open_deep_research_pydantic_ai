"""Utilities that support testing of SSE streaming behaviour."""

from __future__ import annotations

import json
from typing import Any

from sse_starlette.event import ServerSentEvent


def decode_sse_event(event: ServerSentEvent) -> dict[str, Any]:
    """Decode an SSE event into a comparable dictionary structure."""

    data: Any
    if event.data is None:
        data = None
    elif isinstance(event.data, (bytes, bytearray)):
        data = json.loads(event.data.decode("utf-8"))
    elif isinstance(event.data, str):
        data = json.loads(event.data)
    else:
        data = event.data

    return {
        "event": event.event,
        "id": event.id,
        "retry": event.retry,
        "data": data,
    }


__all__ = ["decode_sse_event"]
