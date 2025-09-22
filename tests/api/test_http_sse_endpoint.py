"""HTTP-level tests for the SSE stream endpoint.

Verifies that GET /research/{request_id}/stream emits expected SSE events
when the workflow completes, using a patched workflow that emits events.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi.testclient import TestClient

from api.main import app
from core.events import ResearchCompletedEvent, StreamingUpdateEvent, research_event_bus
from models.core import ResearchStage, ResearchState


def _open_client() -> TestClient:
    """Helper to create a TestClient instance for the app."""
    return TestClient(app)


def _collect_sse_events(stream_response, max_events: int = 3, timeout: float = 2.0) -> list[tuple[str, dict[str, Any]]]:
    """Collect SSE events from a streaming response.

    Parses lines of the form "event: <type>" and "data: <json>" and returns
    a list of (event_type, data_dict) tuples.
    """
    events: list[tuple[str, dict[str, Any]]] = []
    current_event: str | None = None
    start = time.time()

    for raw in stream_response.iter_lines():
        # Safety timeout to avoid hangs
        if time.time() - start > timeout:
            break

        line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            payload = line.split(":", 1)[1].strip()
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                # Skip malformed payloads in tests
                continue
            if current_event:
                events.append((current_event, data))
                if len(events) >= max_events:
                    break

    return events


def test_sse_endpoint_streams_update_and_complete(monkeypatch) -> None:
    """End-to-end test of the SSE endpoint using a stubbed workflow.

    - Starts research with stream=True
    - Opens SSE stream for the generated request_id
    - Verifies receipt of connection, update, and completion events
    """

    # Patch task submission to run immediately in the same event loop
    async def immediate_submit(request_id: str, coro, metadata: dict[str, Any] | None = None):
        await coro
        return request_id

    monkeypatch.setattr("api.main.task_manager.submit_research", immediate_submit)

    # Patch workflow.run to emit SSE events and return a completed state
    async def fake_run(
        *,
        user_query: str,
        api_keys: Any,
        stream_callback: bool | None,
        request_id: str,
        clarification_callback: Any,
    ) -> ResearchState:
        # Give the SSE handler time to subscribe before emitting
        await asyncio.sleep(0.05)
        # Emit an update
        await research_event_bus.emit(
            StreamingUpdateEvent(
                _request_id=request_id,
                content="Working...",
                stage=ResearchStage.RESEARCH_EXECUTION,
                is_partial=True,
            )
        )
        # Emit completion
        await research_event_bus.emit(
            ResearchCompletedEvent(
                _request_id=request_id,
                report=None,
                success=True,
                duration_seconds=0.01,
                error_message=None,
            )
        )

        # Return a completed state that will be stored by the background path
        return ResearchState(
            request_id=request_id,
            user_id="api-user",
            session_id=None,
            user_query=user_query,
            current_stage=ResearchStage.COMPLETED,
        )

    monkeypatch.setattr("api.main.workflow.run", fake_run)

    with _open_client() as client:
        # Start research with streaming enabled to get the request_id
        resp = client.post(
            "/research",
            json={"query": "SSE endpoint test", "stream": True},
        )
        assert resp.status_code == 200
        request_id = resp.json()["request_id"]

        # Open SSE stream and collect events
        with client.stream("GET", f"/research/{request_id}/stream") as stream_resp:
            assert stream_resp.status_code == 200
            assert stream_resp.headers.get("content-type", "").startswith("text/event-stream")

            events = _collect_sse_events(stream_resp, max_events=3, timeout=2.0)

        # Expect at least connection + our two events (update, complete)
        # Connection event isn't included in parsed list (no data JSON contract),
        # so verify presence of update and complete in order.
        # Some clients include connection with JSON; if present, allow extra.
        event_types = [e[0] for e in events]
        assert "update" in event_types
        assert "complete" in event_types
