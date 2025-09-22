"""Shared fixtures for API integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from api.main import app, active_sessions


@pytest_asyncio.fixture
async def mock_research_event_bus(monkeypatch):
    """Provide an isolated ResearchEventBus instance for tests."""

    from core.events import ResearchEventBus

    bus = ResearchEventBus()

    # Ensure both core.events and api.sse_handler reference the isolated bus
    monkeypatch.setattr("core.events.research_event_bus", bus)
    monkeypatch.setattr("api.sse_handler.research_event_bus", bus)

    try:
        yield bus
    finally:
        await bus.cleanup()


@pytest.fixture(scope="session", autouse=True)
def load_env_from_dotenv() -> None:
    """Ensure environment variables from .env are available during tests."""
    dotenv_path = Path(".env")
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)
    else:
        load_dotenv(override=False)


@pytest.fixture
def client():
    """Provide a fresh TestClient with isolated active session state."""
    active_sessions.clear()
    with TestClient(app) as test_client:
        yield test_client
    active_sessions.clear()
