"""Shared fixtures for API integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from api.main import app, active_sessions


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
