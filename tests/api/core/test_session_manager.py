"""Tests for the asynchronous session manager."""

import asyncio
import random

import pytest
import pytest_asyncio

from api.core.session import InMemorySessionStore, SessionManager, SessionState


@pytest_asyncio.fixture
async def session_manager():
    manager = SessionManager(store=InMemorySessionStore(), cleanup_interval=1)
    await manager.start()
    yield manager
    await manager.stop()


@pytest.mark.asyncio
async def test_concurrent_session_creation(session_manager: SessionManager) -> None:
    async def create_session(index: int) -> str:
        session = await session_manager.create_session(query=f"Test query {index}")
        return str(session.id)

    session_ids = await asyncio.gather(*(create_session(i) for i in range(10)))
    assert len(session_ids) == len(set(session_ids))


@pytest.mark.asyncio
async def test_concurrent_read_write(session_manager: SessionManager) -> None:
    session = await session_manager.create_session(query="Concurrent test")
    session_id = str(session.id)

    async def reader() -> None:
        for _ in range(5):
            loaded = await session_manager.get_session(session_id)
            assert loaded is not None
            await asyncio.sleep(random.uniform(0.001, 0.01))

    async def writer() -> None:
        for _ in range(3):
            loaded = await session_manager.get_session(session_id, for_update=True)
            assert loaded is not None
            loaded.record_error("simulated")
            await session_manager.update_session(loaded)
            await asyncio.sleep(random.uniform(0.001, 0.01))

    await asyncio.gather(*(reader() for _ in range(5)), *(writer() for _ in range(2)))

    final = await session_manager.get_session(session_id)
    assert final is not None
    assert final.error_count >= 6


@pytest.mark.asyncio
async def test_cleanup_removes_expired_sessions(session_manager: SessionManager) -> None:
    session = await session_manager.create_session(query="cleanup test")
    session.metadata.expires_at = session.metadata.created_at
    await session_manager.store.set(session)

    await session_manager.cleanup_expired()

    assert await session_manager.get_session(str(session.id)) is None


@pytest.mark.asyncio
async def test_state_transition(session_manager: SessionManager) -> None:
    session = await session_manager.create_session(query="state test")
    session_id = str(session.id)

    transitioned = await session_manager.transition_state(session_id, SessionState.SYNTHESIZING)
    assert transitioned is True

    loaded = await session_manager.get_session(session_id)
    assert loaded is not None
    assert loaded.state == SessionState.SYNTHESIZING
