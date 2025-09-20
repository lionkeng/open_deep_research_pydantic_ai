"""Session manager that coordinates storage, locks, and cleanup."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import logfire

from core.resilience import retry_async

from ..locks import AsyncReadWriteLock
from .models import (
    ClarificationExchange,
    ResearchSession,
    SessionConfig,
    SessionMetadata,
    SessionState,
)
from .store import InMemorySessionStore, SessionStore


class SessionManager:
    """High-level API for creating, updating, and cleaning research sessions."""

    def __init__(self, store: SessionStore | None = None, cleanup_interval: int = 60) -> None:
        self.store: SessionStore = store or InMemorySessionStore()
        self._cleanup_interval = cleanup_interval
        self._locks: dict[str, AsyncReadWriteLock] = {}
        self._locks_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logfire.info("Session manager started")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logfire.info("Session manager stopped")

    async def create_session(
        self,
        query: str,
        *,
        session_id: str | None = None,
        config: SessionConfig | None = None,
        metadata: SessionMetadata | None = None,
    ) -> ResearchSession:
        session = ResearchSession(
            id=session_id or str(uuid4()),
            query=query,
            config=config or SessionConfig(),
            metadata=metadata or SessionMetadata(),
            state=SessionState.RESEARCHING,
        )
        session.metadata.update_access()
        session.apply_ttl()
        await retry_async(lambda: self.store.set(session, ttl=session.config.ttl_seconds))
        await self._ensure_lock(session.id)
        logfire.debug("Session created", session_id=session.id)
        return session

    async def get_session(
        self, session_id: str, *, for_update: bool = False
    ) -> ResearchSession | None:
        lock = await self._ensure_lock(session_id)
        ctx = lock.write_locked() if for_update else lock.read_locked()
        async with ctx:  # type: ignore[arg-type]
            session = await retry_async(lambda: self.store.get(session_id))
            if session:
                session.metadata.update_access()
                await retry_async(lambda: self.store.set(session))
            return session

    async def update_session(self, session: ResearchSession) -> bool:
        session.metadata.updated_at = datetime.now(UTC)
        session.apply_ttl()
        lock = await self._ensure_lock(session.id)
        async with lock.write_locked():
            success = await retry_async(
                lambda: self.store.set(session, ttl=session.config.ttl_seconds)
            )
            if success:
                logfire.debug("Session updated", session_id=session.id)
            return success

    async def transition_state(self, session_id: str, new_state: SessionState) -> bool:
        lock = await self._ensure_lock(session_id)
        async with lock.write_locked():
            session = await retry_async(lambda: self.store.get(session_id))
            if not session:
                logfire.warning(
                    "Attempted state transition for missing session",
                    session_id=session_id,
                )
                return False
            if not session.transition_to(new_state):
                logfire.warning(
                    "Invalid session transition",
                    session_id=session_id,
                    from_state=session.state.value,
                    to_state=new_state.value,
                )
                return False
            success = await retry_async(
                lambda: self.store.set(session, ttl=session.config.ttl_seconds)
            )
            if success:
                logfire.debug("Session transitioned", session_id=session_id, state=new_state.value)
            return success

    async def append_clarification_exchange(
        self, session_id: str, exchange: ClarificationExchange
    ) -> bool:
        lock = await self._ensure_lock(session_id)
        async with lock.write_locked():
            session = await retry_async(lambda: self.store.get(session_id))
            if not session:
                return False
            session.clarification_exchanges.append(exchange)
            session.metadata.updated_at = datetime.now(UTC)
            success = await retry_async(
                lambda: self.store.set(session, ttl=session.config.ttl_seconds)
            )
            return success

    async def cleanup_expired(self) -> int:
        deleted = await retry_async(self.store.cleanup_expired)
        if deleted:
            async with self._locks_lock:
                active = set(await retry_async(self.store.list_keys))
                to_remove = [sid for sid in self._locks.keys() if sid not in active]
                for sid in to_remove:
                    self._locks.pop(sid, None)
        return deleted

    async def list_sessions(self) -> list[str]:
        return await retry_async(self.store.list_keys)

    async def delete_session(self, session_id: str) -> bool:
        deleted = await retry_async(lambda: self.store.delete(session_id))
        if deleted:
            async with self._locks_lock:
                self._locks.pop(session_id, None)
        return deleted

    async def _ensure_lock(self, session_id: str) -> AsyncReadWriteLock:
        async with self._locks_lock:
            if session_id not in self._locks:
                self._locks[session_id] = AsyncReadWriteLock(name=f"session:{session_id}")
            return self._locks[session_id]

    async def _cleanup_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self._cleanup_interval)
                deleted = await self.cleanup_expired()
                if deleted:
                    logfire.info("Cleaned expired sessions", count=deleted)
        except asyncio.CancelledError:  # pragma: no cover - expected during shutdown
            pass
        except Exception as exc:  # pragma: no cover
            logfire.exception("Session cleanup loop failed", error=str(exc))
