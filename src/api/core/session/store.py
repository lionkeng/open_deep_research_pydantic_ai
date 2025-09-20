"""Session storage abstractions."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from .models import ResearchSession


class SessionStore(ABC):
    """Abstract base class for session persistence backends."""

    @abstractmethod
    async def get(self, session_id: str) -> ResearchSession | None: ...

    @abstractmethod
    async def set(self, session: ResearchSession, ttl: int | None = None) -> bool: ...

    @abstractmethod
    async def delete(self, session_id: str) -> bool: ...

    @abstractmethod
    async def exists(self, session_id: str) -> bool: ...

    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> list[str]: ...

    @abstractmethod
    async def cleanup_expired(self) -> int: ...


class InMemorySessionStore(SessionStore):
    """In-memory session store suitable for development and testing."""

    def __init__(self) -> None:
        self._sessions: dict[str, ResearchSession] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> ResearchSession | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            return session.model_copy(deep=True) if session else None

    async def set(self, session: ResearchSession, ttl: int | None = None) -> bool:  # noqa: ARG002
        async with self._lock:
            self._sessions[str(session.id)] = session.model_copy(deep=True)
            return True

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    async def exists(self, session_id: str) -> bool:
        async with self._lock:
            return session_id in self._sessions

    async def list_keys(self, pattern: str = "*") -> list[str]:  # noqa: ARG002
        async with self._lock:
            return list(self._sessions.keys())

    async def cleanup_expired(self) -> int:
        async with self._lock:
            to_delete = [
                session_id
                for session_id, session in self._sessions.items()
                if session.metadata.is_expired()
            ]
            for session_id in to_delete:
                self._sessions.pop(session_id, None)
            return len(to_delete)
