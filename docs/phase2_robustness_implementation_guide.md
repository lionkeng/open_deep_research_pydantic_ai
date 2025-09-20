# Phase 2: Robustness & State Management - Implementation Guide

## Executive Summary

This guide provides detailed implementation instructions for Phase 2 of the HTTP Mode API Alignment project, focusing on robustness and state management. Phase 2 transforms the basic HTTP server from Phase 1 into a production-ready system capable of handling concurrent requests, managing session state, and recovering from failures gracefully.

## Prerequisites

### Phase 1 Completion Checklist
- ✅ Basic FastAPI server running (`src/api/main.py`)
- ✅ Core `/research` endpoint functional
- ✅ Pydantic models for request/response validation
- ✅ Integration with workflow singleton
- ✅ Background task manager implemented

### Required Dependencies
```bash
# Add new dependencies
uv add redis aioredis asyncio-lock tenacity circuitbreaker
```

> **Phase 2c update:** Redis integration is deferred. Continue to rely on the in-memory store during this sub-phase and focus on wiring retries, circuit breakers, and error handling infrastructure.

## Technical Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         FastAPI App                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Routers    │  │  Middleware  │  │   SSE Handler│     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
│  ┌──────▼──────────────────▼──────────────────▼────────┐   │
│  │              Session Manager with RWLock             │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │           Redis/In-Memory Session Store              │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │         Two-Phase Clarification Handler              │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │      Circuit Breaker & Error Recovery System         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### State Machine Design

```python
from enum import Enum

class SessionState(str, Enum):
    """Session state machine states."""
    IDLE = "idle"
    RESEARCHING = "researching"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    CLARIFICATION_TIMEOUT = "clarification_timeout"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    ERROR = "error"
    EXPIRED = "expired"
```

## Day 3: Enhanced Session Management

### File Structure
```
src/
├── core/
│   ├── session/
│   │   ├── __init__.py
│   │   ├── manager.py        # Session manager with Redis backend
│   │   ├── models.py         # Session data models
│   │   ├── store.py          # Storage abstraction layer
│   │   └── cleanup.py        # TTL and cleanup service
│   └── locks/
│       ├── __init__.py
│       └── rwlock.py         # Read-write lock implementation
```

### Implementation Steps

#### Step 1: Session Models (`src/api/core/session/models.py`)

```python
"""Session models for state management."""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from models.clarification import ClarificationExchange
from models.research_executor import ResearchResults


class SessionConfig(BaseModel):
    """Configuration for a research session."""

    ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    max_clarifications: int = Field(default=3, ge=0, le=10)
    clarification_timeout_seconds: int = Field(default=300, ge=30, le=1800)
    enable_caching: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0, le=10)


class SessionMetadata(BaseModel):
    """Metadata for tracking session lifecycle."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(default=None)
    access_count: int = Field(default=0)
    last_accessed_at: Optional[datetime] = Field(default=None)
    client_ip: Optional[str] = Field(default=None)
    user_agent: Optional[str] = Field(default=None)

    def update_access(self) -> None:
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class SessionState(str, Enum):
    """Session state machine states."""
    IDLE = "idle"
    RESEARCHING = "researching"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    CLARIFICATION_TIMEOUT = "clarification_timeout"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    ERROR = "error"
    EXPIRED = "expired"

    @classmethod
    def valid_transitions(cls) -> Dict[str, List[str]]:
        """Define valid state transitions."""
        return {
            cls.IDLE: [cls.RESEARCHING, cls.EXPIRED],
            cls.RESEARCHING: [cls.AWAITING_CLARIFICATION, cls.SYNTHESIZING, cls.COMPLETED, cls.ERROR],
            cls.AWAITING_CLARIFICATION: [cls.RESEARCHING, cls.CLARIFICATION_TIMEOUT, cls.ERROR],
            cls.CLARIFICATION_TIMEOUT: [cls.RESEARCHING, cls.COMPLETED, cls.ERROR],
            cls.SYNTHESIZING: [cls.COMPLETED, cls.ERROR],
            cls.COMPLETED: [cls.EXPIRED],
            cls.ERROR: [cls.IDLE, cls.EXPIRED],
            cls.EXPIRED: []
        }

    def can_transition_to(self, target: "SessionState") -> bool:
        """Check if transition to target state is valid."""
        return target.value in self.valid_transitions().get(self.value, [])


class ResearchSession(BaseModel):
    """Complete session model for research workflow."""

    id: UUID = Field(default_factory=uuid4)
    state: SessionState = Field(default=SessionState.IDLE)
    config: SessionConfig = Field(default_factory=SessionConfig)
    metadata: SessionMetadata = Field(default_factory=SessionMetadata)

    # Research context
    query: Optional[str] = Field(default=None)
    research_results: Optional[ResearchResults] = Field(default=None)
    clarification_exchanges: List[ClarificationExchange] = Field(default_factory=list)
    synthesis_result: Optional[Dict[str, Any]] = Field(default=None)

    # Error tracking
    error_count: int = Field(default=0)
    last_error: Optional[str] = Field(default=None)
    error_history: List[Dict[str, Any]] = Field(default_factory=list)

    def transition_to(self, new_state: SessionState) -> bool:
        """Attempt state transition."""
        if not self.state.can_transition_to(new_state):
            return False
        self.state = new_state
        self.metadata.updated_at = datetime.now(timezone.utc)
        return True

    def record_error(self, error: Exception) -> None:
        """Record an error in the session."""
        self.error_count += 1
        self.last_error = str(error)
        self.error_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(error),
            "state": self.state.value
        })
        self.metadata.updated_at = datetime.now(timezone.utc)

    def model_dump_safe(self) -> Dict[str, Any]:
        """Dump model for storage, handling datetime serialization."""
        data = self.model_dump()
        # Convert datetime objects to ISO format strings
        def convert_datetime(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj
        return convert_datetime(data)
```

#### Step 2: Read-Write Lock (`src/api/core/locks/rwlock.py`)

```python
"""Async read-write lock implementation for session concurrency control."""

import asyncio
from typing import Optional

import logfire


class AsyncReadWriteLock:
    """
    Async read-write lock for managing concurrent session access.

    Allows multiple concurrent readers but only one writer at a time.
    Writers have priority over readers to prevent starvation.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize the read-write lock."""
        self.name = name or "rwlock"
        self._read_ready = asyncio.Condition()
        self._readers = 0
        self._writers = 0
        self._write_waiters = 0
        self._read_waiters = 0

    @logfire.instrument("Acquiring read lock")
    async def acquire_read(self) -> None:
        """Acquire a read lock."""
        async with self._read_ready:
            self._read_waiters += 1
            try:
                # Wait while there are writers or waiting writers
                while self._writers > 0 or self._write_waiters > 0:
                    await self._read_ready.wait()
                self._readers += 1
            finally:
                self._read_waiters -= 1
            logfire.debug(f"Read lock acquired for {self.name}, readers={self._readers}")

    @logfire.instrument("Releasing read lock")
    async def release_read(self) -> None:
        """Release a read lock."""
        async with self._read_ready:
            if self._readers <= 0:
                raise RuntimeError("release_read() called without acquire_read()")
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
            logfire.debug(f"Read lock released for {self.name}, readers={self._readers}")

    @logfire.instrument("Acquiring write lock")
    async def acquire_write(self) -> None:
        """Acquire a write lock."""
        async with self._read_ready:
            self._write_waiters += 1
            try:
                # Wait while there are readers or other writers
                while self._readers > 0 or self._writers > 0:
                    await self._read_ready.wait()
                self._writers += 1
            finally:
                self._write_waiters -= 1
            logfire.debug(f"Write lock acquired for {self.name}")

    @logfire.instrument("Releasing write lock")
    async def release_write(self) -> None:
        """Release a write lock."""
        async with self._read_ready:
            if self._writers <= 0:
                raise RuntimeError("release_write() called without acquire_write()")
            self._writers -= 1
            self._read_ready.notify_all()
            logfire.debug(f"Write lock released for {self.name}")

    def read_locked(self):
        """Context manager for read locking."""
        return _ReadLockContext(self)

    def write_locked(self):
        """Context manager for write locking."""
        return _WriteLockContext(self)


class _ReadLockContext:
    """Context manager for read lock."""

    def __init__(self, lock: AsyncReadWriteLock):
        self.lock = lock

    async def __aenter__(self):
        await self.lock.acquire_read()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.lock.release_read()


class _WriteLockContext:
    """Context manager for write lock."""

    def __init__(self, lock: AsyncReadWriteLock):
        self.lock = lock

    async def __aenter__(self):
        await self.lock.acquire_write()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.lock.release_write()
```

#### Step 3: Storage Abstraction (`src/api/core/session/store.py`)

```python
"""Storage abstraction layer for session persistence."""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import logfire
from tenacity import retry, stop_after_attempt, wait_exponential

from api.core.session.models import ResearchSession


class SessionStore(ABC):
    """Abstract base class for session storage."""

    @abstractmethod
    async def get(self, session_id: str) -> Optional[ResearchSession]:
        """Retrieve a session by ID."""
        pass

    @abstractmethod
    async def set(self, session: ResearchSession, ttl: Optional[int] = None) -> bool:
        """Store a session with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        pass

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        pass

    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List all session keys matching pattern."""
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired sessions, return count deleted."""
        pass


class RedisSessionStore(SessionStore):
    """Redis-based session storage."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis session store."""
        self.redis_url = redis_url
        self._redis = None
        self.key_prefix = "session:"

    async def connect(self) -> None:
        """Establish Redis connection."""
        if not self._redis:
            import aioredis
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50,
            )
            logfire.info("Connected to Redis session store")

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logfire.info("Disconnected from Redis session store")

    def _make_key(self, session_id: str) -> str:
        """Create Redis key from session ID."""
        return f"{self.key_prefix}{session_id}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    @logfire.instrument("Redis get session")
    async def get(self, session_id: str) -> Optional[ResearchSession]:
        """Retrieve session from Redis."""
        if not self._redis:
            await self.connect()

        try:
            key = self._make_key(session_id)
            data = await self._redis.get(key)

            if data is None:
                return None

            session_dict = json.loads(data)
            return ResearchSession.model_validate(session_dict)
        except Exception as e:
            logfire.error(f"Error retrieving session {session_id}: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    @logfire.instrument("Redis set session")
    async def set(self, session: ResearchSession, ttl: Optional[int] = None) -> bool:
        """Store session in Redis with TTL."""
        if not self._redis:
            await self.connect()

        try:
            key = self._make_key(str(session.id))
            data = json.dumps(session.model_dump_safe())

            if ttl:
                await self._redis.setex(key, ttl, data)
            else:
                await self._redis.set(key, data)

            return True
        except Exception as e:
            logfire.error(f"Error storing session {session.id}: {e}")
            return False

    async def delete(self, session_id: str) -> bool:
        """Delete session from Redis."""
        if not self._redis:
            await self.connect()

        try:
            key = self._make_key(session_id)
            result = await self._redis.delete(key)
            return result > 0
        except Exception as e:
            logfire.error(f"Error deleting session {session_id}: {e}")
            return False

    async def exists(self, session_id: str) -> bool:
        """Check if session exists in Redis."""
        if not self._redis:
            await self.connect()

        key = self._make_key(session_id)
        return bool(await self._redis.exists(key))

    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List all session keys matching pattern."""
        if not self._redis:
            await self.connect()

        search_pattern = f"{self.key_prefix}{pattern}"
        keys = await self._redis.keys(search_pattern)
        # Strip prefix from keys
        return [k.replace(self.key_prefix, "") for k in keys]

    async def cleanup_expired(self) -> int:
        """Clean up expired sessions."""
        if not self._redis:
            await self.connect()

        deleted_count = 0
        keys = await self._redis.keys(f"{self.key_prefix}*")

        for key in keys:
            try:
                data = await self._redis.get(key)
                if data:
                    session_dict = json.loads(data)
                    session = ResearchSession.model_validate(session_dict)
                    if session.metadata.is_expired():
                        await self._redis.delete(key)
                        deleted_count += 1
                        logfire.info(f"Deleted expired session: {session.id}")
            except Exception as e:
                logfire.error(f"Error checking session expiry for {key}: {e}")

        return deleted_count


class InMemorySessionStore(SessionStore):
    """In-memory session storage for development/testing."""

    def __init__(self):
        """Initialize in-memory store."""
        self._sessions: Dict[str, ResearchSession] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> Optional[ResearchSession]:
        """Retrieve session from memory."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def set(self, session: ResearchSession, ttl: Optional[int] = None) -> bool:
        """Store session in memory."""
        async with self._lock:
            self._sessions[str(session.id)] = session
            return True

    async def delete(self, session_id: str) -> bool:
        """Delete session from memory."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    async def exists(self, session_id: str) -> bool:
        """Check if session exists in memory."""
        async with self._lock:
            return session_id in self._sessions

    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List all session keys."""
        async with self._lock:
            if pattern == "*":
                return list(self._sessions.keys())
            # Simple pattern matching for development
            return [k for k in self._sessions.keys() if pattern.replace("*", "") in k]

    async def cleanup_expired(self) -> int:
        """Clean up expired sessions from memory."""
        async with self._lock:
            expired = []
            for session_id, session in self._sessions.items():
                if session.metadata.is_expired():
                    expired.append(session_id)

            for session_id in expired:
                del self._sessions[session_id]

            return len(expired)
```

#### Step 4: Session Manager (`src/api/core/session/manager.py`)

```python
"""Session manager for handling session lifecycle."""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import logfire

from api.core.locks.rwlock import AsyncReadWriteLock
from api.core.session.models import ResearchSession, SessionConfig, SessionState
from api.core.session.store import InMemorySessionStore, RedisSessionStore, SessionStore


class SessionManager:
    """
    Manages research session lifecycle with concurrent access control.

    Features:
    - Thread-safe session management with read-write locks
    - Automatic TTL-based cleanup
    - Session state machine enforcement
    - Persistent storage with Redis (or in-memory for development)
    """

    def __init__(self, store: Optional[SessionStore] = None):
        """Initialize session manager."""
        if store is None:
            # Use Redis in production, in-memory for development
            if os.getenv("REDIS_URL"):
                self.store = RedisSessionStore(os.getenv("REDIS_URL"))
            else:
                logfire.warning("Using in-memory session store (not suitable for production)")
                self.store = InMemorySessionStore()
        else:
            self.store = store

        self._locks: Dict[str, AsyncReadWriteLock] = {}
        self._locks_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 60  # seconds

    async def start(self) -> None:
        """Start the session manager and cleanup service."""
        if isinstance(self.store, RedisSessionStore):
            await self.store.connect()

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logfire.info("Session manager started")

    async def stop(self) -> None:
        """Stop the session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if isinstance(self.store, RedisSessionStore):
            await self.store.disconnect()

        logfire.info("Session manager stopped")

    async def _get_lock(self, session_id: str) -> AsyncReadWriteLock:
        """Get or create a lock for a session."""
        async with self._locks_lock:
            if session_id not in self._locks:
                self._locks[session_id] = AsyncReadWriteLock(f"session:{session_id}")
            return self._locks[session_id]

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up expired sessions."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                deleted_count = await self.cleanup_expired()
                if deleted_count > 0:
                    logfire.info(f"Cleaned up {deleted_count} expired sessions")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logfire.error(f"Error in cleanup loop: {e}")

    @logfire.instrument("Create session")
    async def create_session(
        self,
        query: str,
        config: Optional[SessionConfig] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> ResearchSession:
        """Create a new research session."""
        session = ResearchSession(
            query=query,
            config=config or SessionConfig(),
        )

        # Set metadata
        session.metadata.client_ip = client_ip
        session.metadata.user_agent = user_agent

        # Calculate expiry time
        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=session.config.ttl_seconds
        )
        session.metadata.expires_at = expires_at

        # Store session
        success = await self.store.set(session, ttl=session.config.ttl_seconds)
        if not success:
            raise RuntimeError("Failed to create session")

        logfire.info(f"Created session {session.id} for query: {query[:50]}...")
        return session

    @logfire.instrument("Get session")
    async def get_session(
        self,
        session_id: str,
        for_update: bool = False,
    ) -> Optional[ResearchSession]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier
            for_update: If True, acquires write lock; otherwise read lock
        """
        lock = await self._get_lock(session_id)

        if for_update:
            async with lock.write_locked():
                session = await self.store.get(session_id)
                if session:
                    session.metadata.update_access()
                    await self.store.set(session)
                return session
        else:
            async with lock.read_locked():
                session = await self.store.get(session_id)
                if session:
                    # Update access count in background
                    asyncio.create_task(self._update_access_async(session_id))
                return session

    async def _update_access_async(self, session_id: str) -> None:
        """Update session access count asynchronously."""
        try:
            lock = await self._get_lock(session_id)
            async with lock.write_locked():
                session = await self.store.get(session_id)
                if session:
                    session.metadata.update_access()
                    await self.store.set(session)
        except Exception as e:
            logfire.error(f"Error updating session access: {e}")

    @logfire.instrument("Update session")
    async def update_session(self, session: ResearchSession) -> bool:
        """Update an existing session."""
        session_id = str(session.id)
        lock = await self._get_lock(session_id)

        async with lock.write_locked():
            session.metadata.updated_at = datetime.now(timezone.utc)
            return await self.store.set(session)

    @logfire.instrument("Transition session state")
    async def transition_state(
        self,
        session_id: str,
        new_state: SessionState,
    ) -> bool:
        """
        Transition a session to a new state.

        Returns:
            True if transition was successful, False otherwise
        """
        lock = await self._get_lock(session_id)

        async with lock.write_locked():
            session = await self.store.get(session_id)
            if not session:
                logfire.error(f"Session {session_id} not found")
                return False

            if not session.state.can_transition_to(new_state):
                logfire.warning(
                    f"Invalid state transition for session {session_id}: "
                    f"{session.state} -> {new_state}"
                )
                return False

            session.state = new_state
            session.metadata.updated_at = datetime.now(timezone.utc)

            success = await self.store.set(session)
            if success:
                logfire.info(f"Session {session_id} transitioned to {new_state}")

            return success

    async def list_sessions(self, pattern: str = "*") -> List[str]:
        """List all active session IDs."""
        return await self.store.list_keys(pattern)

    async def cleanup_expired(self) -> int:
        """Clean up expired sessions."""
        deleted_count = await self.store.cleanup_expired()

        # Clean up orphaned locks
        async with self._locks_lock:
            active_sessions = set(await self.store.list_keys())
            orphaned_locks = set(self._locks.keys()) - active_sessions
            for session_id in orphaned_locks:
                del self._locks[session_id]

            if orphaned_locks:
                logfire.info(f"Cleaned up {len(orphaned_locks)} orphaned locks")

        return deleted_count
```

## Day 4: Two-Phase Clarification Flow

### Implementation Steps

#### Step 1: Clarification Handler (`src/api/core/clarification/handler.py`)

```python
"""Two-phase clarification flow handler."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional

import logfire

from api.core.session.manager import SessionManager
from api.core.session.models import SessionState
from models.clarification import (
    ClarificationExchange,
    ClarificationRequest,
    ClarificationResponse,
)


class ClarificationHandler:
    """
    Manages two-phase clarification flow.

    Phase 1: Generate and send clarification request
    Phase 2: Receive and process clarification response
    """

    def __init__(self, session_manager: SessionManager):
        """Initialize clarification handler."""
        self.session_manager = session_manager
        self._pending_clarifications: Dict[str, asyncio.Future] = {}

    @logfire.instrument("Request clarification")
    async def request_clarification(
        self,
        session_id: str,
        request: ClarificationRequest,
    ) -> ClarificationExchange:
        """
        Phase 1: Request clarification from user.

        Args:
            session_id: Session identifier
            request: Clarification request details

        Returns:
            ClarificationExchange with pending status
        """
        # Get session with write lock
        session = await self.session_manager.get_session(session_id, for_update=True)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Verify state allows clarification
        if session.state != SessionState.RESEARCHING:
            raise ValueError(
                f"Cannot request clarification in state {session.state}"
            )

        # Check clarification limit
        if len(session.clarification_exchanges) >= session.config.max_clarifications:
            logfire.warning(f"Session {session_id} reached clarification limit")
            raise ValueError("Maximum clarification limit reached")

        # Create clarification exchange
        exchange = ClarificationExchange(
            request=request,
            timestamp_requested=datetime.now(timezone.utc).isoformat(),
        )

        # Update session
        session.clarification_exchanges.append(exchange)
        await self.session_manager.transition_state(
            session_id, SessionState.AWAITING_CLARIFICATION
        )
        await self.session_manager.update_session(session)

        # Create future for async response handling
        future = asyncio.Future()
        self._pending_clarifications[f"{session_id}:{len(session.clarification_exchanges)-1}"] = future

        # Start timeout task
        asyncio.create_task(
            self._handle_timeout(
                session_id,
                len(session.clarification_exchanges) - 1,
                session.config.clarification_timeout_seconds,
            )
        )

        logfire.info(f"Clarification requested for session {session_id}")
        return exchange

    @logfire.instrument("Submit clarification response")
    async def submit_response(
        self,
        session_id: str,
        response: ClarificationResponse,
    ) -> bool:
        """
        Phase 2: Process clarification response from user.

        Args:
            session_id: Session identifier
            response: User's clarification response

        Returns:
            True if response was processed successfully
        """
        # Get session with write lock
        session = await self.session_manager.get_session(session_id, for_update=True)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Verify state
        if session.state != SessionState.AWAITING_CLARIFICATION:
            raise ValueError(
                f"Session not awaiting clarification (state: {session.state})"
            )

        # Find pending clarification
        pending_index = None
        for i, exchange in enumerate(session.clarification_exchanges):
            if exchange.response is None and not exchange.timed_out:
                pending_index = i
                break

        if pending_index is None:
            raise ValueError("No pending clarification found")

        # Update exchange
        exchange = session.clarification_exchanges[pending_index]
        exchange.response = response
        exchange.timestamp_responded = datetime.now(timezone.utc).isoformat()

        # Transition state back to researching
        await self.session_manager.transition_state(
            session_id, SessionState.RESEARCHING
        )
        await self.session_manager.update_session(session)

        # Resolve future if exists
        future_key = f"{session_id}:{pending_index}"
        if future_key in self._pending_clarifications:
            future = self._pending_clarifications.pop(future_key)
            if not future.done():
                future.set_result(response)

        logfire.info(f"Clarification response processed for session {session_id}")
        return True

    async def _handle_timeout(
        self,
        session_id: str,
        exchange_index: int,
        timeout_seconds: int,
    ) -> None:
        """Handle clarification timeout."""
        await asyncio.sleep(timeout_seconds)

        future_key = f"{session_id}:{exchange_index}"
        if future_key not in self._pending_clarifications:
            return  # Already responded

        # Get session and mark as timed out
        session = await self.session_manager.get_session(session_id, for_update=True)
        if not session:
            return

        if (
            session.state == SessionState.AWAITING_CLARIFICATION
            and exchange_index < len(session.clarification_exchanges)
        ):
            exchange = session.clarification_exchanges[exchange_index]
            if exchange.response is None:
                exchange.timed_out = True

                # Transition to timeout state
                await self.session_manager.transition_state(
                    session_id, SessionState.CLARIFICATION_TIMEOUT
                )
                await self.session_manager.update_session(session)

                # Resolve future with timeout
                future = self._pending_clarifications.pop(future_key, None)
                if future and not future.done():
                    future.set_exception(TimeoutError("Clarification request timed out"))

                logfire.warning(f"Clarification timeout for session {session_id}")
```

#### Step 2: HTTP Endpoint Integration (`src/api/main.py`)

To expose the new clarification workflow over HTTP:

1. **Initialize shared services** – create the session manager and clarification handler alongside the FastAPI app and start/stop them in the lifespan events.
   ```python
   from api.core.session.manager import SessionManager
   from api.core.session.store import RedisSessionStore, InMemorySessionStore
   from api.core.clarification.handler import ClarificationHandler

   session_store = RedisSessionStore(settings.redis_url) if settings.use_redis else InMemorySessionStore()
   session_manager = SessionManager(store=session_store)
   clarification_handler = ClarificationHandler(session_manager=session_manager)

   @app.on_event("startup")
   async def startup() -> None:
       await session_manager.start()

   @app.on_event("shutdown")
   async def shutdown() -> None:
       await session_manager.stop()
   ```

2. **Add the request endpoint** – called by the workflow when clarification is required. It persists the exchange and returns the questions to the caller.
   ```python
   @app.post("/research/{session_id}/clarification", response_model=ClarificationRequestPayload)
   async def request_clarification(session_id: str, trigger: ClarificationTrigger) -> ClarificationRequestPayload:
       exchange = await clarification_handler.request_clarification(session_id, trigger.request)
       return ClarificationRequestPayload(
           session_id=session_id,
           state=SessionState.AWAITING_CLARIFICATION,
           request=exchange.request,
       )
   ```

3. **Add the response endpoint** – accepts the user’s answers, stores them, and resumes the workflow asynchronously.
   ```python
   @app.post("/research/{session_id}/clarification/response", response_model=ClarificationResumePayload)
   async def submit_clarification(session_id: str, payload: ClarificationResponsePayload) -> ClarificationResumePayload:
       processed = await clarification_handler.submit_response(session_id, payload.response)
       if not processed:
           raise HTTPException(status_code=409, detail="Clarification already fulfilled")

       asyncio.create_task(workflow.resume_research(session_id=session_id))

       session = await session_manager.get_session(session_id)
       return ClarificationResumePayload(session_id=session_id, state=session.state.value)
   ```

4. **Update the workflow** – when Phase 1 detects a clarification requirement, call the request endpoint (or directly invoke `clarification_handler.request_clarification`) and pause the workflow until the response arrives.

## Day 5: Error Handling Architecture

### Implementation Steps

#### Step 1: Exception Hierarchy (`src/core/exceptions.py`)

```python
"""Custom exception hierarchy for robust error handling."""

from typing import Any, Dict, Optional


class OpenDeepResearchError(Exception):
    """Base exception for all application errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class SessionError(OpenDeepResearchError):
    """Base exception for session-related errors."""
    pass


class SessionNotFoundError(SessionError):
    """Raised when a session cannot be found."""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session {session_id} not found",
            error_code="SESSION_NOT_FOUND",
            details={"session_id": session_id},
        )


class SessionExpiredError(SessionError):
    """Raised when attempting to use an expired session."""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session {session_id} has expired",
            error_code="SESSION_EXPIRED",
            details={"session_id": session_id},
        )


class ClarificationError(OpenDeepResearchError):
    """Base exception for clarification-related errors."""
    pass


class ClarificationLimitError(ClarificationError):
    """Raised when clarification limit is exceeded."""

    def __init__(self, session_id: str, limit: int):
        super().__init__(
            message=f"Session {session_id} exceeded clarification limit of {limit}",
            error_code="CLARIFICATION_LIMIT_EXCEEDED",
            details={"session_id": session_id, "limit": limit},
        )


class ResearchError(OpenDeepResearchError):
    """Base exception for research-related errors."""
    pass


class ExternalServiceError(OpenDeepResearchError):
    """Raised when external service (Redis, AI model) fails."""

    def __init__(self, service: str, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message=f"{service} error: {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            details={
                "service": service,
                "original_error": str(original_error) if original_error else None,
            },
        )


class RateLimitError(OpenDeepResearchError):
    """Raised when rate limit is exceeded."""

    def __init__(self, limit: int, window_seconds: int):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            error_code="RATE_LIMIT_EXCEEDED",
            details={"limit": limit, "window_seconds": window_seconds},
        )
```

#### Step 2: Error Middleware & Structured Responses (`src/api/error_handlers.py`)

Create centralized exception handlers so each failure yields the same JSON structure:

```python
"""Application-wide exception handlers."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.exceptions import OpenDeepResearchError


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(OpenDeepResearchError)
    async def known_error_handler(request: Request, exc: OpenDeepResearchError) -> JSONResponse:  # noqa: ANN001
        return JSONResponse(
            status_code=exc.details.get("status_code", 400),
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "path": request.url.path,
            },
        )

    @app.exception_handler(Exception)
    async def fallback_error_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: ANN001
        logfire.exception("Unhandled error", request=str(request.url))
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "Unexpected server error",
                "details": {},
                "path": request.url.path,
            },
        )
```

Call `install_error_handlers(app)` immediately after constructing the FastAPI `app` instance in `src/api/main.py`.

#### Step 3: Circuit Breaker (`src/core/resilience/circuit_breaker.py`)

```python
"""Circuit breaker implementation for external service protection."""

import asyncio
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar

import logfire

from core.exceptions import ExternalServiceError

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Service is failing, calls are rejected
    - HALF_OPEN: Testing if service has recovered
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type[Exception] = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Service name for logging
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying recovery
            expected_exception: Exception type to catch
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
        self._lock = asyncio.Lock()

    @logfire.instrument("Circuit breaker call")
    async def call(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            ExternalServiceError: If circuit is open
        """
        async with self._lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logfire.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    raise ExternalServiceError(
                        service=self.name,
                        message="Circuit breaker is OPEN",
                    )

        try:
            # Execute function
            result = await func(*args, **kwargs)

            # Record success
            await self._on_success()
            return result

        except self.expected_exception as e:
            # Record failure
            await self._on_failure()
            raise ExternalServiceError(
                service=self.name,
                message=f"Service call failed: {str(e)}",
                original_error=e,
            )

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self.failure_count = 0

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= 3:  # Require 3 successes to close
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
                    logfire.info(f"Circuit breaker {self.name} closed after recovery")

    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logfire.warning(f"Circuit breaker {self.name} reopened after test failure")
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logfire.warning(
                    f"Circuit breaker {self.name} opened after {self.failure_count} failures"
                )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        time_since_failure = datetime.now(timezone.utc) - self.last_failure_time
        return time_since_failure > timedelta(seconds=self.recovery_timeout)
```

## Testing Strategy

### Concurrency Tests (`tests/test_concurrency.py`)

```python
"""Concurrency tests for session management."""

import asyncio
import random
from typing import List

import pytest

from api.core.session.manager import SessionManager
from api.core.session.models import SessionConfig, SessionState
from api.core.session.store import InMemorySessionStore


@pytest.fixture
async def session_manager():
    """Create session manager for testing."""
    store = InMemorySessionStore()
    manager = SessionManager(store=store)
    await manager.start()
    yield manager
    await manager.stop()


@pytest.mark.asyncio
async def test_concurrent_session_creation(session_manager: SessionManager):
    """Test creating multiple sessions concurrently."""
    num_sessions = 50

    async def create_session(index: int):
        session = await session_manager.create_session(
            query=f"Test query {index}",
            config=SessionConfig(ttl_seconds=300),
        )
        return session.id

    # Create sessions concurrently
    tasks = [create_session(i) for i in range(num_sessions)]
    session_ids = await asyncio.gather(*tasks)

    # Verify all sessions created
    assert len(session_ids) == num_sessions
    assert len(set(session_ids)) == num_sessions  # All unique


@pytest.mark.asyncio
async def test_concurrent_read_write(session_manager: SessionManager):
    """Test concurrent reads and writes to same session."""
    # Create a session
    session = await session_manager.create_session(
        query="Test query",
        config=SessionConfig(ttl_seconds=300),
    )
    session_id = str(session.id)

    read_count = 0
    write_count = 0

    async def reader():
        nonlocal read_count
        for _ in range(10):
            session = await session_manager.get_session(session_id, for_update=False)
            assert session is not None
            read_count += 1
            await asyncio.sleep(random.uniform(0.001, 0.01))

    async def writer():
        nonlocal write_count
        for _ in range(5):
            session = await session_manager.get_session(session_id, for_update=True)
            assert session is not None
            write_count += 1
            await asyncio.sleep(random.uniform(0.001, 0.01))

    # Run readers and writers concurrently
    tasks = [reader() for _ in range(5)] + [writer() for _ in range(3)]
    await asyncio.gather(*tasks)

    assert read_count == 50
    assert write_count == 15
```

## Acceptance Criteria Checklist

### Day 3: Enhanced Session Management
- [ ] Read-write locks protecting session access
- [ ] TTL-based cleanup running in background
- [ ] Session state machine with valid transitions
- [ ] Concurrent access tests passing

### Day 4: Two-Phase Clarification Flow
- [ ] Clarification request endpoint functional (`POST /research/{session_id}/clarification`)
- [ ] Clarification response endpoint functional (`POST /research/{session_id}/clarification/response`)
- [ ] Timeout handling implemented
- [ ] Clarification limit enforcement
- [ ] Workflow pause/resume validated via HTTP integration tests

### Day 5: Error Handling & Recovery
- [ ] Complete exception hierarchy defined
- [ ] Circuit breakers protecting external services
- [ ] Error middleware returning structured responses
- [ ] Retry logic with exponential backoff (Tenacity)
- [ ] All errors logged with session/user context
- [ ] Unhandled exceptions monitored during soak tests

## Conclusion

Phase 2 transforms the basic HTTP server into a production-ready system with robust session management, two-phase clarification flow, and comprehensive error handling. The implementation provides a solid foundation for Phase 3's advanced features while ensuring system reliability and maintainability.
