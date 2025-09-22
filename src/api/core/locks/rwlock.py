"""Async read-write lock implementation used by session management."""

from __future__ import annotations

import asyncio

import logfire


class AsyncReadWriteLock:
    """Asynchronous read/write lock.

    Multiple readers may hold the lock concurrently. Writers obtain exclusive
    access and are given priority so they are not starved by readers.
    """

    def __init__(self, name: str | None = None):
        self._name = name or "rwlock"
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writers = 0
        self._read_waiters = 0
        self._write_waiters = 0

    async def acquire_read(self) -> None:
        """Acquire the lock for reading."""
        async with self._condition:
            self._read_waiters += 1
            try:
                while self._writers > 0 or self._write_waiters > 0:
                    await self._condition.wait()
                self._readers += 1
                logfire.trace(
                    "Read lock acquired",
                    lock=self._name,
                    readers=self._readers,
                )
            finally:
                self._read_waiters -= 1

    async def release_read(self) -> None:
        """Release the read lock."""
        async with self._condition:
            if self._readers <= 0:
                raise RuntimeError("release_read() called without matching acquire")
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()
            logfire.trace(
                "Read lock released",
                lock=self._name,
                readers=self._readers,
            )

    async def acquire_write(self) -> None:
        """Acquire the lock for writing."""
        async with self._condition:
            self._write_waiters += 1
            try:
                while self._readers > 0 or self._writers > 0:
                    _ = await self._condition.wait()
                self._writers += 1
                logfire.debug("Write lock acquired", lock=self._name)
            finally:
                self._write_waiters -= 1

    async def release_write(self) -> None:
        """Release the write lock."""
        async with self._condition:
            if self._writers != 1:
                raise RuntimeError("release_write() called without matching acquire")
            self._writers -= 1
            self._condition.notify_all()
            logfire.debug("Write lock released", lock=self._name)

    def read_locked(self) -> _ReadLockContext:
        """Return context manager for read locking."""
        return _ReadLockContext(self)

    def write_locked(self) -> _WriteLockContext:
        """Return context manager for write locking."""
        return _WriteLockContext(self)


class _ReadLockContext:
    def __init__(self, lock: AsyncReadWriteLock):
        self._lock = lock

    async def __aenter__(self):
        await self._lock.acquire_read()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._lock.release_read()


class _WriteLockContext:
    def __init__(self, lock: AsyncReadWriteLock):
        self._lock = lock

    async def __aenter__(self):
        await self._lock.acquire_write()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._lock.release_write()
