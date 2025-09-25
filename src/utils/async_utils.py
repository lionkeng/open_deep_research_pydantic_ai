"""Utilities for safely invoking async code from sync contexts.

Avoid creating coroutines in a context where they won't be awaited. Use a
factory that only constructs the coroutine when we know we can run it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any


def run_awaitable_if_no_loop(factory: Callable[[], Awaitable[Any]]) -> Any | None:
    """Run an awaitable if not already inside a running event loop.

    Returns None when a loop is already running so the caller can gracefully
    fall back to a sync-safe path.
    """
    try:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            return None
        return asyncio.run(factory())  # type: ignore[arg-type]
    except Exception:
        return None
