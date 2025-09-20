"""Tenacity-powered retry helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

import logfire
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.exceptions import ExternalServiceError, OpenDeepResearchError

P = ParamSpec("P")
T = TypeVar("T")


def _log_retry(retry_state: RetryCallState) -> None:
    attempt = retry_state.attempt_number
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logfire.warning(
        "Retrying operation",
        attempt=attempt,
        error=str(exception) if exception else None,
    )


async def retry_async(
    func: Callable[P, Awaitable[T]],
    *args: P.args,
    attempts: int = 3,
    base_delay: float = 0.2,
    **kwargs: P.kwargs,
) -> T:
    """Execute an async callable with exponential-backoff retries."""

    retry_policy = AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=base_delay, min=base_delay, max=2.0),
        retry=retry_if_exception_type(
            (OpenDeepResearchError, ExternalServiceError, ConnectionError, TimeoutError)
        ),
        before_sleep=_log_retry,
        reraise=True,
    )

    async for attempt in retry_policy:
        with attempt:
            return await func(*args, **kwargs)

    raise AssertionError("retry_async exhausted without result")  # pragma: no cover
