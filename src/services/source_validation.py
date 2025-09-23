"""Validation pipeline for registering research sources."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
import logfire

from core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from models.research_executor import ResearchSource
from services.source_repository import AbstractSourceRepository, SourceIdentity

LOGGER = logfire
_URL_REGEX = re.compile(r"^https?://", re.IGNORECASE)


@dataclass
class SourceValidationPipeline:
    """Validate sources before persisting them into the repository."""

    repository: AbstractSourceRepository
    http_client: httpx.AsyncClient
    circuit_breaker: CircuitBreaker[str] | None = None
    request_timeout: float = 5.0

    async def validate_and_register(self, raw_source: dict[str, Any]) -> ResearchSource:
        """Validate the source metadata and register it, handling degraded fallbacks."""

        try:
            validated_payload = await self._validate_payload(raw_source)
            identity = await self.repository.register(ResearchSource(**validated_payload))
            stored = await self.repository.get(identity.source_id)
            return stored if stored is not None else ResearchSource(**validated_payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.trace(
                "Source validation failed; registering degraded source",
                error=str(exc),
                source_title=raw_source.get("title"),
            )
            degraded = await self._register_degraded_source(raw_source, exc)
            return degraded

    async def _validate_payload(self, raw_source: dict[str, Any]) -> dict[str, Any]:
        payload = dict(raw_source)
        url = payload.get("url")
        if url:
            validated_url = self._normalise_url(url)
            await self._verify_remote_resource(validated_url)
            payload["url"] = validated_url
        return payload

    def _normalise_url(self, url: str) -> str:
        if not _URL_REGEX.match(url):
            raise ValueError(f"Invalid URL scheme for source: {url}")
        parsed = urlparse(url)
        cleaned = parsed.geturl()
        if not parsed.netloc:
            raise ValueError(f"URL missing host: {url}")
        return cleaned

    async def _verify_remote_resource(self, url: str) -> None:
        if not self.circuit_breaker:
            await self._head_request(url)
            return

        async with self.circuit_breaker.protect("source_validation"):
            await self._head_request(url)

    async def _head_request(self, url: str) -> None:
        try:
            response = await self.http_client.head(url, timeout=self.request_timeout)
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "unhealthy source", request=response.request, response=response
                )
        except httpx.HTTPError as exc:
            raise ValueError(f"Failed to validate source url {url}: {exc}") from exc

    async def _register_degraded_source(
        self, raw_source: dict[str, Any], error: Exception
    ) -> ResearchSource:
        degraded_metadata = dict(raw_source)
        metadata = degraded_metadata.setdefault("metadata", {})
        validation_info = metadata.setdefault("validation", {})
        validation_info.update(
            {
                "state": "degraded",
                "error": str(error),
            }
        )
        identity: SourceIdentity = await self.repository.register(
            ResearchSource(**degraded_metadata)
        )
        stored = await self.repository.get(identity.source_id)
        return stored if stored is not None else ResearchSource(**degraded_metadata)


def create_default_validation_pipeline(
    repository: AbstractSourceRepository,
    http_client: httpx.AsyncClient,
) -> SourceValidationPipeline:
    """Factory that builds a pipeline with a sensible circuit breaker."""

    circuit = CircuitBreaker[str](
        config=CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout_seconds=10.0,
            half_open_max_attempts=1,
            name="source_validation",
        )
    )
    return SourceValidationPipeline(
        repository=repository, http_client=http_client, circuit_breaker=circuit
    )
