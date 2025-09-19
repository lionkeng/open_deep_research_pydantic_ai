"""Tests for the source validation pipeline."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from models.research_executor import ResearchSource
from services.source_repository import InMemorySourceRepository
from services.source_validation import SourceValidationPipeline


@pytest.mark.asyncio
async def test_validate_and_register_success(monkeypatch) -> None:
    repository = InMemorySourceRepository()
    http_client = AsyncMock(spec=httpx.AsyncClient)
    http_client.head.return_value = httpx.Response(status_code=200, request=httpx.Request("HEAD", "https://valid.test"))

    pipeline = SourceValidationPipeline(repository=repository, http_client=http_client)

    registered = await pipeline.validate_and_register({
        "title": "Valid",
        "url": "https://valid.test",
        "source_type": "article",
    })

    assert isinstance(registered, ResearchSource)
    assert registered.source_id == "S1"
    http_client.head.assert_awaited()


@pytest.mark.asyncio
async def test_validate_and_register_degraded(monkeypatch) -> None:
    repository = InMemorySourceRepository()
    http_client = AsyncMock(spec=httpx.AsyncClient)
    http_client.head.side_effect = httpx.HTTPError("boom")

    pipeline = SourceValidationPipeline(repository=repository, http_client=http_client)

    registered = await pipeline.validate_and_register({
        "title": "Broken",
        "url": "https://broken.test",
        "source_type": "article",
    })

    assert registered.metadata["validation"]["state"] == "degraded"
    assert registered.source_id == "S1"
