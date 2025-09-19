"""Unit tests for the in-memory source repository."""

import asyncio

import pytest

from models.research_executor import ResearchSource
from services.source_repository import InMemorySourceRepository


@pytest.mark.asyncio
async def test_register_and_retrieve_source() -> None:
    repository = InMemorySourceRepository()
    source = ResearchSource(title="Example", url="https://example.com")

    identity = await repository.register(source)
    assert identity.source_id == "S1"

    stored = await repository.get(identity.source_id)
    assert stored is not None
    assert stored.source_id == "S1"
    assert stored.url == "https://example.com"


@pytest.mark.asyncio
async def test_deduplicate_on_url() -> None:
    repository = InMemorySourceRepository()
    first = ResearchSource(title="Example", url="https://example.com")
    second = ResearchSource(title="Updated", url="https://example.com", publisher="Daily News")

    identity1 = await repository.register(first)
    identity2 = await repository.register(second)

    assert identity1.source_id == identity2.source_id
    stored = await repository.get(identity1.source_id)
    assert stored is not None
    assert stored.publisher == "Daily News"


@pytest.mark.asyncio
async def test_usage_tracking() -> None:
    repository = InMemorySourceRepository()
    source = ResearchSource(title="Example", url="https://example.com")
    identity = await repository.register(source)

    await repository.register_usage(identity.source_id, finding_id="finding-1")
    await repository.register_usage(identity.source_id, report_section="final_report")

    usage = await repository.get_usage(identity.source_id)
    assert usage is not None
    assert usage.finding_ids == ["finding-1"]
    assert usage.report_sections == ["final_report"]



@pytest.mark.asyncio
async def test_concurrent_registration_unique_ids() -> None:
    repository = InMemorySourceRepository()
    source = ResearchSource(title="Concurrent", url="https://concurrent.test")

    async def register_copy() -> str:
        identity = await repository.register(source)
        return identity.source_id

    ids = await asyncio.gather(*(register_copy() for _ in range(5)))
    assert len(set(ids)) == 1
    stored = await repository.get(ids[0])
    assert stored is not None


@pytest.mark.asyncio
async def test_deduplication_without_url() -> None:
    repository = InMemorySourceRepository()
    first = ResearchSource(title="No URL", url=None, metadata={"snippet": "sample"})
    second = ResearchSource(title="No URL", url=None, metadata={"snippet": "sample"})

    identity1 = await repository.register(first)
    identity2 = await repository.register(second)

    assert identity1.source_id == identity2.source_id
    stored = await repository.get(identity1.source_id)
    assert stored is not None


@pytest.mark.asyncio
async def test_pattern_usage_tracking() -> None:
    repository = InMemorySourceRepository()
    identity = await repository.register(ResearchSource(title="Pattern", url="https://pattern.test"))

    await repository.register_usage(identity.source_id, pattern_id="pattern-1")
    usage = await repository.get_usage(identity.source_id)
    assert usage is not None
    assert usage.pattern_ids == ["pattern-1"]
