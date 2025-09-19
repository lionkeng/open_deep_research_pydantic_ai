"""Source repository abstractions for managing research sources."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from hashlib import sha256

import logfire
from pydantic import BaseModel, Field

from models.research_executor import ResearchSource, SourceUsage


class SourceIdentity(BaseModel):
    """Identifier and canonical key for a registered source."""

    source_id: str
    canonical_key: str
    version: int = Field(default=1, ge=1)


class AbstractSourceRepository(ABC):
    """Protocol-like base class for source repositories."""

    @abstractmethod
    async def register(self, source: ResearchSource) -> SourceIdentity:
        raise NotImplementedError

    @abstractmethod
    async def get(self, source_id: str) -> ResearchSource | None:
        raise NotImplementedError

    @abstractmethod
    async def find_by_key(self, canonical_key: str) -> SourceIdentity | None:
        raise NotImplementedError

    @abstractmethod
    def iter_all(self) -> AsyncIterator[tuple[SourceIdentity, ResearchSource]]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def ordered_sources(self) -> list[ResearchSource]:
        """Return sources ordered by their numeric identifier."""
        raise NotImplementedError

    @abstractmethod
    async def register_usage(
        self,
        source_id: str,
        *,
        finding_id: str | None = None,
        cluster_id: str | None = None,
        contradiction_id: str | None = None,
        pattern_id: str | None = None,
        report_section: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_usage(self, source_id: str) -> SourceUsage | None:
        raise NotImplementedError


@dataclass
class InMemorySourceRepository(AbstractSourceRepository):
    """Simple in-memory source repository suitable for unit tests and phase 1."""

    _sources: list[ResearchSource] = field(default_factory=list)
    _key_index: dict[str, SourceIdentity] = field(default_factory=dict)
    _usage: dict[str, SourceUsage] = field(default_factory=dict)
    _register_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def register(self, source: ResearchSource) -> SourceIdentity:
        async with self._register_lock:
            canonical_key = self._build_canonical_key(source)
            existing_identity = self._key_index.get(canonical_key)
            if existing_identity:
                existing_source = await self.get(existing_identity.source_id)
                if existing_source:
                    self._merge_source(existing_source, source)
                return existing_identity

            source_id = f"S{len(self._sources) + 1}"
            source_with_id = source.model_copy(
                update={"source_id": source_id, "canonical_key": canonical_key}
            )
            self._sources.append(source_with_id)
            identity = SourceIdentity(source_id=source_id, canonical_key=canonical_key)
            self._key_index[canonical_key] = identity
            logfire.debug("Registered new source", source_id=source_id, url=source_with_id.url)
            return identity

    async def get(self, source_id: str) -> ResearchSource | None:
        index = self._index_from_id(source_id)
        if index is None:
            return None
        if 0 <= index < len(self._sources):
            return self._sources[index]
        return None

    async def find_by_key(self, canonical_key: str) -> SourceIdentity | None:
        return self._key_index.get(canonical_key)

    def iter_all(self) -> AsyncIterator[tuple[SourceIdentity, ResearchSource]]:
        async def _iter():
            for source in self._sources:
                if source.source_id is None or source.canonical_key is None:
                    continue
                identity = SourceIdentity(
                    source_id=source.source_id, canonical_key=source.canonical_key
                )
                yield identity, source

        return _iter()

    async def ordered_sources(self) -> list[ResearchSource]:
        return sorted(self._sources, key=self._source_sort_key)

    async def register_usage(
        self,
        source_id: str,
        *,
        finding_id: str | None = None,
        cluster_id: str | None = None,
        contradiction_id: str | None = None,
        pattern_id: str | None = None,
        report_section: str | None = None,
    ) -> None:
        usage = self._usage.setdefault(source_id, SourceUsage(source_id=source_id))
        if finding_id:
            usage.record_finding(finding_id)
        if cluster_id:
            usage.record_cluster(cluster_id)
        if contradiction_id:
            usage.record_contradiction(contradiction_id)
        if pattern_id:
            usage.record_pattern(pattern_id)
        if report_section:
            usage.record_report_section(report_section)

    async def get_usage(self, source_id: str) -> SourceUsage | None:
        return self._usage.get(source_id)

    def _build_canonical_key(self, source: ResearchSource) -> str:
        if source.url:
            return source.url.strip().lower()
        payload = f"{source.title}|{source.metadata.get('snippet', '')}"
        return sha256(payload.encode("utf-8")).hexdigest()

    def _index_from_id(self, source_id: str) -> int | None:
        try:
            return int(source_id.removeprefix("S")) - 1
        except (ValueError, AttributeError):
            return None

    def _merge_source(self, existing: ResearchSource, incoming: ResearchSource) -> None:
        """Merge richer metadata from the incoming source into the stored one."""
        updates: dict[str, object] = {}
        for field_name in ("author", "publisher", "date", "source_type"):
            new_value = getattr(incoming, field_name, None)
            if new_value and not getattr(existing, field_name, None):
                updates[field_name] = new_value
        if incoming.metadata:
            merged_metadata = existing.metadata.copy()
            merged_metadata.update(incoming.metadata)
            updates["metadata"] = merged_metadata
        if updates:
            index = self._index_from_id(existing.source_id or "")
            if index is not None and 0 <= index < len(self._sources):
                self._sources[index] = existing.model_copy(update=updates)

    def _source_sort_key(self, source: ResearchSource) -> tuple[int, str]:
        index = self._index_from_id(source.source_id or "")
        return (index if index is not None else 10**6, source.title)


async def ensure_repository(
    repository: AbstractSourceRepository | None,
) -> AbstractSourceRepository:
    """Ensure that we always have an in-memory repository available."""
    if repository is not None:
        return repository
    return InMemorySourceRepository()


def summarize_sources_for_prompt(sources: Iterable[ResearchSource]) -> str:
    """Create a short markdown bullet list of sources for prompt injection."""
    lines: list[str] = []
    for source in sources:
        if not source.source_id:
            continue
        descriptor = source.title
        if source.metadata.get("publisher"):
            descriptor += f" — {source.metadata['publisher']}"
        if source.date:
            descriptor += f" ({source.date.strftime('%Y-%m-%d')})"
        url = f" <{source.url}>" if source.url else ""
        lines.append(f"- {source.source_id}: {descriptor}{url}")
    return "\n".join(lines)
