"""Cache management service for research executor."""

import hashlib
import json
import sys
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

import logfire
from pydantic import BaseModel

from src.models.research_executor import CacheMetadata, OptimizationConfig

T = TypeVar("T")


class CacheManager:
    """Manages caching for the research executor system."""

    def __init__(self, config: OptimizationConfig):
        """Initialize the cache manager.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.cache: OrderedDict[str, tuple[Any, CacheMetadata]] = OrderedDict()
        self.total_size_bytes = 0
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0,
        }
        self.logger = logfire

    def _generate_key(self, cache_type: str, content: Any) -> str:
        """Generate a cache key based on content hash.

        Args:
            cache_type: Type of cache (synthesis, vectorization, etc.)
            content: Content to cache

        Returns:
            Cache key string
        """
        # Convert content to string for hashing
        if isinstance(content, BaseModel):
            content_str = content.model_dump_json()
        elif isinstance(content, list | dict):
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)

        content_hash = hashlib.sha256(content_str.encode()).hexdigest()
        return f"{cache_type}:{content_hash[:16]}"

    def _calculate_size(self, obj: Any) -> int:
        """Calculate approximate size of an object in bytes.

        Args:
            obj: Object to calculate size for

        Returns:
            Size in bytes
        """
        if isinstance(obj, BaseModel):
            try:
                json_str = obj.model_dump_json()
                if len(json_str) > 1024 * 1024:  # 1MB limit
                    return sys.getsizeof(obj)
                return len(json_str.encode())
            except (MemoryError, RecursionError):
                return sys.getsizeof(obj)
        elif isinstance(obj, str):
            return len(obj.encode())
        elif isinstance(obj, list | dict):
            try:
                # Limit serialization size for safety
                json_str = json.dumps(obj)
                if len(json_str) > 1024 * 1024:  # 1MB limit
                    return sys.getsizeof(obj)  # Fallback to rough estimate
                return len(json_str.encode())
            except (MemoryError, RecursionError):
                return sys.getsizeof(obj)
        else:
            return sys.getsizeof(obj)

    def _enforce_size_limit(self) -> None:
        """Enforce the maximum cache size limit."""
        max_size_bytes = self.config.max_cache_size_mb * 1024 * 1024

        while self.total_size_bytes > max_size_bytes and self.cache:
            # Remove least recently used item
            key, (_, metadata) = self.cache.popitem(last=False)
            self.total_size_bytes -= metadata.size_bytes
            self.metrics["evictions"] += 1
            self.logger.debug(f"Evicted cache entry: {key}")

    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        expired_keys = []
        now = datetime.now(UTC)

        for key, (_, metadata) in self.cache.items():
            if metadata.expires_at < now:
                expired_keys.append(key)

        for key in expired_keys:
            _, metadata = self.cache.pop(key)
            self.total_size_bytes -= metadata.size_bytes
            self.logger.debug(f"Removed expired cache entry: {key}")

    def get(self, cache_type: str, content_key: Any) -> Any | None:
        """Get an item from cache.

        Args:
            cache_type: Type of cache
            content_key: Content to generate key from

        Returns:
            Cached value or None if not found
        """
        if not self.config.enable_caching:
            return None

        self.metrics["total_requests"] += 1
        key = self._generate_key(cache_type, content_key)

        if key in self.cache:
            value, metadata = self.cache[key]

            # Check if expired
            if metadata.is_expired:
                self.cache.pop(key)
                self.total_size_bytes -= metadata.size_bytes
                self.metrics["misses"] += 1
                return None

            # Update access metadata
            metadata.access_count += 1
            metadata.last_accessed = datetime.now(UTC)

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            self.metrics["hits"] += 1
            self.logger.debug(f"Cache hit for {cache_type}: {key}")
            return value

        self.metrics["misses"] += 1
        return None

    def set(
        self, cache_type: str, content_key: Any, value: Any, ttl_override: int | None = None
    ) -> str:
        """Store an item in cache.

        Args:
            cache_type: Type of cache
            content_key: Content to generate key from
            value: Value to cache
            ttl_override: Optional TTL override in seconds

        Returns:
            Cache key
        """
        if not self.config.enable_caching:
            return ""

        # Cleanup expired entries periodically
        if self.metrics["total_requests"] % 100 == 0:
            self._cleanup_expired()

        key = self._generate_key(cache_type, content_key)
        size_bytes = self._calculate_size(value)
        ttl = ttl_override or self.config.cache_ttl_seconds

        metadata = CacheMetadata(
            key=key,
            content_hash=hashlib.sha256(str(content_key).encode()).hexdigest(),
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl),
            size_bytes=size_bytes,
            cache_type=cache_type,
        )

        # Remove old entry if exists
        if key in self.cache:
            _, old_metadata = self.cache.pop(key)
            self.total_size_bytes -= old_metadata.size_bytes

        self.cache[key] = (value, metadata)
        self.total_size_bytes += size_bytes

        # Enforce size limit
        self._enforce_size_limit()

        self.logger.debug(f"Cached {cache_type}: {key} (size: {size_bytes} bytes)")
        return key

    def invalidate(self, cache_type: str | None = None, pattern: str | None = None) -> int:
        """Invalidate cache entries.

        Args:
            cache_type: Optional cache type to invalidate
            pattern: Optional pattern to match keys

        Returns:
            Number of entries invalidated
        """
        invalidated = 0
        keys_to_remove = []

        for key, (_, metadata) in self.cache.items():
            should_remove = False

            if (
                (cache_type and metadata.cache_type == cache_type)
                or (pattern and pattern in key)
                or (not cache_type and not pattern)
            ):
                should_remove = True

            if should_remove:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            _, metadata = self.cache.pop(key)
            self.total_size_bytes -= metadata.size_bytes
            invalidated += 1

        self.logger.info(f"Invalidated {invalidated} cache entries")
        return invalidated

    def get_metrics(self) -> dict[str, Any]:
        """Get cache metrics.

        Returns:
            Dictionary of cache metrics
        """
        hit_rate = self.metrics["hits"] / max(self.metrics["total_requests"], 1)

        return {
            "hits": self.metrics["hits"],
            "misses": self.metrics["misses"],
            "evictions": self.metrics["evictions"],
            "hit_rate": hit_rate,
            "total_requests": self.metrics["total_requests"],
            "current_size_mb": self.total_size_bytes / (1024 * 1024),
            "max_size_mb": self.config.max_cache_size_mb,
            "num_entries": len(self.cache),
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.total_size_bytes = 0
        self.logger.info("Cache cleared")
