# Implementation Plan: Pattern Analysis Cache Remediation

## Executive Summary

Pattern analysis fails when results are read back from cache because the metrics collector assumes cached entries are dictionaries while the cache returns `PatternAnalysis` models. In parallel, cache serialization uses naive `str()`/`json.dumps` conversions that do not understand nested Pydantic models, producing unstable cache keys and occasional serialization errors. This plan fixes both issues within the existing synchronous cache design—no async wrappers or API changes required—while giving developers precise implementation steps.

## Problem Statement

- **Observed exception**: `AttributeError: 'PatternAnalysis' object has no attribute 'get'`
- **Origin**: `src/services/metrics_collector.py:135-149` when `record_pattern_strength` iterates over cached items returned by `CacheManager.get`.
- **Contributing factors**:
  - `record_pattern_strength` treats every list item as a mapping; cached values are `PatternAnalysis` instances produced in `PatternRecognizer.detect_patterns`.
  - `CacheManager._calculate_size` / `_generate_key` rely on `json.dumps` and `str(obj)`, which break for nested Pydantic models and yield inconsistent hashes.
  - `assess_synthesis_quality` awaits `deps.metrics_collector.record_synthesis_metrics`, but `MetricsCollector` lacks that coroutine.

## Goals

- Support both dict inputs and `PatternAnalysis` models inside `MetricsCollector.record_pattern_strength`.
- Provide an async-friendly `record_synthesis_metrics` implementation that matches current call sites.
- Ensure cache key generation and sizing are deterministic for Pydantic models and collections.
- Keep the cache API synchronous to avoid regressions.
- Cover changes with focused unit/integration tests.

## Remediation Overview

| Workstream | Focus | Owners |
|------------|-------|--------|
| A | Metrics collector robustness | Services team |
| B | Cache serialization + key determinism | Platform team |
| C | Tests & migration | QA / DevEx |

---

## Workstream A – Metrics Collector Improvements (HIGH)

**Target file**: `src/services/metrics_collector.py`

1. **Normalize mixed pattern payloads**
   - Add helper:
     ```python
     def _normalize_patterns(
         patterns: Sequence[PatternAnalysis | Mapping[str, Any]]
     ) -> list[dict[str, Any]]:
         normalized: list[dict[str, Any]] = []
         for item in patterns:
             if isinstance(item, PatternAnalysis):
                 normalized.append(item.model_dump(mode="json", exclude_none=True))
             elif isinstance(item, Mapping):
                 normalized.append(dict(item))
             else:
                 raise TypeError(
                     f"Unsupported pattern payload: {type(item)!r}"
                 )
         return normalized
     ```
   - Update `record_pattern_strength` to call `_normalize_patterns` before computing mean/max counts. Keep method synchronous—the agent already runs it inside an async function without awaiting.

2. **Implement `record_synthesis_metrics` coroutine**
   - Add to `MetricsCollector`:
     ```python
     async def record_synthesis_metrics(
         self, metrics: Mapping[str, Any]
     ) -> None:
         if not self.config.enable_metrics_collection:
             return
         if self.current_snapshot is None:
             self.start_collection()
         self.current_snapshot.quality_metrics.update(dict(metrics))
     ```
   - Do **not** introduce `_record_metrics`; rely on existing state (`current_snapshot`).
   - Update tests that patch `record_synthesis_metrics` to exercise the real implementation.

3. **Optional ergonomic helper**
   - If desired, expose a synchronous `record_synthesis_metrics_sync` that simply wraps the coroutine with `asyncio.run` for non-async contexts. Keep out of hot paths.

4. **Tests**
   - New unit cases in `tests/unit/services/test_metrics_collector.py`:
     - Mixed inputs (`PatternAnalysis` + dict) produce normalized output.
     - `record_synthesis_metrics` populates `current_snapshot.quality_metrics`.
     - Calling `record_synthesis_metrics` with collection disabled is a no-op.

---

## Workstream B – Cache Serialization & Key Determinism (HIGH)

**Target files**: `src/services/cache_manager.py`, `src/agents/research_executor_tools.py`, `src/utils/cache_serialization.py` *(new)*

1. **Create shared serialization helper**
   - New module `src/utils/cache_serialization.py`:
     ```python
     import json
     from collections.abc import Mapping, Sequence
     from typing import Any

     from pydantic import BaseModel

     def normalize_for_cache(value: Any) -> Any:
         if isinstance(value, BaseModel):
             return value.model_dump(mode="json", exclude_none=True)
         if isinstance(value, Mapping):
             return {
                 str(key): normalize_for_cache(sub_value)
                 for key, sub_value in sorted(value.items(), key=lambda item: str(item[0]))
             }
         if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
             return [normalize_for_cache(item) for item in value]
         return value

     def dumps_for_cache(value: Any) -> str:
         normalized = normalize_for_cache(value)
         return json.dumps(normalized, sort_keys=True, separators=(",", ":"))
     ```
     - Import `json` within the module.

2. **Use helper inside `CacheManager`**
   - Inject `normalize_for_cache` / `dumps_for_cache` into `_generate_key` and `_calculate_size`:
     ```python
     from utils.cache_serialization import normalize_for_cache, dumps_for_cache

     def _generate_key(self, cache_type: str, content: Any) -> str:
         serialized = dumps_for_cache({"cache_type": cache_type, "content": content})
         content_hash = hashlib.sha256(serialized.encode()).hexdigest()
         return f"{cache_type}:{content_hash[:16]}"

     def _calculate_size(self, obj: Any) -> int:
         try:
             serialized = dumps_for_cache(obj)
         except (TypeError, ValueError):
             return sys.getsizeof(obj)
         return len(serialized.encode())
     ```
   - This keeps API unchanged and handles nested models safely.

3. **Update `_generate_cache_key` utility**
   - In `src/agents/research_executor_tools.py`, replace the current `str(args)` approach with:
     ```python
     from utils.cache_serialization import dumps_for_cache

     def _generate_cache_key(*args: Any) -> str:
         serialized = dumps_for_cache(args)
         return hashlib.md5(serialized.encode()).hexdigest()[:16]
     ```
   - Callers continue to pass findings/clusters lists—no signature changes required.

4. **Keep cache interactions synchronous**
   - Ensure documentation and code samples use `deps.cache_manager.get(...)` / `set(...)` without `await`. The cache implementation is synchronous; awaiting recreates earlier TypeErrors.

5. **Tests**
   - Add `tests/unit/services/test_cache_manager.py` cases:
     - `_generate_key` yields stable hashes for lists of `PatternAnalysis` models.
     - `_calculate_size` handles large nested structures without raising.
   - Add snapshot-based test for `_generate_cache_key` stability with mixed argument types.

---

## Workstream C – Verification, Documentation & Rollout (MEDIUM)

1. **Integration coverage**
   - Extend `tests/integration/test_research_executor_integration.py` to:
     - Execute `analyze_patterns` twice—first run populates cache, second run validates cache hit returns `PatternAnalysis` models without errors.
     - Assert `metrics_collector.current_snapshot.quality_metrics` includes pattern strength after cached execution.

2. **Update documentation**
   - `docs/cache-system-bug-fixes.md`: replace async mismatch narrative with the actual root cause and link to this remediation.

3. **Deployment checklist**
   - Clear existing cache (format change).
   - Deploy modifications.
   - Monitor logfire warnings for cache serialization failures.

4. **Rollback plan**
   - Gate new serialization helpers behind config flag if desired:
     ```python
     if not self.config.enable_serialized_cache_helpers:
         return super()._calculate_size(obj)
     ```
   - Disable flag to revert to previous behaviour without redeploy.

---

## Estimated Effort

| Workstream | Tasks | Estimate |
|------------|-------|----------|
| A | Metrics collector refactor + unit tests | 4 h |
| B | Cache serialization helper + integration | 6 h |
| C | Integration tests, docs, rollout | 3 h |

**Total**: ~13 hours.

---

## Success Criteria

- `record_pattern_strength` handles cached `PatternAnalysis` models without raising.
- Cache hit ratio for pattern analysis improves (baseline to be captured before rollout).
- All new unit/integration tests pass in CI.
- No new async/sync regressions in agent execution paths.

---

## Sign-off

| Role | Name | Date | Approval |
|------|------|------|----------|
| Technical Lead | | | ☐ |
| QA Lead | | | ☐ |
| Product Owner | | | ☐ |

---

**Document Version**: 3.1
**Last Updated**: 2025-01-17
**Author**: AI Architecture Team
**Status**: Draft – Pending Review
