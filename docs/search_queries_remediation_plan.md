# Research Workflow Search Queries Remediation Plan

## Executive Summary

This document provides a comprehensive plan to fix critical design issues and bugs in the research workflow system, specifically related to how search queries are handled throughout the pipeline. The plan addresses 5 major issues with a phased implementation approach prioritized by impact and risk.

## Issues Identified

### 1. **[CRITICAL] Timing/Data Flow Bug**
- **Location**: `src/core/workflow.py` line 482-486
- **Problem**: `_execute_search_queries(deps)` runs while no `SearchQueryBatch` has been attached to the dependencies
- **Impact**: Search execution no-ops (returns empty results), so the Research Executor runs without any findings
- **Root Cause**: Search queries are only re-materialized inside `_run_agent_with_circuit_breaker` after the failed call path

### 2. **[HIGH] Dead Code**
- **Location**: `src/models/metadata.py` line 120
- **Problem**: `QueryMetadata.search_queries` field is never used anywhere
- **Impact**: Confusion for developers, wasted memory
- **Evidence**: No code reads or writes to this field

### 3. **[MEDIUM] Multiple Redundancies**
- **Problem**: `search_queries` is defined in 4+ places:
  - `TransformedQuery.search_queries` (source of truth)
  - `ResearchDependencies.search_queries` (for passing data)
  - `QueryMetadata.search_queries` (unused)
  - `metadata.query.transformed_query["search_queries"]` (serialized)
- **Impact**: Confusion about source of truth, maintenance burden

### 4. **[MEDIUM] Parameter Redundancy**
- **Location**: `src/core/workflow.py` line 467
- **Problem**: `_execute_research_stages` takes both `research_state` and `deps`, but `deps.research_state` contains the same object
- **Impact**: Violates DRY principle, confusion about which to use

### 5. **[LOW] Poor Data Flow Design**
- **Problem**: Excessive serialization/deserialization instead of passing objects
- **Impact**: Performance overhead, potential data loss

## Implementation Plan

### Phase 1: Establish Single Source of Truth + Fix Timing Bug (Immediate - 1.5 hours)

#### Changes Required

**Files:**

- `src/models/metadata.py`
- `src/models/research_plan_models.py`
- `src/core/workflow.py`
- `src/agents/base.py`

**Key Steps:**

1. **Store the `TransformedQuery` object directly in metadata**
   - Change `QueryMetadata.transformed_query` to type `TransformedQuery | None` (instead of `dict[str, Any]`).
   - Update `_execute_two_phase_clarification` to assign the returned `enhanced_query` object directly:
     ```python
     research_state.metadata.query.transformed_query = enhanced_query
     ```
     This keeps a single, typed copy of the search queries and research plan.

2. **Provide a zero-copy accessor on dependencies**
   - Add a helper on `ResearchDependencies` (e.g., `def get_search_query_batch(self) -> SearchQueryBatch | None`) that returns
     `self.research_state.metadata.query.transformed_query.search_queries` when available.
   - Update all call sites to rely on the accessor immediately—no fallback or duplicate storage.

3. **Plumb the accessor through the workflow**
   - In `_execute_research_stages`, fetch the batch via the accessor before calling `_execute_search_queries`:
     ```python
     batch = deps.get_search_query_batch()
     if not batch:
         raise ValueError("Missing SearchQueryBatch prior to search execution")
     deps.search_results = await self._execute_search_queries(batch, deps)
     ```
   - Update `_execute_search_queries` to accept the batch directly (or call the accessor internally) instead of reading `deps.search_queries`.

4. **Remove the redundant reconstruction block**
   - Delete lines 200-219 in `_run_agent_with_circuit_breaker`; the Research Executor should rely on `deps.get_search_query_batch()` inside its agent implementation and not re-attach data.


#### Testing
```python
# tests/test_search_queries_timing.py
async def test_search_queries_present_before_execution():
    """Search execution uses the batch stored on metadata without copying."""
    # Arrange workflow with a TransformedQuery already on metadata
    # Assert that get_search_query_batch returns the same instance the transformer produced
    # Verify _execute_search_queries receives a populated batch
```

### Phase 2: Remove Dead Code (30 minutes)

#### Changes Required

**File: `src/models/metadata.py`**

**Line 120 - DELETE this line entirely:**
```python
# REMOVE THIS LINE:
search_queries: list[str] = Field(default_factory=list, description="Search queries executed")
```

**The class should look like:**
```python
class QueryMetadata(BaseModel):
    """Metadata specific to the query transformation agent."""

    model_config = ConfigDict(validate_assignment=True)

    transformed_query: TransformedQuery | None = Field(
        default=None, description="Query transformation results"
    )
    # search_queries line REMOVED - it was never used
```

#### Verification
```bash
# Ensure no references remain:
grep -r "metadata.query.search_queries" src/ tests/
# Should return no results
```

### Phase 3: Fix Parameter Redundancy (1 hour)

#### Changes Required

**File: `src/core/workflow.py`**

**Update Method Signature (line 465):**
```python
# BEFORE:
async def _execute_research_stages(
    self,
    research_state: ResearchState,  # REDUNDANT
    deps: ResearchDependencies,
    stream_callback: Any | None = None,
) -> None:

# AFTER:
async def _execute_research_stages(
    self,
    deps: ResearchDependencies,
    stream_callback: Any | None = None,
) -> None:
    """Execute the main research stages after clarification and transformation."""
    research_state = deps.research_state  # Get from deps
    # Rest of method unchanged...
```

**Update Call Sites:**
```python
# Line 439 - BEFORE:
await self._execute_research_stages(research_state, deps, stream_callback)

# Line 439 - AFTER:
await self._execute_research_stages(deps, stream_callback)

# Line 715 - BEFORE:
await self._execute_research_stages(research_state, deps, stream_callback)

# Line 715 - AFTER:
await self._execute_research_stages(deps, stream_callback)
```

### Phase 4: Clean Up Redundant Fields (1 hour)

#### Changes Required

1. Remove `search_queries` from `ResearchDependencies` once all call sites rely on the accessor.
2. Update `_execute_search_queries` signature to accept the batch directly.
3. Ensure `ResearchExecutorAgent` and related helpers (`_summarize_search_queries`, logging, etc.) call the accessor rather than touching a field.

#### Final State
```python
# Only ONE place owns the search batch:
class QueryMetadata(BaseModel):
    transformed_query: TransformedQuery | None

# Access pattern:
# 1. Query Transformation returns TransformedQuery (with SearchQueryBatch inside)
# 2. Workflow stores the object on metadata
# 3. Downstream consumers obtain it via ResearchDependencies.get_transformed_query()/get_search_query_batch()
# 4. No serialization/deserialization unless persisting state externally
```

## Testing Strategy

### Unit Tests

```python
# tests/unit/test_search_queries_fix.py
class TestSearchQueriesRemediation:
    async def test_query_batch_attached_before_execution(self):
        """get_search_query_batch returns the same instance produced by transformation."""

    def test_metadata_holds_single_transformed_query(self):
        """QueryMetadata stores a TransformedQuery object and no redundant list field."""

    async def test_execute_research_stages_uses_accessor(self):
        """_execute_research_stages calls accessor and raises if batch missing."""
```

### Integration Tests

```python
# tests/integration/test_workflow_search_queries.py
async def test_full_workflow_executes_search_queries():
    """End-to-end verification that searches run and findings propagate."""
    # Run workflow with mocked search service
    # Assert the orchestrator receives the expected SearchQueryBatch and returns results
    # Confirm Research Executor instructions see accurate query counts
```

### Regression Coverage

- Update (or add) a CLI smoke test to ensure `uv run python -m src.cli ...` performs real searches after remediation.
- Extend `tests/integration/test_research_executor_integration.py` with a fixture that injects a `SearchQueryBatch` via the accessor to prove downstream agents consume the new data path.

### CLI Smoke Test

```bash
# Manual verification
uv run python -m src.cli research "What are the latest AI developments?" --verbose

# Should complete without errors
# Should show search queries being executed
```

## Risk Assessment & Mitigation

| Phase | Risk Level | Mitigation Strategy |
|-------|------------|-------------------|
| 1 | MEDIUM | Update all call sites and run focused unit tests on accessor path |
| 2 | LOW | Grep for usage before removal |
| 3 | LOW | Update all call sites |
| 4 | MEDIUM | Coordinate field removal with thorough test pass |

## Rollback Procedures

### Git-Based Rollback
```bash
# Tag before each phase
git tag pre-phase1
git commit -m "Phase 1: Fix timing bug"
git tag phase1-complete

# To rollback
git checkout pre-phase1
```

### File Backup
```bash
# Before each modification
cp src/core/workflow.py src/core/workflow.py.backup
cp src/models/metadata.py src/models/metadata.py.backup
```

## Success Metrics

### Phase 1 Success
- ✅ `_execute_research_stages` acquires a populated `SearchQueryBatch` via the accessor before search execution
- ✅ Search orchestrator receives non-empty batches in automated tests
- ✅ All updated call sites rely on the accessor without fallback data paths

### Overall Success
- ✅ Redundant search-query fields removed or deprecated
- ✅ Single source of truth for search queries enforced through `QueryMetadata.transformed_query`
- ✅ No parameter redundancy
- ✅ Improved code maintainability score
- ✅ Zero regression in functionality

## Implementation Timeline

| Phase | Duration | Dependencies | Start Date | Completion |
|-------|----------|--------------|------------|------------|
| 1 | 1.5 hours | None | Immediate | [ ] |
| 2 | 30 min | Phase 1 | After 1 | [ ] |
| 3 | 1 hour | Phase 1-2 | After 2 | [ ] |
| 4 | 1 hour | Phase 1-3 | After 3 | [ ] |

**Total Estimated Time**: 4 hours

## Command Reference

```bash
# Run specific tests
uv run pytest tests/test_search_queries_timing.py -v

# Run all tests
uv run pytest

# Type checking
uv run pyright src

# Linting
uv run ruff check src tests

# Test CLI
uv run python -m src.cli research "test query" --verbose

# Check for usage of a field
grep -r "search_queries" src/ tests/ --include="*.py"

# Find class definitions
grep -n "class.*QueryMetadata" src/ --include="*.py"
```

## Notes for Implementers

1. **Always test after each phase** - Don't batch changes
2. **Keep backups** - Use git tags or file copies
3. **Watch for circular imports** when moving code
4. **Check type hints** after removing fields
5. **Update documentation** if public APIs change

## Appendix: Current vs. Future State

### Current State (Buggy)
```
TransformedQuery created → Serialized to dict → Stored in metadata
                                                          ↓
Research workflow attempts search → Accessor finds nothing → `_execute_search_queries` returns []
                                                          ↓
Research Executor runs without findings
```

### Future State (Fixed)
```
TransformedQuery created → Stored (typed) on metadata
                                ↓
                Accessor returns stored SearchQueryBatch
                                ↓
                    Execute search queries
                                ↓
                         Research continues
```

## Review Checklist

Before marking each phase complete:

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] CLI works manually
- [ ] No new linting errors
- [ ] Type checking passes
- [ ] Performance unchanged or improved
- [ ] Documentation updated if needed
- [ ] Git commit with descriptive message
- [ ] Tagged with phase number

---

*Document Version: 1.0*
*Last Updated: [Current Date]*
*Author: Python Expert Engineer Agent*
*Status: Ready for Implementation*
