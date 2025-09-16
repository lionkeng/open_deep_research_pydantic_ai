# Research Workflow Search Queries Remediation Plan

## Executive Summary

This document provides a comprehensive plan to fix critical design issues and bugs in the research workflow system, specifically related to how search queries are handled throughout the pipeline. The plan addresses 5 major issues with a phased implementation approach prioritized by impact and risk.

## Issues Identified

### 1. **[CRITICAL] Timing Bug**
- **Location**: `src/core/workflow.py` line 484
- **Problem**: `_execute_search_queries(deps)` is called before `deps.search_queries` is populated
- **Impact**: CLI crashes with AttributeError
- **Root Cause**: Search queries are extracted from metadata AFTER execution is attempted

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

### Phase 1: Critical Timing Bug Fix (Immediate - 1 hour)

#### Changes Required

**File: `src/core/workflow.py`**

**Current Buggy Code (around line 483-484):**
```python
# Research executor now receives SearchQueryBatch directly
if not getattr(deps, "search_results", None):
    deps.search_results = await self._execute_search_queries(deps)  # BUG: deps.search_queries is None!
```

**Fixed Code:**
```python
# Extract search queries from metadata FIRST
if (deps.research_state.metadata
    and deps.research_state.metadata.query.transformed_query):
    transformed_query_data = deps.research_state.metadata.query.transformed_query
    from models.search_query_models import SearchQueryBatch
    search_queries_data = transformed_query_data.get("search_queries", {})
    if search_queries_data:
        deps.search_queries = SearchQueryBatch.model_validate(search_queries_data)
        logfire.info(
            "Extracted search queries from metadata",
            num_queries=len(deps.search_queries.queries) if deps.search_queries else 0
        )

# NOW execute search queries (they're available)
if not getattr(deps, "search_results", None):
    deps.search_results = await self._execute_search_queries(deps)
```

**Also Remove Duplicate Logic (lines 202-219):**
```python
# DELETE or simplify this entire block in _run_agent_with_circuit_breaker
# since we're extracting in _execute_research_stages instead
```

#### Testing
```python
# tests/test_search_queries_timing.py
async def test_search_queries_extracted_before_execution():
    """Verify search queries are available when _execute_search_queries is called"""
    # Mock workflow with metadata containing search_queries
    # Verify deps.search_queries is populated before execution
    # Assert no AttributeError
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

    transformed_query: dict[str, Any] | None = Field(
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

### Phase 4: Consolidate Search Queries (2 hours)

#### Design Goals
- Single source of truth: `TransformedQuery.search_queries`
- Pass by reference, not copy
- Remove duplicate storage

#### Changes Required

**Step 1: Add Backward Compatibility Property**

**File: `src/agents/base.py`**
```python
class ResearchDependencies(BaseModel):
    # Keep the field for now but deprecate it
    search_queries: SearchQueryBatch | None = None  # Will be removed in Phase 5

    @property
    def get_search_queries(self) -> SearchQueryBatch | None:
        """Get search queries from metadata if not directly set."""
        if self.search_queries:
            return self.search_queries

        # Fallback to metadata
        if (self.research_state.metadata
            and self.research_state.metadata.query.transformed_query):
            data = self.research_state.metadata.query.transformed_query.get("search_queries")
            if data:
                return SearchQueryBatch.model_validate(data)
        return None
```

**Step 2: Update Usage Pattern**
```python
# Throughout codebase, replace:
deps.search_queries
# With:
deps.get_search_queries
```

### Phase 5: Final Cleanup (1 hour)

#### Changes Required

1. **Remove `search_queries` from `ResearchDependencies`** after ensuring all code uses the property
2. **Simplify `_execute_search_queries` method** to use the property
3. **Remove all duplicate extraction logic**

#### Final State
```python
# Only ONE place defines search_queries:
class TransformedQuery(BaseModel):
    search_queries: SearchQueryBatch  # Source of truth

# Access pattern:
# 1. Query Transformation creates TransformedQuery
# 2. Stored in metadata.query.transformed_query
# 3. Accessed via deps.research_state.metadata when needed
# 4. No duplicate storage
```

## Testing Strategy

### Unit Tests

```python
# tests/unit/test_search_queries_fix.py
class TestSearchQueriesRemediation:
    async def test_timing_fix(self):
        """Verify extraction happens before execution"""

    def test_no_dead_fields(self):
        """Verify QueryMetadata has no unused fields"""

    async def test_single_parameter_pattern(self):
        """Verify _execute_research_stages uses only deps"""

    def test_single_source_of_truth(self):
        """Verify search_queries defined in only one place"""
```

### Integration Tests

```python
# tests/integration/test_workflow_search_queries.py
async def test_full_workflow_with_search_queries():
    """End-to-end test of search query handling"""
    # Create workflow
    # Verify query transformation produces search_queries
    # Verify execution receives search_queries
    # Verify no errors or duplications
```

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
| 1 | LOW | Test thoroughly, keep backup |
| 2 | LOW | Grep for usage before removal |
| 3 | LOW | Update all call sites |
| 4 | MEDIUM | Use compatibility property |
| 5 | MEDIUM | Gradual migration with deprecation |

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
- ✅ CLI runs without AttributeError
- ✅ Search queries execute successfully
- ✅ All existing tests pass

### Overall Success
- ✅ 50% reduction in redundant field definitions
- ✅ Single source of truth for search_queries
- ✅ No parameter redundancy
- ✅ Improved code maintainability score
- ✅ Zero regression in functionality

## Implementation Timeline

| Phase | Duration | Dependencies | Start Date | Completion |
|-------|----------|--------------|------------|------------|
| 1 | 1 hour | None | Immediate | [ ] |
| 2 | 30 min | Phase 1 | After 1 | [ ] |
| 3 | 1 hour | Phase 1-2 | After 2 | [ ] |
| 4 | 2 hours | Phase 1-3 | After 3 | [ ] |
| 5 | 1 hour | Phase 1-4 | After 4 | [ ] |

**Total Estimated Time**: 5.5 hours

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
ResearchExecutor called → deps.search_queries is None ← Not extracted yet!
            ↓
    AttributeError!
```

### Future State (Fixed)
```
TransformedQuery created → Stored in metadata
                                ↓
                        Extract to deps
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
