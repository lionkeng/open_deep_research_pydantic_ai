# Robust Improvements to Source Tracking & Footnote Integration Plan

## Executive Summary

After architectural and Python engineering review of the original plan, this document identifies critical gaps and provides concrete improvements to make the implementation more robust, scalable, and maintainable.

## Critical Gaps & Solutions

### 1. Architecture & Design Patterns

#### Gap: Tight Coupling with Embedded Registry

**Original Plan**: SourceRegistry helper embedded in ResearchExecutorAgent
**Issue**: Violates Single Responsibility Principle, hard to test and maintain

**Solution**: Repository Pattern with Clear Boundaries

```python
class SourceRepository:
    """Domain aggregate for source management"""
    async def register_source(self, source: ResearchSource) -> SourceIdentity
    async def get_by_identity(self, identity: SourceIdentity) -> ResearchSource

class SourceIdentity:
    """Value object for source identity"""
    id: str  # S1, S2, etc.
    canonical_key: str  # For deduplication
    version: int  # Handle updates
```

#### Gap: Source ID Propagation Through Every Model

**Original Plan**: Add source_id fields to all models
**Issue**: Model pollution, maintenance nightmare

**Solution**: Context-Aware Tracking with ContextVars

```python
from contextvars import ContextVar

source_context: ContextVar[SourceTrackingContext] = ContextVar("source_tracking")

class SourceTrackingContext:
    repository: SourceRepository
    attribution_map: dict[str, list[SourceIdentity]]

# Models access context implicitly
class HierarchicalFinding:
    @property
    def source_ids(self) -> list[str]:
        ctx = source_context.get()
        return ctx.attribution_map.get(self.id, [])
```

### 2. Scalability & Performance with Context Engineering

#### Context Engineering Principles (Inspired by Manus)

The key insight from advanced AI agent systems is that context management is the critical bottleneck. We need to treat source context as a dynamic, externalized memory system that optimizes for both KV-cache efficiency and semantic coherence.

#### Gap: Naive In-Memory Storage

**Original Plan**: Keep all sources in memory
**Issue**: Memory exhaustion, no KV-cache optimization, context fragmentation

**Solution**: Context-Aware Source Management System

##### 1. Append-Only Source Registry with KV-Cache Optimization

```python
class AppendOnlySourceRegistry:
    """
    Optimized for KV-cache reuse by maintaining stable prefixes.
    New sources are always appended, never inserted in the middle.
    """
    def __init__(self):
        self._sources: list[ResearchSource] = []
        self._index_map: dict[str, int] = {}  # url/hash -> position
        self._context_breakpoints: list[int] = []  # Cache-friendly positions
        self._file_backed_store = FileBackedSourceStore()  # External context

    async def register(self, source: ResearchSource) -> SourceIdentity:
        """Always append to maintain stable context prefix"""
        position = len(self._sources)
        self._sources.append(source)

        # Mark breakpoint for KV-cache optimization
        if position % 50 == 0:  # Every 50 sources
            self._context_breakpoints.append(position)
            await self._flush_to_disk(position - 50, position)

        return SourceIdentity(f"S{position}", position)

    async def _flush_to_disk(self, start: int, end: int):
        """Use file system as ultimate context storage"""
        chunk = self._sources[start:end]
        await self._file_backed_store.persist_chunk(chunk, start)
        # Keep only reference in memory
        for i in range(start, end):
            self._sources[i] = SourceReference(i)
```

##### 2. Hierarchical Context Compression with Restoration

```python
class RestorableSourceCompressor:
    """
    Implements restorable compression that preserves critical metadata
    while reducing context size - key principle from Manus
    """

    def __init__(self):
        self.compression_levels = [
            self._level_0_full,      # Full source with content
            self._level_1_summary,   # Title, URL, key sentences
            self._level_2_metadata,  # Just title, URL, score
            self._level_3_reference  # Just source ID
        ]

    async def compress_context(
        self,
        sources: list[ResearchSource],
        target_tokens: int,
        preserve_recent: int = 10
    ) -> CompressedContext:
        """Progressive compression with restoration capability"""

        # Never compress recent sources (maintain working set)
        recent = sources[-preserve_recent:]
        older = sources[:-preserve_recent]

        compressed = []
        token_count = self._count_tokens(recent)

        # Apply progressive compression to older sources
        for i, source in enumerate(reversed(older)):
            distance_factor = i / len(older)  # Further = more compression
            level = min(3, int(distance_factor * 4))

            compressed_source = await self.compression_levels[level](source)
            compressed.insert(0, compressed_source)

            token_count += self._count_tokens([compressed_source])
            if token_count >= target_tokens:
                break

        return CompressedContext(
            sources=compressed + recent,
            compression_map=self._build_restoration_map(compressed),
            can_restore=True
        )

    async def restore_source(self, compressed: CompressedSource) -> ResearchSource:
        """Restore full source from compressed representation"""
        if compressed.has_full_content:
            return compressed.as_full_source()

        # Fetch from external storage
        return await self._file_backed_store.retrieve(compressed.source_id)
```

##### 3. Smart Context Pruning with Attention Manipulation

```python
class AttentionAwareContextManager:
    """
    Manages source context with attention manipulation techniques
    to prevent repetitive behavior and maintain focus
    """

    def __init__(self):
        self._attention_weights: dict[str, float] = {}
        self._access_history: deque = deque(maxlen=100)
        self._source_objectives: list[str] = []  # "Todo list" pattern

    async def prepare_context_for_agent(
        self,
        query: str,
        all_sources: list[ResearchSource],
        max_context_tokens: int = 8000
    ) -> AgentContext:
        """
        Prepare optimized context for agent consumption
        using attention manipulation principles
        """

        # 1. Compute relevance scores with attention decay
        scored_sources = []
        for source in all_sources:
            base_score = self._compute_relevance(query, source)

            # Apply attention decay based on access history
            access_count = self._access_history.count(source.source_id)
            decay_factor = 1.0 / (1.0 + access_count * 0.1)

            # Boost recently registered sources (recency bias)
            recency_boost = self._compute_recency_boost(source)

            final_score = base_score * decay_factor * recency_boost
            scored_sources.append((source, final_score))

        # 2. Sort by score but maintain some randomness to prevent loops
        scored_sources.sort(key=lambda x: x[1], reverse=True)

        # Introduce controlled randomness (top-k sampling)
        top_k = min(20, len(scored_sources))
        selected_indices = self._sample_with_temperature(
            scores=[s[1] for s in scored_sources[:top_k]],
            temperature=0.3,
            k=10
        )

        selected_sources = [scored_sources[i][0] for i in selected_indices]

        # 3. Create context with objectives (todo list pattern)
        context_objectives = [
            f"Find sources about: {query}",
            f"Already reviewed: {len(self._access_history)} sources",
            f"Focus on: {', '.join(self._extract_key_topics(selected_sources))}"
        ]

        # 4. Build final context with compression if needed
        if self._count_tokens(selected_sources) > max_context_tokens:
            compressed = await self.compressor.compress_context(
                selected_sources,
                target_tokens=max_context_tokens * 0.8,  # Leave room for objectives
                preserve_recent=5
            )
            selected_sources = compressed.sources

        # Record access for attention manipulation
        for source in selected_sources:
            self._access_history.append(source.source_id)

        return AgentContext(
            objectives=context_objectives,
            sources=selected_sources,
            compression_map=compressed.compression_map if compressed else None,
            cache_breakpoint=self._find_optimal_breakpoint(selected_sources)
        )
```

##### 4. File System as External Context Storage

```python
class FileBackedSourceStore:
    """
    Uses file system as ultimate context storage
    Key insight: File system provides unlimited, persistent context
    """

    def __init__(self, base_dir: Path = Path(".source_context")):
        self.base_dir = base_dir
        self.base_dir.mkdir(exist_ok=True)

        # Hierarchical storage for efficient retrieval
        self.chunks_dir = self.base_dir / "chunks"
        self.index_dir = self.base_dir / "indices"
        self.embeddings_dir = self.base_dir / "embeddings"

        for dir in [self.chunks_dir, self.index_dir, self.embeddings_dir]:
            dir.mkdir(exist_ok=True)

        # Memory-mapped index for fast lookups
        self._mmap_index = None
        self._init_mmap_index()

    async def persist_chunk(
        self,
        sources: list[ResearchSource],
        start_position: int
    ):
        """Persist source chunk to disk with indexing"""

        chunk_file = self.chunks_dir / f"chunk_{start_position:08d}.json"

        # Serialize with compression
        data = {
            "start": start_position,
            "sources": [s.model_dump() for s in sources],
            "timestamp": datetime.utcnow().isoformat()
        }

        # Use msgpack for efficient binary serialization
        import msgpack
        compressed = msgpack.packb(data, use_bin_type=True)

        async with aiofiles.open(chunk_file, 'wb') as f:
            await f.write(compressed)

        # Update indices
        await self._update_indices(sources, start_position)

    async def retrieve_by_similarity(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> list[ResearchSource]:
        """Retrieve sources using vector similarity search"""

        # Use FAISS for efficient similarity search
        if not hasattr(self, '_faiss_index'):
            await self._load_or_create_faiss_index()

        distances, indices = self._faiss_index.search(
            query_embedding.reshape(1, -1),
            top_k
        )

        # Retrieve sources from chunks
        sources = []
        for idx in indices[0]:
            if idx >= 0:
                source = await self._retrieve_by_position(int(idx))
                if source:
                    sources.append(source)

        return sources
```

##### 5. Content-Aware Deduplication with Semantic Hashing

```python
class SemanticHasher:
    """
    Advanced deduplication using semantic hashing
    Prevents duplicate content while maintaining O(1) lookup
    """

    def __init__(self):
        self._simhash_index = {}  # simhash -> source_ids
        self._lsh_forest = LSHForest(n_estimators=10, n_candidates=50)
        self._content_cache = ContentCache(max_size=1000)

    def compute_semantic_hash(self, source: ResearchSource) -> tuple[int, np.ndarray]:
        """Compute both SimHash and semantic embedding"""

        # Fast SimHash for exact duplicate detection
        features = self._extract_features(source.content)
        simhash = Simhash(features).value

        # Semantic embedding for near-duplicate detection
        if source.content in self._content_cache:
            embedding = self._content_cache[source.content]
        else:
            # Use lightweight model for efficiency
            embedding = self._compute_embedding(source.content[:512])
            self._content_cache[source.content] = embedding

        return simhash, embedding

    async def find_duplicates(
        self,
        source: ResearchSource,
        threshold: float = 0.85
    ) -> list[SourceIdentity]:
        """Find semantic duplicates efficiently"""

        simhash, embedding = self.compute_semantic_hash(source)

        # Check exact duplicates first (O(1))
        if simhash in self._simhash_index:
            return self._simhash_index[simhash]

        # Check near-duplicates using LSH (O(log n))
        candidates = self._lsh_forest.kneighbors(
            embedding.reshape(1, -1),
            n_neighbors=10,
            return_distance=True
        )

        duplicates = []
        for distance, idx in zip(candidates[0][0], candidates[1][0]):
            similarity = 1 - distance
            if similarity >= threshold:
                duplicates.append(self._get_source_by_index(idx))

        return duplicates
```

##### 6. Performance Monitoring and Adaptation

```python
class ContextPerformanceMonitor:
    """
    Monitors and adapts context strategies based on performance
    Key principle: Context engineering is experimental science
    """

    def __init__(self):
        self.metrics = {
            'cache_hit_rate': 0.0,
            'compression_ratio': 0.0,
            'retrieval_latency_p50': 0.0,
            'retrieval_latency_p99': 0.0,
            'dedup_effectiveness': 0.0,
            'context_tokens_used': 0
        }
        self._adaptation_history = []

    async def adapt_strategy(self):
        """Dynamically adjust context strategies based on metrics"""

        # Analyze recent performance
        if self.metrics['cache_hit_rate'] < 0.7:
            # Adjust cache size or eviction policy
            await self._increase_cache_size()

        if self.metrics['retrieval_latency_p99'] > 100:  # ms
            # Enable more aggressive prefetching
            await self._enable_prefetching()

        if self.metrics['context_tokens_used'] > 7000:
            # Increase compression aggressiveness
            await self._adjust_compression_threshold(0.8)

        # Record adaptation for learning
        self._adaptation_history.append({
            'timestamp': datetime.utcnow(),
            'metrics': self.metrics.copy(),
            'adaptations': self._current_adaptations
        })
```

#### Implementation Requirements Summary

1. **Append-Only Architecture**: Maintain stable context prefixes for KV-cache optimization
2. **Restorable Compression**: Never lose information permanently, always allow restoration
3. **File System Backend**: Use disk as unlimited context storage with memory-mapped indices
4. **Attention Manipulation**: Prevent loops through controlled randomness and access tracking
5. **Semantic Deduplication**: Use SimHash + LSH for O(log n) duplicate detection
6. **Dynamic Adaptation**: Monitor performance and adjust strategies automatically
7. **Hierarchical Storage**: Organize sources in chunks with multiple index types
8. **Context Objectives**: Maintain "todo lists" to guide agent attention

These implementations follow the core principle from Manus: treat context as dynamic, externalized memory that requires careful engineering for optimal agent performance.

### 3. Error Handling & Recovery

#### Gap: No Recovery Strategy for Failures

**Original Plan**: Basic error logging only
**Issue**: System fails completely on partial errors

**Solution**: Resilient Pipeline with Circuit Breakers

```python
class SourceValidationPipeline:
    async def validate_and_register(self, raw_source: dict) -> SourceIdentity | None:
        try:
            # Stage 1: URL validation with timeout
            validated_url = await self._validate_url(raw_source.get("url"))

            # Stage 2: Circuit breaker for external calls
            if self.circuit_breaker.is_closed:
                await self._verify_content_availability(validated_url)

            # Stage 3: Register with fallback
            return await self.repository.register_source(source)

        except SourceValidationError as e:
            # Graceful degradation
            return await self._register_degraded_source(raw_source, e)
```

### 4. Python-Specific Improvements

#### Enhanced Pydantic Models

```python
class ResearchSource(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,  # Runtime validation
        use_enum_values=True,
        extra="forbid"  # Prevent unknown fields
    )

    # Smart validators
    @field_validator("source_id", mode="before")
    @classmethod
    def generate_source_id(cls, v: Any, info) -> str:
        if not v and "url" in info.data:
            return f"S_{hashlib.md5(info.data['url'].encode()).hexdigest()[:8]}"
        return v
```

#### Async Best Practices

```python
# Concurrent validation with proper error handling
async def validate_sources_concurrently(
    sources: List[ResearchSource],
    max_concurrent: int = 10
) -> List[ResearchSource]:
    async with AsyncExitStack() as stack:
        validator = await stack.enter_async_context(URLValidator(max_concurrent))

        # Create tasks for better performance
        tasks = [asyncio.create_task(validator.validate(source)) for source in sources]

        # Gather with exception handling
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                source.validation_state = "failed"
```

#### Type Safety Enhancements

```python
from typing import TypeAlias, Protocol, TypeGuard

SourceID: TypeAlias = str
FootnoteNumber: TypeAlias = int

class FootnoteFormatter(Protocol):
    def format_inline(self, source_id: SourceID) -> str: ...
    def format_footnote(self, number: FootnoteNumber, source: ResearchSource) -> str: ...

def is_validated_source(source: ResearchSource) -> TypeGuard[ResearchSource]:
    return source.validation_state == "validated"
```

### 5. Testing Improvements

#### Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(
    sources=st.lists(
        st.builds(ResearchSource,
            url=st.text(min_size=10),
            title=st.text(min_size=1),
            content=st.text(min_size=10),
            relevance_score=st.floats(min_value=0.0, max_value=1.0)
        ),
        min_size=1,
        max_size=100
    )
)
def test_deduplication_preserves_unique_content(sources):
    registry = SourceRegistry()
    registered = [registry.register(s) for s in sources]

    # Property: All unique content preserved
    unique_content = set(s.content for s in sources)
    registered_content = set(r.source.content for r in registered)
    assert unique_content == registered_content
```

#### Async Test Patterns

```python
@pytest.mark.asyncio
async def test_concurrent_registration():
    sources = [create_test_source(i) for i in range(100)]

    # Register concurrently
    tasks = [registry.register(source) for source in sources]
    ids = await asyncio.gather(*tasks)

    # Verify uniqueness and consistency
    assert len(set(ids)) == len(sources)
```

## Implementation Priority Matrix

| Component          | Priority | Effort | Risk   | Week |
| ------------------ | -------- | ------ | ------ | ---- |
| Repository Pattern | HIGH     | Medium | Low    | 1    |
| Context Tracking   | HIGH     | High   | Medium | 1    |
| Tiered Caching     | MEDIUM   | Medium | Low    | 2    |

## Anti-Patterns to Avoid

1. **God Object**: Don't let SourceRegistry handle everything
2. **Anemic Domain Model**: Sources need behavior, not just data
3. **Primitive Obsession**: Use value objects (SourceIdentity) not strings
4. **Temporal Coupling**: Avoid strict operation ordering
5. **Hidden Dependencies**: Make tracking dependencies explicit
6. **Blocking I/O in Async**: Always use aiohttp/httpx, never requests
7. **Mutable Defaults**: Never use mutable defaults in Pydantic models

## Performance Targets

- Source registration: < 100ms per source
- Batch registration: < 10ms per source (amortized)
- Deduplication check: < 10ms
- Footnote processing: < 1s for 100 citations
- Memory usage: < 10MB per 1000 sources
- Cache hit rate: > 80% for L1, > 95% for L1+L2

## Success Criteria

1. **Functional**

   - 100% source attribution coverage
   - Zero orphaned citations
   - Basic source validation

2. **Performance**

   - Meet all performance targets
   - No memory leaks
   - Graceful degradation under load

3. **Quality**

   - > 90% deduplication accuracy
   - > 95% test coverage
   - Clean, maintainable code

4. **Maintainability**
   - Clean architecture boundaries
   - Comprehensive documentation
   - Easy to extend with plugins

## Conclusion

The original plan provides a good foundation but needs significant architectural and implementation improvements to be production-ready. Key areas requiring attention:

1. **Architecture**: Move from embedded registry to repository pattern
2. **Scalability**: Implement tiered caching and context engineering principles
3. **Quality**: Add source scoring leveraging existing semantic clustering
4. **Testing**: Implement property-based and async testing patterns

These improvements will create a robust and scalable source tracking system that can evolve with requirements while maintaining high performance and code quality.
