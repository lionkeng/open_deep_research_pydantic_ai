"""Data models for search queries and batch execution."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class SearchQueryType(str, Enum):
    """Types of search queries with different processing strategies."""

    FACTUAL = "factual"  # Seeking specific facts or data
    ANALYTICAL = "analytical"  # Requiring analysis or interpretation
    EXPLORATORY = "exploratory"  # Broad exploration of a topic
    COMPARATIVE = "comparative"  # Comparing multiple items/concepts
    TEMPORAL = "temporal"  # Time-based or historical queries


class SearchSource(str, Enum):
    """Available search sources with different content types."""

    WEB_GENERAL = "web_general"  # General web search
    ACADEMIC = "academic"  # Academic papers and journals
    NEWS = "news"  # News articles and current events
    TECHNICAL_DOCS = "technical_docs"  # Technical documentation
    GOVERNMENT = "government"  # Government sources
    INDUSTRY_REPORTS = "industry_reports"  # Industry analysis
    SOCIAL_MEDIA = "social_media"  # Social media content
    MEDICAL_JOURNALS = "medical_journals"  # Medical/health sources


class ExecutionStrategy(str, Enum):
    """Strategies for executing search query batches."""

    SEQUENTIAL = "sequential"  # Execute queries one by one
    PARALLEL = "parallel"  # Execute all queries simultaneously
    ADAPTIVE = "adaptive"  # Adjust strategy based on results
    HIERARCHICAL = "hierarchical"  # Execute based on dependencies


class TemporalContext(BaseModel):
    """Temporal context for time-sensitive queries."""

    start_date: str | None = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: str | None = Field(default=None, description="End date (YYYY-MM-DD)")
    recency_preference: str | None = Field(
        default=None, description="Preference for recent content (e.g., 'last_year', 'last_month')"
    )
    historical_context: str | None = Field(
        default=None, description="Historical context if relevant"
    )


class SearchQuery(BaseModel):
    """Individual search query with metadata and execution parameters."""

    # Core query fields
    id: str = Field(description="Unique identifier for this query")
    query: str = Field(..., min_length=3, max_length=500, description="The search query string")
    query_type: SearchQueryType = Field(
        default=SearchQueryType.FACTUAL, description="Type of query for processing"
    )

    # Prioritization and execution
    priority: int = Field(default=3, ge=1, le=5, description="Priority level (1=highest, 5=lowest)")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results to retrieve")

    # Context and targeting
    search_sources: list[SearchSource] = Field(
        default_factory=list, description="Preferred search sources"
    )
    temporal_context: TemporalContext | None = Field(
        default=None, description="Temporal context if applicable"
    )

    # Reasoning and expectations
    rationale: str = Field(description="Why this query is needed")
    expected_result_type: str = Field(
        default="", description="Type of results expected (e.g., statistics, examples, definitions)"
    )

    # Linkage to objectives
    objective_id: str | None = Field(
        default=None, description="ID of the research objective this query serves"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensure query is well-formed."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v

    @field_validator("search_sources")
    @classmethod
    def validate_sources(cls, v: list[SearchSource]) -> list[SearchSource]:
        """Ensure unique sources."""
        return list(set(v)) if v else []


class SearchQueryBatch(BaseModel):
    """Batch of search queries with execution strategy."""

    queries: list[SearchQuery] = Field(
        ..., min_length=1, max_length=20, description="List of search queries"
    )
    execution_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.ADAPTIVE, description="How to execute the queries"
    )

    # Execution parameters
    max_parallel: int = Field(default=5, ge=1, le=10, description="Max parallel queries")
    timeout_seconds: int = Field(
        default=30, ge=10, le=120, description="Timeout per query in seconds"
    )
    total_timeout: int = Field(
        default=300, ge=60, le=600, description="Total timeout for batch in seconds"
    )

    # Metadata
    estimated_queries: int = Field(
        default=0, description="Estimated total queries (including potential follow-ups)"
    )

    @model_validator(mode="after")
    def validate_batch(self) -> "SearchQueryBatch":
        """Validate the batch configuration."""
        # Set estimated queries if not provided
        if self.estimated_queries == 0:
            self.estimated_queries = len(self.queries)

        # Validate unique IDs
        query_ids = [q.id for q in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("Query IDs must be unique within a batch")

        # Validate strategy matches queries
        if self.execution_strategy == ExecutionStrategy.HIERARCHICAL:
            # Check that high priority queries come first
            priorities = [q.priority for q in self.queries]
            if priorities != sorted(priorities):
                # Re-sort by priority for hierarchical execution
                self.queries.sort(key=lambda q: q.priority)

        return self

    def get_queries_by_priority(self, priority: int) -> list[SearchQuery]:
        """Get all queries with a specific priority."""
        return [q for q in self.queries if q.priority == priority]

    def get_queries_for_objective(self, objective_id: str) -> list[SearchQuery]:
        """Get all queries linked to a specific objective."""
        return [q for q in self.queries if q.objective_id == objective_id]

    def get_execution_groups(self) -> list[list[SearchQuery]]:
        """Get query groups based on execution strategy."""
        if self.execution_strategy == ExecutionStrategy.SEQUENTIAL:
            # Each query is its own group
            return [[q] for q in self.queries]
        elif self.execution_strategy == ExecutionStrategy.PARALLEL:
            # All queries in one group
            return [self.queries]
        elif self.execution_strategy == ExecutionStrategy.HIERARCHICAL:
            # Group by priority
            groups = {}
            for q in self.queries:
                if q.priority not in groups:
                    groups[q.priority] = []
                groups[q.priority].append(q)
            return [groups[p] for p in sorted(groups.keys())]
        else:  # ADAPTIVE
            # Default to priority groups with max parallel limit
            groups = []
            current_group = []
            for q in sorted(self.queries, key=lambda x: x.priority):
                current_group.append(q)
                if len(current_group) >= self.max_parallel:
                    groups.append(current_group)
                    current_group = []
            if current_group:
                groups.append(current_group)
            return groups
