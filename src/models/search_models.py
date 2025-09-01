"""Pydantic models for search providers and responses."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class SearchProvider(str, Enum):
    """Available search providers."""

    TAVILY = "tavily"
    SERPER = "serper"
    DUCKDUCKGO = "duckduckgo"
    MOCK = "mock"


class SearchDepth(str, Enum):
    """Search depth options."""

    BASIC = "basic"
    ADVANCED = "advanced"


class TavilySearchParams(BaseModel):
    """Tavily-specific search parameters."""

    search_depth: SearchDepth = Field(default=SearchDepth.ADVANCED, description="Search depth")
    include_answer: bool = Field(default=False, description="Include AI-generated answer")
    include_raw_content: bool = Field(default=False, description="Include raw HTML content")
    include_images: bool = Field(default=False, description="Include image results")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum results to return")


class TavilySearchResult(BaseModel):
    """Tavily search result format."""

    title: str = Field(description="Result title")
    url: str = Field(description="Result URL")
    content: str = Field(description="Content snippet")
    score: float = Field(default=0.5, ge=0.0, le=1.0, description="Relevance score")
    published_date: str | None = Field(default=None, description="Publication date")
    author: str | None = Field(default=None, description="Content author")
    raw_content: str | None = Field(default=None, description="Raw HTML content if requested")


class TavilySearchResponse(BaseModel):
    """Tavily API response format."""

    query: str = Field(description="Search query")
    results: list[TavilySearchResult] = Field(description="Search results")
    answer: str | None = Field(default=None, description="AI-generated answer if requested")
    response_time: float = Field(default=0.0, description="API response time")
    images: list[str] | None = Field(default=None, description="Image URLs if requested")


class SearchProviderConfig(BaseModel):
    """Configuration for a search provider."""

    model_config = ConfigDict(extra="forbid")

    provider: SearchProvider = Field(description="Provider type")
    api_key: str | None = Field(default=None, description="API key if required")
    base_url: HttpUrl | None = Field(default=None, description="Custom base URL")
    timeout: float = Field(default=30.0, gt=0, le=120, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    default_params: dict[str, Any] = Field(
        default_factory=dict, description="Default search parameters"
    )


class UnifiedSearchRequest(BaseModel):
    """Unified search request across providers."""

    query: str = Field(min_length=1, max_length=1000, description="Search query")
    num_results: int = Field(default=5, ge=1, le=50, description="Number of results")
    provider: SearchProvider | None = Field(
        default=None, description="Specific provider or None for auto"
    )
    provider_params: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific parameters"
    )


class UnifiedSearchResult(BaseModel):
    """Unified search result format."""

    model_config = ConfigDict(extra="allow")

    title: str = Field(description="Result title")
    url: HttpUrl = Field(description="Result URL")
    snippet: str = Field(description="Content snippet")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score")
    provider: SearchProvider = Field(description="Source provider")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class UnifiedSearchResponse(BaseModel):
    """Unified search response."""

    request: UnifiedSearchRequest = Field(description="Original request")
    results: list[UnifiedSearchResult] = Field(description="Search results")
    total_results: int = Field(description="Total number of results")
    response_time: float = Field(description="Total response time in seconds")
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")
