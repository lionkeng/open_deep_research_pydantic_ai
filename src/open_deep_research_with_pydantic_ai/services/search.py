"""Search service implementations for research execution."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, cast

import httpx
import logfire
from pydantic import BaseModel, ConfigDict, Field

from open_deep_research_with_pydantic_ai.core.config import config


class SearchResult(BaseModel):
    """Standardized search result."""

    model_config = ConfigDict(extra="allow")

    title: str = Field(description="Result title")
    url: str = Field(description="Result URL")
    snippet: str = Field(description="Content snippet")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchResponse(BaseModel):
    """Standardized search response."""

    query: str = Field(description="Search query")
    results: list[SearchResult] = Field(description="Search results")
    total_results: int = Field(description="Total number of results")
    source: str = Field(description="Search provider")


class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @abstractmethod
    async def search(self, query: str, num_results: int = 5, **kwargs: Any) -> SearchResponse:
        """Execute a search query.

        Args:
            query: Search query
            num_results: Number of results to return
            **kwargs: Provider-specific parameters

        Returns:
            Search response
        """
        pass


class TavilySearchProvider(SearchProvider):
    """Tavily search API provider."""

    def __init__(self, api_key: str | None = None):
        """Initialize Tavily provider.

        Args:
            api_key: Tavily API key or None to use from config
        """
        self.api_key = api_key or config.tavily_api_key
        self.base_url = "https://api.tavily.com"

    async def search(
        self, query: str, num_results: int = 5, search_depth: str = "advanced", **kwargs: Any
    ) -> SearchResponse:
        """Search using Tavily API.

        Args:
            query: Search query
            num_results: Number of results
            search_depth: "basic" or "advanced"
            **kwargs: Additional parameters

        Returns:
            Search response
        """
        if not self.api_key:
            logfire.warning("Tavily API key not configured, using mock results")
            return self._mock_search(query, num_results)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": num_results,
                        "search_depth": search_depth,
                        "include_answer": False,
                        "include_raw_content": False,
                        **kwargs,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()

                data = response.json()

                # Convert to standardized format
                results = []
                for item in data.get("results", []):
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            snippet=item.get("content", ""),
                            score=item.get("score", 0.5),
                            metadata={
                                "published_date": item.get("published_date"),
                            },
                        )
                    )

                return SearchResponse(
                    query=query, results=results, total_results=len(results), source="tavily"
                )

            except httpx.HTTPError as e:
                logfire.error(f"Tavily search failed: {str(e)}")
                return self._mock_search(query, num_results)

    def _mock_search(self, query: str, num_results: int) -> SearchResponse:
        """Fallback mock search for testing."""
        results: list[SearchResult] = []
        for i in range(min(num_results, 3)):
            results.append(
                SearchResult(
                    title=f"Result {i + 1}: {query}",
                    url=f"https://example.com/result{i + 1}",
                    snippet=f"This is relevant information about {query}. "
                    f"It provides comprehensive details that address the research question.",
                    score=0.9 - (i * 0.1),
                )
            )

        return SearchResponse(
            query=query, results=results, total_results=len(results), source="mock"
        )


class WebSearchService:
    """Unified web search service supporting multiple providers."""

    def __init__(self):
        """Initialize search service."""
        self.providers: dict[str, SearchProvider] = {}
        self._setup_providers()

    def _setup_providers(self) -> None:
        """Set up available search providers."""
        # Add Tavily if configured
        if config.tavily_api_key:
            self.providers["tavily"] = TavilySearchProvider()

        # Add more providers as needed
        # self.providers["serper"] = SerperSearchProvider()
        # self.providers["duckduckgo"] = DuckDuckGoProvider()

        # Always have mock as fallback
        self.providers["mock"] = TavilySearchProvider(api_key=None)

    async def search(
        self, query: str, num_results: int = 5, provider: str | None = None, **kwargs: Any
    ) -> SearchResponse:
        """Execute search with automatic provider selection.

        Args:
            query: Search query
            num_results: Number of results
            provider: Specific provider or None for auto
            **kwargs: Provider-specific parameters

        Returns:
            Search response
        """
        # Select provider
        if provider and provider in self.providers:
            search_provider = self.providers[provider]
        elif self.providers:
            # Use first available non-mock provider
            for name, prov in self.providers.items():
                if name != "mock":
                    search_provider = prov
                    break
            else:
                search_provider = self.providers.get("mock")
        else:
            # No providers available
            raise ValueError("No search providers configured")

        # Execute search
        return await search_provider.search(query, num_results, **kwargs)

    async def parallel_search(
        self, queries: list[str], num_results: int = 5, **kwargs: Any
    ) -> list[SearchResponse]:
        """Execute multiple searches in parallel.

        Args:
            queries: List of search queries
            num_results: Results per query
            **kwargs: Provider parameters

        Returns:
            List of search responses
        """
        tasks = [self.search(query, num_results, **kwargs) for query in queries]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        valid_results: list[SearchResponse] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logfire.error(f"Search failed for query '{queries[i]}': {str(result)}")
                # Add empty result
                valid_results.append(
                    SearchResponse(query=queries[i], results=[], total_results=0, source="error")
                )
            else:
                valid_results.append(cast(SearchResponse, result))

        return valid_results


# Global search service instance
search_service = WebSearchService()
