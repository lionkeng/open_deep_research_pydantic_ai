"""Pydantic-AI compliant dependency injection system.

This module defines the dependency containers following Pydantic-AI best practices
for type-safe dependency injection across all research agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import logfire

from .models.api_models import APIKeys
from .models.core import ResearchState
from .models.metadata import ResearchMetadata


@dataclass
class ResearchDependencies:
    """Main dependency container for all research agents.

    Following Pydantic-AI best practices, this dataclass contains all external
    dependencies needed by agents including HTTP clients, API keys, and state.
    """

    http_client: httpx.AsyncClient
    api_keys: APIKeys
    research_state: ResearchState
    metadata: ResearchMetadata | None = None

    # Configuration options
    max_concurrent_requests: int = 5
    request_timeout: float = 30.0
    retry_attempts: int = 3

    async def fetch_external_data(self, url: str, **kwargs: Any) -> str:
        """Fetch data from external URL with proper error handling."""
        try:
            response = await self.http_client.get(url, timeout=self.request_timeout, **kwargs)
            response.raise_for_status()
            return response.text
        except httpx.TimeoutException as e:
            logfire.warning(f"Request timeout for {url}: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logfire.error(f"HTTP error for {url}: {e.response.status_code}")
            raise

    async def post_external_data(self, url: str, data: Any, **kwargs: Any) -> str:
        """Post data to external URL with proper error handling."""
        try:
            response = await self.http_client.post(
                url, json=data, timeout=self.request_timeout, **kwargs
            )
            response.raise_for_status()
            return response.text
        except httpx.TimeoutException as e:
            logfire.warning(f"Request timeout for {url}: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logfire.error(f"HTTP error for {url}: {e.response.status_code}")
            raise

    def update_research_state(self, **updates: Any) -> None:
        """Update research state with new data."""
        for key, value in updates.items():
            if hasattr(self.research_state, key):
                setattr(self.research_state, key, value)
            else:
                logfire.warning(f"Unknown research state field: {key}")

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to research state."""
        if self.research_state.metadata is None:
            from .models.metadata import ResearchMetadata

            self.research_state.metadata = ResearchMetadata()
        # Use setattr for Pydantic models with extra="allow"
        setattr(self.research_state.metadata, key, value)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from research state."""
        if self.research_state.metadata is None:
            return default
        # Use getattr for Pydantic models
        return getattr(self.research_state.metadata, key, default)


@dataclass
class ClarificationDependencies:
    """Specialized dependencies for clarification agents."""

    research_deps: ResearchDependencies

    # Clarification-specific config
    breadth_threshold: float = 0.6
    confidence_threshold: float = 0.7
    max_clarification_questions: int = 3

    async def validate_query_external(self, query: str) -> dict[str, Any]:
        """Validate query against external service if needed."""
        # This would call external validation services
        # For now, return mock validation
        return {"is_valid": True, "complexity_score": 0.5, "domain": "general"}


@dataclass
class TransformationDependencies:
    """Specialized dependencies for query transformation agents."""

    research_deps: ResearchDependencies

    # Transformation-specific config
    min_specificity_improvement: float = 0.2
    max_transformation_attempts: int = 3

    async def enhance_query_context(self, query: str) -> dict[str, Any]:
        """Enhance query with additional context."""
        # This would call external context enhancement services
        return {
            "enhanced_context": f"Enhanced context for: {query}",
            "related_topics": ["topic1", "topic2"],
            "suggested_filters": ["filter1", "filter2"],
        }


@dataclass
class BriefDependencies:
    """Specialized dependencies for brief generation agents."""

    research_deps: ResearchDependencies

    # Brief-specific config
    min_brief_length: int = 100
    target_brief_length: int = 500
    max_brief_length: int = 1000

    async def fetch_domain_knowledge(self, domain: str) -> dict[str, Any]:
        """Fetch domain-specific knowledge for brief generation."""
        # This would call knowledge base services
        return {
            "domain_info": f"Knowledge for domain: {domain}",
            "key_concepts": ["concept1", "concept2"],
            "common_patterns": ["pattern1", "pattern2"],
        }
