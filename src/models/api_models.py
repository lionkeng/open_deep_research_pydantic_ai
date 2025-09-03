"""Pydantic models for API requests, responses, and configuration."""

from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationInfo, field_validator

from .clarification import ClarificationRequest, ClarificationResponse


class APIKeys(BaseModel):
    """API keys for various services with secure handling.

    Uses SecretStr to prevent accidental logging of sensitive keys.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    openai: SecretStr | None = Field(default=None, description="OpenAI API key")
    anthropic: SecretStr | None = Field(default=None, description="Anthropic API key")
    tavily: SecretStr | None = Field(default=None, description="Tavily search API key")

    @field_validator("openai", "anthropic", "tavily", mode="before")
    @classmethod
    def validate_api_key_format(
        cls, v: str | SecretStr | None, info: ValidationInfo
    ) -> SecretStr | None:
        """Validate API key format and convert to SecretStr."""
        if v is None:
            return None

        # If already SecretStr, get the secret value for validation
        if isinstance(v, SecretStr):
            key_str = v.get_secret_value()
        else:
            key_str = str(v).strip()

        # Skip validation for empty strings
        if not key_str:
            return None

        # Validate format based on field name (only for non-empty keys)
        # Allow flexible formats for development and different providers
        field_name = info.field_name
        if field_name == "openai" and not key_str.startswith(("sk-", "test-", "demo-")):
            raise ValueError("Invalid OpenAI API key format")
        elif field_name == "anthropic" and not key_str.startswith(
            ("anthropic-", "sk-", "test-", "demo-")
        ):
            # Allow sk- prefix for compatibility with DeepSeek and other providers
            raise ValueError("Invalid Anthropic API key format")
        elif field_name == "tavily" and not key_str.startswith(("tvly-", "test-", "demo-")):
            raise ValueError("Invalid Tavily API key format")

        # Return as SecretStr
        return SecretStr(key_str) if not isinstance(v, SecretStr) else v

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary with revealed secrets for internal use.

        WARNING: Only use this method when passing keys to services.
        Never log or expose the result of this method.
        """
        result = {}
        if self.openai:
            result["openai"] = self.openai.get_secret_value()
        if self.anthropic:
            result["anthropic"] = self.anthropic.get_secret_value()
        if self.tavily:
            result["tavily"] = self.tavily.get_secret_value()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "APIKeys":
        """Create from dictionary of plain strings."""
        from pydantic import SecretStr

        return cls(
            openai=SecretStr(data["openai"]) if data.get("openai") else None,
            anthropic=SecretStr(data["anthropic"]) if data.get("anthropic") else None,
            tavily=SecretStr(data["tavily"]) if data.get("tavily") else None,
        )


class ConversationMessage(TypedDict):
    """Typed structure for conversation messages between user and assistant.

    Used to track the dialogue history during the research workflow,
    particularly for clarification exchanges.
    """

    role: Literal["user", "assistant", "system"]
    content: str


class ResearchMetadata(BaseModel):
    """Strongly typed metadata for research workflow.

    This consolidates all metadata fields used throughout the research workflow,
    providing type safety and validation for all metadata operations.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Conversation and clarification fields
    conversation_messages: list[ConversationMessage] = Field(
        default_factory=list, description="Conversation history between user and system"
    )
    clarification_assessment: dict[str, Any] | None = Field(
        default=None, description="Clarification agent's assessment results"
    )
    clarification_request: ClarificationRequest | None = Field(
        default=None, description="Current clarification request with questions"
    )
    clarification_response: ClarificationResponse | None = Field(
        default=None, description="User's responses to clarification questions"
    )
    clarification_question: str | None = Field(
        default=None, description="Current clarification question awaiting response (formatted for display)"
    )
    awaiting_clarification: bool = Field(
        default=False, description="Whether system is waiting for clarification response"
    )

    # Query transformation fields
    transformed_query: dict[str, Any] | None = Field(
        default=None, description="Query transformation results"
    )

    # Research brief fields
    research_brief_text: str | None = Field(
        default=None, description="Generated research brief text"
    )
    research_brief_full: dict[str, Any] | None = Field(
        default=None, description="Complete research brief with all metadata"
    )
    research_brief_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score for research brief"
    )

    # Compression and findings fields
    compressed_findings_summary: dict[str, Any] | None = Field(
        default=None, description="Summary of compressed research findings"
    )
    compressed_findings_full: Any | None = Field(
        default=None, description="Full compressed findings object"
    )

    # Legacy fields (kept for compatibility)
    clarifying_questions: list[str] = Field(
        default_factory=list, description="Questions that need clarification"
    )
    search_queries: list[str] = Field(default_factory=list, description="Search queries executed")
    sources_consulted: int = Field(default=0, description="Number of sources consulted")
    total_tokens_used: int = Field(default=0, description="Total LLM tokens consumed")
    processing_time: float = Field(default=0.0, description="Total processing time in seconds")
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Research confidence score"
    )
    tags: list[str] = Field(default_factory=list, description="Research topic tags")


class APIHealthResponse(BaseModel):
    """Health check response model."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(description="Service health status")
    version: str = Field(description="API version")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Health check timestamp"
    )
    checks: dict[str, bool] = Field(
        default_factory=dict, description="Individual component health checks"
    )


class APIRootResponse(BaseModel):
    """Root endpoint response with API information."""

    name: str = Field(default="Deep Research API", description="API name")
    version: str = Field(default="1.0.0", description="API version")
    description: str = Field(
        default="AI-powered research system with streaming support", description="API description"
    )
    endpoints: dict[str, str] = Field(description="Available endpoints and descriptions")
    documentation_url: str = Field(default="/docs", description="API documentation URL")


class ResearchRequest(BaseModel):
    """Research request with typed fields."""

    query: str = Field(min_length=1, max_length=5000, description="Research query")
    api_keys: APIKeys | None = Field(default=None, description="Optional API keys")
    stream: bool = Field(default=True, description="Enable streaming updates")
    max_search_results: int = Field(
        default=5, ge=1, le=50, description="Maximum search results per query"
    )
    model: str | None = Field(default=None, description="Optional model override")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and clean query."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v


class ResearchResponse(BaseModel):
    """Research initiation response."""

    request_id: str = Field(description="Unique request identifier")
    status: Literal["accepted", "rejected"] = Field(description="Request acceptance status")
    message: str = Field(description="Status message")
    stream_url: str | None = Field(default=None, description="SSE stream endpoint URL")
    report_url: str | None = Field(default=None, description="Final report endpoint URL")


class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: dict[str, Any] | None = Field(default=None, description="Additional error details")
    request_id: str | None = Field(default=None, description="Request ID if available")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Error timestamp"
    )
