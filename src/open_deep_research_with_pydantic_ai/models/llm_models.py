"""Pydantic models for LLM configuration and settings."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMProvider(str, Enum):
    """Available LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROQ = "groq"
    MISTRAL = "mistral"


class OpenAIModel(str, Enum):
    """Available OpenAI models."""

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_35_TURBO = "gpt-3.5-turbo"


class AnthropicModel(str, Enum):
    """Available Anthropic models."""

    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20241022"


class ModelConfig(BaseModel):
    """Configuration for an LLM model."""

    model_config = ConfigDict(extra="forbid")

    provider: LLMProvider = Field(description="Model provider")
    model_name: str = Field(description="Model identifier")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(
        default=None, gt=0, le=128000, description="Maximum output tokens"
    )
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str, info) -> str:
        """Validate model name format."""
        if ":" in v:
            # Format: provider:model
            provider, model = v.split(":", 1)
            return model
        return v

    def to_pydantic_ai_format(self) -> str:
        """Convert to Pydantic AI model string format."""
        return f"{self.provider}:{self.model_name}"


class ModelCapabilities(BaseModel):
    """Model capabilities and limits."""

    supports_streaming: bool = Field(default=True, description="Supports streaming responses")
    supports_functions: bool = Field(default=True, description="Supports function calling")
    supports_vision: bool = Field(default=False, description="Supports image inputs")
    context_window: int = Field(gt=0, description="Maximum context window size")
    max_output_tokens: int = Field(gt=0, description="Maximum output token limit")
    cost_per_1k_input_tokens: float = Field(ge=0, description="Cost per 1000 input tokens in USD")
    cost_per_1k_output_tokens: float = Field(ge=0, description="Cost per 1000 output tokens in USD")


class ModelRegistry(BaseModel):
    """Registry of available models with their configurations."""

    models: dict[str, ModelConfig] = Field(
        default_factory=dict, description="Available model configurations"
    )
    capabilities: dict[str, ModelCapabilities] = Field(
        default_factory=dict, description="Model capabilities"
    )
    default_model: str = Field(default="openai:gpt-4o", description="Default model to use")

    def get_model_config(self, model_id: str | None = None) -> ModelConfig:
        """Get configuration for a specific model.

        Args:
            model_id: Model identifier or None for default

        Returns:
            Model configuration

        Raises:
            KeyError: If model not found
        """
        model_id = model_id or self.default_model

        if model_id not in self.models:
            # Try to create a default config
            if ":" in model_id:
                provider_str, model_name = model_id.split(":", 1)
                try:
                    provider = LLMProvider(provider_str)
                    return ModelConfig(provider=provider, model_name=model_name)
                except ValueError as e:
                    raise KeyError(f"Unknown model: {model_id}") from e
            raise KeyError(f"Unknown model: {model_id}")

        return self.models[model_id]

    def get_capabilities(self, model_id: str) -> ModelCapabilities | None:
        """Get capabilities for a model.

        Args:
            model_id: Model identifier

        Returns:
            Model capabilities or None if not defined
        """
        return self.capabilities.get(model_id)


class LLMResponse(BaseModel):
    """Structured LLM response."""

    content: str = Field(description="Response content")
    model: str = Field(description="Model used")
    usage: dict[str, int] = Field(description="Token usage statistics")
    finish_reason: Literal["stop", "length", "function_call", "error"] = Field(
        description="Completion reason"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Pre-configured model registry with common models
DEFAULT_MODEL_REGISTRY = ModelRegistry(
    models={
        "openai:gpt-4o": ModelConfig(
            provider=LLMProvider.OPENAI, model_name="gpt-4o", temperature=0.7
        ),
        "openai:gpt-4o-mini": ModelConfig(
            provider=LLMProvider.OPENAI, model_name="gpt-4o-mini", temperature=0.7
        ),
        "anthropic:claude-3-5-sonnet": ModelConfig(
            provider=LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-20241022", temperature=0.7
        ),
    },
    capabilities={
        "openai:gpt-4o": ModelCapabilities(
            supports_streaming=True,
            supports_functions=True,
            supports_vision=True,
            context_window=128000,
            max_output_tokens=4096,
            cost_per_1k_input_tokens=0.005,
            cost_per_1k_output_tokens=0.015,
        ),
        "openai:gpt-4o-mini": ModelCapabilities(
            supports_streaming=True,
            supports_functions=True,
            supports_vision=True,
            context_window=128000,
            max_output_tokens=16384,
            cost_per_1k_input_tokens=0.00015,
            cost_per_1k_output_tokens=0.0006,
        ),
        "anthropic:claude-3-5-sonnet": ModelCapabilities(
            supports_streaming=True,
            supports_functions=True,
            supports_vision=True,
            context_window=200000,
            max_output_tokens=8192,
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
        ),
    },
    default_model="openai:gpt-4o",
)
