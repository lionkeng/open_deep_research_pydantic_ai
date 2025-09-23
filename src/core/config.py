"""Configuration management for the research system."""

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_ai.models import KnownModelName

from models.api_models import APIKeys

# Note: .env file is loaded in src/__init__.py before this module is imported


def _env_secret(name: str) -> SecretStr | None:
    """Get environment variable as SecretStr, returning None if empty or unset."""
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    return SecretStr(v)


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean flag from environment variables.

    Accepts: "1", "true", "TRUE", "True" as True.
    """
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip() in {"1", "true", "TRUE", "True"}


def _env_float_default(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    try:
        return float(v)
    except Exception:
        return default


class APIConfig(BaseModel):
    """API configuration with validation."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    # Use APIKeys model for secure key handling
    api_keys: APIKeys = Field(
        default_factory=lambda: APIKeys(
            openai=_env_secret("OPENAI_API_KEY"),
            anthropic=_env_secret("ANTHROPIC_API_KEY"),
            tavily=_env_secret("TAVILY_API_KEY"),
        ),
        description="API keys for various services",
    )

    default_model: KnownModelName = Field(
        default="openai:gpt-5-mini", description="Default LLM model to use"
    )

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retries for LLM calls")

    # Synthesis feature flags (read once at startup)
    enable_embedding_similarity: bool = Field(
        default_factory=lambda: _env_flag("ENABLE_EMBEDDING_SIMILARITY", False),
        description="Enable embedding-based semantic similarity during synthesis",
    )
    embedding_similarity_threshold: float = Field(
        default_factory=lambda: _env_float_default("EMBEDDING_SIMILARITY_THRESHOLD", 0.55),
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for grouping when embeddings are enabled",
    )
    enable_llm_clean_merge: bool = Field(
        default_factory=lambda: _env_flag("ENABLE_LLM_CLEAN_MERGE", False),
        description="Enable guardrailed LLM clean-merge for report text",
    )

    # Backward compatibility properties
    @property
    def openai_api_key(self) -> str | None:
        """Get OpenAI API key for backward compatibility."""
        return self.api_keys.openai.get_secret_value() if self.api_keys.openai else None

    @property
    def anthropic_api_key(self) -> str | None:
        """Get Anthropic API key for backward compatibility."""
        return self.api_keys.anthropic.get_secret_value() if self.api_keys.anthropic else None

    @property
    def tavily_api_key(self) -> str | None:
        """Get Tavily API key for backward compatibility."""
        return self.api_keys.tavily.get_secret_value() if self.api_keys.tavily else None

    def get_model_config(self, model_name: str | None = None) -> dict[str, Any]:
        """Get configuration for a specific model.

        Args:
            model_name: Model name or None for default

        Returns:
            Model configuration dict
        """
        model = model_name or self.default_model

        # Extract provider from model name
        if ":" in model:
            provider = model.split(":")[0]
        else:
            provider = "openai"  # Default

        config = {
            "model": model,
            "retries": self.max_retries,
        }

        # Add appropriate API key
        if provider == "openai" and self.api_keys.openai:
            config["api_key"] = self.api_keys.openai.get_secret_value()
        elif provider == "anthropic" and self.api_keys.anthropic:
            config["api_key"] = self.api_keys.anthropic.get_secret_value()

        return config


# Global configuration instance
config = APIConfig()
