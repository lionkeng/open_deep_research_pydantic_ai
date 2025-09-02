"""Configuration management for the research system."""

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_ai.models import KnownModelName

from ..models.api_models import APIKeys

# Load .env file if it exists
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed


def _env_secret(name: str) -> SecretStr | None:
    """Get environment variable as SecretStr, returning None if empty or unset."""
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    return SecretStr(v)


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
        default="openai:gpt-5", description="Default LLM model to use"
    )

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retries for LLM calls")

    # Clarification settings (from implementation plan)
    research_interactive: bool = Field(
        default_factory=lambda: os.getenv("RESEARCH_INTERACTIVE", "true").lower() == "true",
        description="Whether to enable interactive clarification in CLI and HTTP modes",
    )

    max_clarification_questions: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CLARIFICATION_QUESTIONS", "2")),
        ge=0,
        le=5,
        description="Maximum number of clarifying questions to ask (0-2 recommended)",
    )

    research_brief_confidence_threshold: float = Field(
        default_factory=lambda: float(os.getenv("RESEARCH_BRIEF_CONFIDENCE_THRESHOLD", "0.7")),
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to proceed without clarification",
    )

    clarification_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("CLARIFICATION_TIMEOUT_SECONDS", "300")),
        ge=30,
        le=1800,
        description="Timeout for interactive clarification in seconds",
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
