"""Configuration management for the research system."""

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.models import KnownModelName

from open_deep_research_with_pydantic_ai.models.api_models import APIKeys

# Load .env file if it exists
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed


class APIConfig(BaseModel):
    """API configuration with validation."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    # Use APIKeys model for secure key handling
    api_keys: APIKeys = Field(
        default_factory=lambda: APIKeys(
            openai=os.getenv("OPENAI_API_KEY"),
            anthropic=os.getenv("ANTHROPIC_API_KEY"),
            tavily=os.getenv("TAVILY_API_KEY"),
        ),
        description="API keys for various services",
    )

    default_model: KnownModelName = Field(
        default="openai:gpt-4o", description="Default LLM model to use"
    )

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retries for LLM calls")

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
