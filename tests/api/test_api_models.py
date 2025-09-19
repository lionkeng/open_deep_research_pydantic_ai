"""Tests for API models."""


import pytest
from pydantic import ValidationError

from models.api_models import APIKeys, ConversationMessage


def test_api_keys_validation():
    """Test APIKeys model validation."""
    # Valid API keys
    keys = APIKeys(
        openai="sk-test123",
        anthropic="anthropic-test123",
        tavily="tvly-test123"
    )
    assert keys.openai is not None
    assert keys.anthropic is not None
    assert keys.tavily is not None

    # Test with None values
    keys = APIKeys()
    assert keys.openai is None
    assert keys.anthropic is None
    assert keys.tavily is None

    # Test to_dict method
    keys = APIKeys(openai="sk-test")
    keys_dict = keys.to_dict()
    assert keys_dict["openai"] == "sk-test"


def test_api_keys_invalid_format():
    """Test APIKeys validation with invalid formats."""
    # Invalid OpenAI key format
    with pytest.raises(ValidationError) as exc_info:
        APIKeys(openai="invalid-key")
    assert "Invalid OpenAI API key format" in str(exc_info.value)

    # Invalid Anthropic key format
    with pytest.raises(ValidationError) as exc_info:
        APIKeys(anthropic="invalid-key")
    assert "Invalid Anthropic API key format" in str(exc_info.value)

    # Invalid Tavily key format
    with pytest.raises(ValidationError) as exc_info:
        APIKeys(tavily="invalid-key")
    assert "Invalid Tavily API key format" in str(exc_info.value)


def test_conversation_message():
    """Test ConversationMessage model."""
    # User message
    msg = ConversationMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"

    # Assistant message
    msg = ConversationMessage(role="assistant", content="Hi there")
    assert msg.role == "assistant"
    assert msg.content == "Hi there"

    # System message
    msg = ConversationMessage(role="system", content="You are helpful")
    assert msg.role == "system"
    assert msg.content == "You are helpful"


def test_conversation_message_validation():
    """Test ConversationMessage validation."""
    # Invalid role
    with pytest.raises(ValidationError):
        ConversationMessage(role="invalid", content="Test")

    # Missing content
    with pytest.raises(ValidationError):
        ConversationMessage(role="user")

    # Missing role
    with pytest.raises(ValidationError):
        ConversationMessage(content="Test")
