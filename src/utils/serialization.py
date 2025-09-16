"""Serialization utilities for clarification models."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID


class ClarificationJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for clarification models.

    Handles special types like UUID and datetime that aren't natively
    JSON serializable.
    """

    def default(self, o: Any) -> Any:
        """Convert non-JSON-serializable objects.

        Args:
            o: Object to encode

        Returns:
            JSON-serializable representation
        """
        if isinstance(o, UUID):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "model_dump"):
            # Handle Pydantic models
            return o.model_dump()
        if hasattr(o, "__dict__"):
            # Handle regular classes
            return o.__dict__

        return super().default(o)


def serialize_clarification_request(request: Any) -> str:
    """Serialize a ClarificationRequest to JSON.

    Args:
        request: ClarificationRequest object

    Returns:
        JSON string representation
    """
    return json.dumps(
        request.model_dump() if hasattr(request, "model_dump") else request,
        cls=ClarificationJSONEncoder,
        indent=2,
    )


def serialize_clarification_response(response: Any) -> str:
    """Serialize a ClarificationResponse to JSON.

    Args:
        response: ClarificationResponse object

    Returns:
        JSON string representation
    """
    return json.dumps(
        response.model_dump() if hasattr(response, "model_dump") else response,
        cls=ClarificationJSONEncoder,
        indent=2,
    )


def deserialize_clarification_request(json_str: str) -> dict[str, Any]:
    """Deserialize a ClarificationRequest from JSON.

    Args:
        json_str: JSON string

    Returns:
        Dictionary that can be used to create ClarificationRequest
    """
    return json.loads(json_str)


def deserialize_clarification_response(json_str: str) -> dict[str, Any]:
    """Deserialize a ClarificationResponse from JSON.

    Args:
        json_str: JSON string

    Returns:
        Dictionary that can be used to create ClarificationResponse
    """
    return json.loads(json_str)


def format_clarification_for_display(request: Any) -> str:
    """Format a ClarificationRequest for human-readable display.

    Args:
        request: ClarificationRequest object

    Returns:
        Formatted string for display
    """
    if not hasattr(request, "questions"):
        return str(request)

    lines = ["Clarification Questions:"]
    for idx, question in enumerate(request.get_sorted_questions(), 1):
        required = "[Required]" if question.is_required else "[Optional]"
        lines.append(f"{idx}. {required} {question.question}")

        if question.context:
            lines.append(f"   Context: {question.context}")

        if question.choices:
            lines.append("   Options:")
            for choice in question.choices:
                lines.append(f"   - {choice}")

    return "\n".join(lines)


def format_response_for_display(response: Any, request: Any) -> str:
    """Format a ClarificationResponse for human-readable display.

    Args:
        response: ClarificationResponse object
        request: Original ClarificationRequest for context

    Returns:
        Formatted string for display
    """
    if not hasattr(response, "answers"):
        return str(response)

    lines = ["Clarification Responses:"]

    for answer in response.answers:
        question = (
            request.get_question_by_id(answer.question_id)
            if hasattr(request, "get_question_by_id")
            else None
        )

        if question:
            if answer.skipped:
                lines.append(f"- {question.question}: [Skipped]")
            else:
                lines.append(f"- {question.question}: {answer.answer}")
        else:
            if answer.skipped:
                lines.append(f"- Question {answer.question_id}: [Skipped]")
            else:
                lines.append(f"- Question {answer.question_id}: {answer.answer}")

    return "\n".join(lines)
