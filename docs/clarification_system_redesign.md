# Clarification System Redesign: Multi-Question Architecture

## Executive Summary

This document outlines the complete redesign of the clarification system to properly handle multiple questions with unique identifiers, replacing the current single-string approach. The new design emphasizes type safety, clear data flow, and Pythonic patterns.

**Key Updates Based on Architectural Review:**

- Fixed critical Pydantic model validation issues
- Added performance optimizations for O(1) lookups
- Simplified architecture by leveraging existing ResearchState management
- Documented future work for conditional questions and security

## 1. Data Model Design

### 1.1 Core Models

```python
# src/models/clarification.py

from typing import Optional, Literal, Self
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import BaseModel, Field, model_validator, PrivateAttr


class ClarificationQuestion(BaseModel):
    """Individual clarification question with metadata."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    is_required: bool = True
    question_type: Literal["text", "choice", "multi_choice"] = "text"
    choices: Optional[list[str]] = None
    context: Optional[str] = None
    order: int = 0

    class Config:
        frozen = True


class ClarificationAnswer(BaseModel):
    """Answer to a clarification question."""

    question_id: str
    answer: Optional[str] = None
    skipped: bool = False
    answered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode='after')
    def validate_answer_consistency(self) -> Self:
        """Validate that answer and skipped states are consistent."""
        if not self.skipped and self.answer is None:
            raise ValueError("Answer must be provided if not skipped")
        if self.skipped and self.answer is not None:
            raise ValueError("Cannot have both answer and skipped=True")
        return self


class ClarificationRequest(BaseModel):
    """Collection of clarification questions to ask the user."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    questions: list[ClarificationQuestion]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    context: Optional[str] = None

    # Private attribute for O(1) lookups
    _question_index: dict[str, ClarificationQuestion] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        """Build index after model initialization."""
        self._question_index = {q.id: q for q in self.questions}

    def get_question_by_id(self, question_id: str) -> Optional[ClarificationQuestion]:
        """Retrieve a question by its ID in O(1) time."""
        return self._question_index.get(question_id)

    def get_required_questions(self) -> list[ClarificationQuestion]:
        """Get only required questions."""
        return [q for q in self.questions if q.is_required]

    def get_sorted_questions(self) -> list[ClarificationQuestion]:
        """Get questions sorted by order."""
        return sorted(self.questions, key=lambda q: q.order)


class ClarificationResponse(BaseModel):
    """Complete response containing all answers."""

    request_id: str
    answers: list[ClarificationAnswer]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Private attribute for O(1) lookups
    _answer_index: dict[str, ClarificationAnswer] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        """Build index after model initialization."""
        self._answer_index = {a.question_id: a for a in self.answers}

    def get_answer_for_question(self, question_id: str) -> Optional[ClarificationAnswer]:
        """Get answer for a specific question ID in O(1) time."""
        return self._answer_index.get(question_id)

    def validate_against_request(self, request: ClarificationRequest) -> list[str]:
        """Validate this response against the original request."""
        errors = []

        # Check all required questions are answered
        for question in request.get_required_questions():
            answer = self.get_answer_for_question(question.id)
            if not answer or (answer.skipped and question.is_required):
                errors.append(f"Required question '{question.id}' not answered")

        # Check no unknown question IDs
        valid_ids = {q.id for q in request.questions}
        for answer in self.answers:
            if answer.question_id not in valid_ids:
                errors.append(f"Unknown question ID: {answer.question_id}")

        return errors


```

### 1.2 Agent Output Model

```python
# src/agents/clarification.py (updated model)

from models.clarification import ClarificationRequest, ClarificationQuestion


class ClarifyWithUser(BaseModel):
    """Agent output containing clarification questions."""

    needs_clarification: bool = Field(
        description="Whether clarification is needed from the user"
    )
    request: Optional[ClarificationRequest] = Field(
        default=None,
        description="Clarification request with questions if needed"
    )
    reasoning: str = Field(
        description="Explanation of why clarification is or isn't needed"
    )

    @model_validator(mode='after')
    def validate_request_consistency(self) -> Self:
        """Ensure request presence matches needs_clarification."""
        if self.needs_clarification and not self.request:
            raise ValueError("Request must be provided when needs_clarification is True")
        if not self.needs_clarification and self.request:
            raise ValueError("Request should be None when needs_clarification is False")
        return self
```

## 2. Implementation Steps (Ordered by Dependency)

### Phase 1: Core Models (Foundation)

1. Create `src/models/clarification.py` with new data models using proper validators
2. Update imports across the codebase

### Phase 2: Agent Updates

1. Update `ClarifyWithUser` class in `src/agents/clarification.py`
2. Modify `ClarificationAgent` to generate `ClarificationRequest` objects
3. Update agent prompts to understand the new structure

### Phase 3: Workflow Integration

1. Update `src/core/workflow.py` to use new clarification models
2. Integrate clarification models with ResearchState.metadata
3. Modify clarification flow to use new models
4. Update query transformation logic

### Phase 4: Interface Updates

1. Update CLI interface (`src/interfaces/cli_clarification.py`)
2. Update API endpoints (`src/api/main.py`)
3. Ensure proper serialization/deserialization

### Phase 5: Test Updates

1. Update unit tests for new models
2. Update integration tests
3. Update evaluation scripts

### Phase 6: Cleanup

1. Remove old code and unused imports
2. Update documentation
3. Run full test suite and fix any issues

## 3. Code Changes Required

### 3.1 src/agents/clarification.py

```python
from typing import Optional, Self
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from models.clarification import (
    ClarificationRequest,
    ClarificationQuestion,
    ClarificationResponse
)


class ClarifyWithUser(BaseModel):
    """Agent output for clarification needs."""

    needs_clarification: bool = Field(
        description="Whether clarification is needed"
    )
    request: Optional[ClarificationRequest] = Field(
        default=None,
        description="Structured clarification request"
    )
    reasoning: str = Field(
        description="Explanation of the decision"
    )

    @model_validator(mode='after')
    def validate_request_consistency(self) -> Self:
        """Ensure request presence matches needs_clarification."""
        if self.needs_clarification and not self.request:
            raise ValueError("Request must be provided when needs_clarification is True")
        if not self.needs_clarification and self.request:
            raise ValueError("Request should be None when needs_clarification is False")
        return self


class ClarificationAgent:
    """Agent for generating clarification questions."""

    def __init__(self, model_name: str = "openai:gpt-5"):
        self.agent = Agent(
            model_name,
            result_type=ClarifyWithUser,
            system_prompt=self._get_system_prompt()
        )

    def _get_system_prompt(self) -> str:
        return """You are a research clarification specialist.

        When analyzing queries, determine if clarification would improve results.

        If clarification is needed, generate structured questions:
        - Each question should have a unique purpose
        - Mark questions as required or optional appropriately
        - Order questions logically (most important first)
        - Provide context when helpful

        Question types:
        - text: Open-ended text response
        - choice: Single selection from options
        - multi_choice: Multiple selections allowed
        """

    async def analyze_query(self, query: str) -> ClarifyWithUser:
        """Analyze if a query needs clarification."""
        result = await self.agent.run(query)
        return result.data
```

### 3.2 src/core/workflow.py

```python
from datetime import datetime, timezone
from models.clarification import (
    ClarificationRequest,
    ClarificationResponse
)
from agents.clarification import ClarificationAgent
from interfaces.cli_clarification import CLIClarificationInterface
from models.core import ResearchState


class ResearchWorkflow:
    """Main workflow orchestrator."""

    def __init__(self):
        self.clarification_agent = ClarificationAgent()
        self.cli_interface = CLIClarificationInterface()

    async def process_with_clarification(
        self,
        state: ResearchState,
        query: str
    ) -> tuple[ResearchState, str]:
        """Process query with clarification if needed.

        Returns:
            Updated state and transformed query
        """

        # Check if clarification needed
        clarify_result = await self.clarification_agent.analyze_query(query)

        if not clarify_result.needs_clarification:
            return state, query  # Use original query

        # Store clarification request in state metadata
        state.metadata.clarification_request = clarify_result.request
        state.metadata.awaiting_clarification = True
        state.metadata.clarification_question = self._format_questions_for_display(
            clarify_result.request
        )

        # Get user responses
        response = await self.cli_interface.ask_questions(clarify_result.request)

        # Validate response
        errors = response.validate_against_request(clarify_result.request)
        if errors:
            raise ValueError(f"Invalid response: {errors}")

        # Store response in state metadata
        state.metadata.clarification_response = response
        state.metadata.awaiting_clarification = False

        # Transform query with answers
        transformed = await self._transform_query(query, clarify_result.request, response)
        state.metadata.transformed_query = transformed

        return state, transformed

    def _format_questions_for_display(self, request: ClarificationRequest) -> str:
        """Format questions for UI display."""
        questions = request.get_sorted_questions()
        formatted = []
        for i, q in enumerate(questions, 1):
            formatted.append(f"{i}. {q.question}")
        return "\n".join(formatted)

    async def _transform_query(
        self,
        original_query: str,
        request: ClarificationRequest,
        response: ClarificationResponse
    ) -> str:
        """Transform query based on clarification answers."""
        # Implementation for query transformation
        parts = [original_query]

        for question in request.get_sorted_questions():
            answer = response.get_answer_for_question(question.id)
            if answer and not answer.skipped:
                parts.append(f"{question.question}: {answer.answer}")

        return " ".join(parts)
```

### 3.3 src/interfaces/cli_clarification.py

```python
from typing import Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from models.clarification import (
    ClarificationRequest,
    ClarificationResponse,
    ClarificationAnswer,
    ClarificationQuestion
)


class CLIClarificationInterface:
    """CLI interface for clarification questions."""

    def __init__(self):
        self.console = Console()

    async def ask_questions(self, request: ClarificationRequest) -> ClarificationResponse:
        """Present questions to user and collect answers."""

        self.console.print(Panel("Clarification needed for better results", style="yellow"))

        if request.context:
            self.console.print(f"Context: {request.context}\n")

        answers = []

        for question in request.get_sorted_questions():
            answer = await self._ask_single_question(question)
            answers.append(answer)

        return ClarificationResponse(
            request_id=request.id,
            answers=answers
        )

    async def _ask_single_question(self, question: ClarificationQuestion) -> ClarificationAnswer:
        """Ask a single question and get answer."""

        # Display question
        required_tag = "[red]*[/red]" if question.is_required else "[dim](optional)[/dim]"
        self.console.print(f"\n{required_tag} {question.question}")

        if question.context:
            self.console.print(f"  [dim]{question.context}[/dim]")

        # Handle different question types
        if question.question_type == "choice" and question.choices:
            answer = await self._ask_choice(question)
        elif question.question_type == "multi_choice" and question.choices:
            answer = await self._ask_multi_choice(question)
        else:
            answer = await self._ask_text(question)

        return answer

    async def _ask_text(self, question: ClarificationQuestion) -> ClarificationAnswer:
        """Ask open-ended text question."""

        if not question.is_required:
            skip = Confirm.ask("Skip this question?", default=False)
            if skip:
                return ClarificationAnswer(
                    question_id=question.id,
                    skipped=True
                )

        answer = Prompt.ask("Your answer")
        return ClarificationAnswer(
            question_id=question.id,
            answer=answer
        )

    async def _ask_choice(self, question: ClarificationQuestion) -> ClarificationAnswer:
        """Ask single-choice question."""

        # Create choice table
        table = Table(show_header=False)
        for i, choice in enumerate(question.choices, 1):
            table.add_row(f"[cyan]{i}[/cyan]", choice)

        self.console.print(table)

        if not question.is_required:
            self.console.print("[dim]0. Skip this question[/dim]")

        while True:
            choice_str = Prompt.ask("Select option")

            if choice_str == "0" and not question.is_required:
                return ClarificationAnswer(
                    question_id=question.id,
                    skipped=True
                )

            try:
                choice_idx = int(choice_str) - 1
                if 0 <= choice_idx < len(question.choices):
                    return ClarificationAnswer(
                        question_id=question.id,
                        answer=question.choices[choice_idx]
                    )
            except ValueError:
                pass

            self.console.print("[red]Invalid selection[/red]")
```

### 3.4 src/api/main.py

```python
# Updates to existing API endpoints in src/api/main.py
# These changes integrate with the existing ResearchState management

from models.clarification import (
    ClarificationRequest,
    ClarificationResponse
)

# The existing /research/{request_id}/clarification endpoint
# already handles clarification - we just need to update it
# to use the new ClarificationRequest and ClarificationResponse models

@app.post("/research/{request_id}/clarification")
async def respond_to_clarification(
    request_id: str,
    response: ClarificationResponse
):
    """Submit clarification answers.

    This endpoint already exists in the API and manages state
    through ResearchState. We update it to use the new models.
    """
    async with _sessions_lock:
        if request_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Research request not found")
        state = active_sessions[request_id]

    # Validate response against the stored request
    if state.metadata.clarification_request:
        errors = response.validate_against_request(state.metadata.clarification_request)
        if errors:
            raise HTTPException(status_code=400, detail={"errors": errors})

    # Store response in metadata
    state.metadata.clarification_response = response
    state.metadata.awaiting_clarification = False

    # Transform query
    workflow = ResearchWorkflow()
    transformed = await workflow._transform_query(
        state.user_query,
        state.metadata.clarification_request,
        response
    )
    state.metadata.transformed_query = transformed

    # Resume research with transformed query
    # ... existing resume logic ...
```

## 4. Test Updates Required

### 4.1 Unit Tests for Models

```python
# tests/unit/models/test_clarification.py

import pytest
from datetime import datetime, timezone
from models.clarification import (
    ClarificationQuestion,
    ClarificationAnswer,
    ClarificationRequest,
    ClarificationResponse,
)


class TestClarificationQuestion:
    def test_create_basic_question(self):
        q = ClarificationQuestion(
            question="What is your budget?",
            is_required=True
        )
        assert q.id is not None
        assert q.question_type == "text"
        assert q.choices is None

    def test_create_choice_question(self):
        q = ClarificationQuestion(
            question="Select priority",
            question_type="choice",
            choices=["Speed", "Accuracy", "Cost"]
        )
        assert len(q.choices) == 3
        assert q.question_type == "choice"


class TestClarificationAnswer:
    def test_valid_answer(self):
        a = ClarificationAnswer(
            question_id="test-id",
            answer="My answer"
        )
        assert not a.skipped
        assert a.answer == "My answer"
        assert a.answered_at.tzinfo == timezone.utc

    def test_skipped_answer(self):
        a = ClarificationAnswer(
            question_id="test-id",
            skipped=True
        )
        assert a.skipped
        assert a.answer is None

    def test_invalid_answer_combinations(self):
        # Can't have both answer and skipped
        with pytest.raises(ValueError, match="Cannot have both"):
            ClarificationAnswer(
                question_id="test-id",
                answer="Answer",
                skipped=True
            )

        # Must have either answer or skipped
        with pytest.raises(ValueError, match="Answer must be provided"):
            ClarificationAnswer(question_id="test-id")


class TestClarificationRequest:
    def test_create_request(self):
        questions = [
            ClarificationQuestion(question="Q1", order=1),
            ClarificationQuestion(question="Q2", order=0)
        ]
        req = ClarificationRequest(questions=questions)

        assert len(req.questions) == 2
        assert req.id is not None
        assert req.created_at.tzinfo == timezone.utc

        # Test sorting
        sorted_q = req.get_sorted_questions()
        assert sorted_q[0].question == "Q2"  # order=0
        assert sorted_q[1].question == "Q1"  # order=1

    def test_o1_lookup_performance(self):
        # Create request with many questions
        questions = [
            ClarificationQuestion(question=f"Q{i}", order=i)
            for i in range(100)
        ]
        req = ClarificationRequest(questions=questions)

        # Test O(1) lookup
        q50 = req.get_question_by_id(questions[50].id)
        assert q50 == questions[50]

    def test_get_required_questions(self):
        questions = [
            ClarificationQuestion(question="Q1", is_required=True),
            ClarificationQuestion(question="Q2", is_required=False),
            ClarificationQuestion(question="Q3", is_required=True)
        ]
        req = ClarificationRequest(questions=questions)

        required = req.get_required_questions()
        assert len(required) == 2
        assert all(q.is_required for q in required)


# Session-related tests removed as we're using ResearchState for tracking
```

### 4.2 Integration Tests

```python
# tests/integration/test_clarification_workflow.py

import pytest
from unittest.mock import AsyncMock, patch
from core.workflow import ResearchWorkflow
from models.clarification import (
    ClarificationRequest,
    ClarificationQuestion,
    ClarificationResponse,
    ClarificationAnswer
)


@pytest.mark.asyncio
async def test_workflow_with_clarification():
    workflow = ResearchWorkflow()

    # Create a test state
    state = ResearchState(
        request_id="test-request",
        user_id="test-user",
        user_query="Find me a car"
    )

    # Mock agent to return clarification needed
    mock_request = ClarificationRequest(
        questions=[
            ClarificationQuestion(
                question="What's your budget?",
                is_required=True
            )
        ]
    )

    with patch.object(workflow.clarification_agent, 'analyze_query') as mock_analyze:
        mock_analyze.return_value = AsyncMock(
            needs_clarification=True,
            request=mock_request
        )

        # Mock CLI interface
        mock_response = ClarificationResponse(
            request_id=mock_request.id,
            answers=[
                ClarificationAnswer(
                    question_id=mock_request.questions[0].id,
                    answer="$10,000"
                )
            ]
        )

        with patch.object(workflow.cli_interface, 'ask_questions') as mock_ask:
            mock_ask.return_value = mock_response

            updated_state, transformed_query = await workflow.process_with_clarification(
                state, "Find me a car"
            )

            assert "Find me a car" in transformed_query
            assert "$10,000" in transformed_query
            assert updated_state.metadata.clarification_request == mock_request
            assert updated_state.metadata.clarification_response == mock_response
            mock_analyze.assert_called_once()
            mock_ask.assert_called_once()
```

## 5. Migration Approach

Since backward compatibility is not required, we can perform a clean cutover:

### 5.1 Migration Steps

```python
# migration_script.py

"""
One-time migration script to update existing data.
Run once during deployment.
"""

import json
from pathlib import Path
from models.clarification import (
    ClarificationRequest,
    ClarificationQuestion
)


def migrate_old_clarifications():
    """Convert old format to new format."""

    # Example of old format conversion
    old_data = {
        "question": "1. What's your budget?\n2. What's the timeline?"
    }

    # Parse old format
    questions_text = old_data["question"].split("\n")
    questions = []

    for i, q_text in enumerate(questions_text):
        # Remove numbering if present
        q_text = q_text.strip()
        if q_text and q_text[0].isdigit():
            q_text = q_text[q_text.index(".") + 1:].strip()

        if q_text:
            questions.append(
                ClarificationQuestion(
                    question=q_text,
                    order=i,
                    is_required=True  # Assume all were required
                )
            )

    # Create new format
    new_request = ClarificationRequest(questions=questions)

    return new_request


# Run migration
if __name__ == "__main__":
    # Migration logic here
    pass
```

### 5.2 Deployment Checklist

1. **Pre-deployment**

   - [ ] All tests passing with new models
   - [ ] Code review completed
   - [ ] Performance benchmarks acceptable

2. **Deployment**

   - [ ] Deploy new code
   - [ ] Run migration script if needed
   - [ ] Verify API endpoints working
   - [ ] Test CLI interface

3. **Post-deployment**
   - [ ] Monitor error rates
   - [ ] Check performance metrics
   - [ ] Gather user feedback

## 6. Future Work

### 6.1 Conditional Question Support (Priority: HIGH)

**Problem:** Some questions become irrelevant based on previous answers. For example, if the user asks about "Python" and clarifies they mean the snake, all programming-related questions should be skipped.

**Solution:** Implement a Hybrid Graph-Based approach where:

- Backend maintains question dependency graph
- Frontend optimizes navigation for responsiveness
- Questions can have conditions based on previous answers

#### Proposed Architecture

```python
# Future implementation for conditional questions

from typing import Dict, Any, Optional, Literal
from enum import Enum


class ConditionOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"


class QuestionCondition(BaseModel):
    """Condition for showing/hiding a question."""

    depends_on_question: str  # Question ID
    operator: ConditionOperator
    value: Any

    def evaluate(self, answers: Dict[str, Any]) -> bool:
        """Check if condition is met."""
        if self.depends_on_question not in answers:
            return False

        answer = answers[self.depends_on_question]

        if self.operator == ConditionOperator.EQUALS:
            return answer == self.value
        elif self.operator == ConditionOperator.NOT_EQUALS:
            return answer != self.value
        elif self.operator == ConditionOperator.IN:
            return answer in self.value
        elif self.operator == ConditionOperator.NOT_IN:
            return answer not in self.value

        return False


class ConditionalQuestion(ClarificationQuestion):
    """Question with conditional display logic."""

    display_condition: Optional[QuestionCondition] = None
    skip_logic: Dict[str, str] = Field(
        default_factory=dict,
        description="Map answer values to next question ID"
    )


class QuestionGraph(BaseModel):
    """Graph structure for conditional question flows."""

    questions: list[ConditionalQuestion]
    entry_point: str
    terminal_nodes: list[str] = Field(default_factory=list)

    def get_next_question(
        self,
        current_id: str,
        answer: Any,
        all_answers: Dict[str, Any]
    ) -> Optional[ConditionalQuestion]:
        """Determine next question based on conditions."""
        current = self.get_question_by_id(current_id)
        if not current:
            return None

        # Check skip logic
        if answer in current.skip_logic:
            next_id = current.skip_logic[answer]
            if next_id == "END":
                return None
            return self.get_question_by_id(next_id)

        # Find next eligible question
        for question in sorted(self.questions, key=lambda q: q.order):
            if question.id == current_id:
                continue
            if question.id in all_answers:
                continue

            # Check display condition
            if question.display_condition:
                if not question.display_condition.evaluate(all_answers):
                    continue

            return question

        return None  # No more questions
```

#### Implementation Phases

1. **Phase 1**: Extend models with conditional fields
2. **Phase 2**: Implement graph traversal logic
3. **Phase 3**: Update agent to generate conditional questions
4. **Phase 4**: Add API support for progressive disclosure
5. **Phase 5**: Update CLI/Web interfaces

#### Benefits

- Reduces unnecessary questions
- Improves user experience
- Enables complex clarification flows
- Supports early termination when sufficient info gathered

### 6.2 Security Enhancements (Priority: MEDIUM)

**Current Gaps:** No input sanitization, rate limiting, or session authentication.

#### Planned Security Improvements

```python
# Future security implementations

from html import escape
import bleach
from fastapi import HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import hashlib
import hmac


# 1. Input Sanitization
class SecureClarificationAnswer(ClarificationAnswer):
    """Answer with input sanitization."""

    @field_validator('answer')
    @classmethod
    def sanitize_answer(cls, v: Optional[str]) -> Optional[str]:
        """Remove dangerous content from answers."""
        if v:
            # Remove HTML/JavaScript
            cleaned = bleach.clean(v, tags=[], strip=True)
            # Limit length
            if len(cleaned) > 1000:
                cleaned = cleaned[:1000]
            return cleaned
        return v


# 2. Rate Limiting
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.post(
    "/clarification/check",
    dependencies=[Depends(RateLimiter(times=10, seconds=60))]
)
async def check_clarification_rate_limited(query: str):
    """Rate-limited clarification check."""
    pass


# 3. Session Authentication
class SessionSecurity:
    """Secure session management."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()

    def generate_session_token(self, session_id: str) -> str:
        """Generate HMAC-signed session token."""
        signature = hmac.new(
            self.secret_key,
            session_id.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{session_id}.{signature}"

    def verify_session_token(self, token: str) -> Optional[str]:
        """Verify and extract session ID from token."""
        try:
            session_id, signature = token.rsplit(".", 1)
            expected_sig = hmac.new(
                self.secret_key,
                session_id.encode(),
                hashlib.sha256
            ).hexdigest()

            if hmac.compare_digest(signature, expected_sig):
                return session_id
        except ValueError:
            pass

        return None


# 4. Audit Logging
from datetime import datetime
import json


class AuditLogger:
    """Log security-relevant events."""

    def log_clarification_request(
        self,
        session_id: str,
        query: str,
        ip_address: str
    ):
        """Log clarification request."""
        event = {
            "type": "clarification_request",
            "session_id": session_id,
            "query_hash": hashlib.sha256(query.encode()).hexdigest(),
            "ip_address": ip_address,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        # Write to audit log
        self._write_audit_log(event)

    def log_suspicious_activity(
        self,
        session_id: str,
        reason: str,
        details: dict
    ):
        """Log suspicious activity."""
        event = {
            "type": "suspicious_activity",
            "session_id": session_id,
            "reason": reason,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self._write_audit_log(event)
```

#### Security Implementation Roadmap

1. **Week 1**: Input sanitization and validation
2. **Week 2**: Rate limiting and DOS protection
3. **Week 3**: Session authentication and tokens
4. **Week 4**: Audit logging and monitoring
5. **Week 5**: Security testing and hardening

### 6.3 Additional Future Enhancements

1. **Analytics and Metrics**

   - Track question completion rates
   - Identify confusing questions
   - Measure clarification effectiveness

2. **Question Templates**

   - Reusable question sets
   - Domain-specific templates
   - ML-based question generation

3. **Multi-language Support**

   - Internationalization (i18n)
   - Language detection
   - Translation integration

4. **Advanced UI Features**
   - Progress indicators
   - Save/resume functionality
   - Question preview/review

## 7. Performance Optimizations

### 7.1 Implemented Optimizations

1. **O(1) Question Lookups**: Using dict indexing instead of linear search
2. **Timezone-aware Timestamps**: Consistent UTC usage
3. **Leveraging ResearchState**: Using existing state management instead of custom session handling

### 7.2 Monitoring and Metrics

```python
# Performance monitoring setup

from prometheus_client import Counter, Histogram, Gauge
import time


# Metrics
clarification_requests = Counter(
    'clarification_requests_total',
    'Total clarification requests'
)
clarification_duration = Histogram(
    'clarification_duration_seconds',
    'Time to complete clarification'
)
active_research_states = Gauge(
    'active_research_states',
    'Number of active research states'
)


# Usage in workflow
async def process_with_monitoring(state: ResearchState, query: str):
    """Process with performance monitoring."""
    clarification_requests.inc()

    start_time = time.time()
    try:
        updated_state, result = await process_with_clarification(state, query)
        clarification_duration.observe(time.time() - start_time)
        return updated_state, result
    finally:
        # Count active research states from existing storage
        active_research_states.set(len(active_sessions))
```

## Conclusion

This redesign provides a robust, type-safe, and extensible clarification system that properly handles multiple questions with unique identifiers. The implementation follows Python best practices, ensures clear separation of concerns, and provides comprehensive error handling and validation.

The key improvements include:

- ✅ Proper data modeling with Pydantic validators (not `__post_init__`)
- ✅ Clear question-answer matching via UUIDs
- ✅ Support for different question types
- ✅ Explicit handling of skipped questions
- ✅ Integration with existing ResearchState management
- ✅ O(1) performance for lookups
- ✅ Timezone-aware datetime handling
- ✅ Type safety throughout the codebase
- ✅ Testable and maintainable architecture

Future work will add conditional question support and security enhancements, making the system even more powerful and production-ready.

Next steps: Implement these changes following the phased approach in Section 2, starting with the critical fixes and core model updates.
