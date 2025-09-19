# Testing Strategy for Pydantic AI Agents

## Problem with Previous Approach

The previous testing approach was mocking entire agent classes, which meant:
- We were testing our mocks, not the actual agent code
- The real agent logic was never exercised
- Tests could pass even with broken agent implementations
- No confidence that agents would work in production

## Improved Testing Strategy

### Core Principles

1. **Test Real Code**: Always use real agent instances, never mock the entire agent
2. **Mock Only External Dependencies**: Only mock LLM API calls and external services
3. **Validate Agent Logic**: Ensure the agent's internal logic and transformations work correctly
4. **Type Safety**: Verify that agents properly handle and return expected data types

### Implementation Pattern

```python
# DO THIS: Create real agent with test configuration
@pytest.fixture
def clarification_agent(test_config):
    with patch('src.agents.base.Agent') as MockAgent:
        mock_agent_instance = MagicMock()
        MockAgent.return_value = mock_agent_instance

        agent = ClarificationAgent(config=test_config)
        agent.agent = mock_agent_instance
        return agent

# DO THIS: Mock only the LLM call
with patch.object(agent.agent, "run", new_callable=AsyncMock) as mock_run:
    mock_run.return_value = mock_llm_response(expected_data)
    result = await agent.agent.run(query, deps=dependencies)

# DON'T DO THIS: Mock the entire agent
mock_agent = MagicMock(spec=ClarificationAgent)  # WRONG!
```

### Key Components

#### 1. Real Agent Instances
- Create actual agent instances with test configurations
- Use dependency injection for testability
- Agents should work with test models that won't make real API calls

#### 2. Focused Mocking
- Mock `agent.run()` method for LLM responses
- Mock external API clients (Tavily, search providers, etc.)
- Use `patch.object()` to mock specific methods, not entire classes

#### 3. Helper Utilities (`test_helpers.py`)
- `MockLLMAgent`: Helper class for managing LLM mocks with call tracking
- `create_mock_llm_response()`: Creates properly structured mock responses
- Validation helpers to ensure correct data structures

#### 4. Test Coverage Areas

##### Unit Tests Should Cover:
- **Happy Path**: Normal operation with expected inputs
- **Edge Cases**: Empty inputs, very long inputs, special characters
- **Error Handling**: API failures, timeout, invalid responses
- **Data Validation**: Ensure outputs match expected schemas
- **Concurrency**: Multiple simultaneous operations
- **Configuration**: Different models and parameters

##### Integration Tests Should Cover:
- Agent-to-agent communication
- Full workflow execution
- Real API calls (in separate test suite)
- Performance under load

### Example Test Structure

```python
class TestClarificationAgent:
    @pytest.mark.asyncio
    async def test_needs_clarification_basic(self, clarification_agent, mock_llm_response, sample_dependencies):
        # Arrange: Set up expected input and output
        query = "Tell me about AI"
        expected_output = ClarifyWithUser(
            needs_clarification=True,
            request=ClarificationRequest(...),
            reasoning="The query is too broad",
            missing_dimensions=["SPECIFICITY & SCOPE"],
            assessment_reasoning="Query needs narrowing"
        )

        # Mock only the LLM call
        with patch.object(clarification_agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_llm_response(expected_output)

            # Act: Call the real agent method
            result = await clarification_agent.agent.run(query, deps=sample_dependencies)

            # Assert: Validate the result
            assert result.output.needs_clarification is True
            assert isinstance(result.output.request, ClarificationRequest)
            mock_run.assert_called_once_with(query, deps=sample_dependencies)
```

### Benefits of This Approach

1. **Confidence**: Tests verify actual agent behavior
2. **Maintainability**: Changes to agent implementation are caught by tests
3. **Debugging**: Failures point to real issues, not mock problems
4. **Documentation**: Tests serve as examples of how to use agents
5. **Refactoring Safety**: Can refactor with confidence that tests catch regressions

### Migration Guide

To update existing tests:

1. Remove mock agent fixtures that mock entire agents
2. Create real agent instances with test configs
3. Replace agent mocks with `patch.object(agent.agent, "run")`
4. Ensure test data matches actual Pydantic models
5. Add validation for all required fields

### Test File Organization

```
tests/
├── unit/
│   └── agents/
│       ├── test_clarification_agent_unit.py  # Real agent, mocked LLM
│       ├── test_research_executor_unit.py
│       └── test_report_generator_unit.py
├── integration/
│   └── test_workflow_integration.py  # Multiple agents working together
├── test_helpers.py  # Shared mock utilities
└── TESTING_STRATEGY.md  # This document
```

### Running Tests

```bash
# Run all unit tests
uv run pytest tests/unit/

# Run specific agent tests
uv run pytest tests/unit/agents/test_clarification_agent_unit.py

# Run with coverage
uv run pytest tests/unit/ --cov=src/agents --cov-report=html

# Run specific test
uv run pytest tests/unit/agents/test_clarification_agent_unit.py::TestClarificationAgent::test_needs_clarification_basic -xvs
```

## Best Practices

1. **One Assertion Per Test**: Keep tests focused on a single behavior
2. **Descriptive Names**: Test names should explain what they're testing
3. **Arrange-Act-Assert**: Follow AAA pattern for clarity
4. **Independent Tests**: Tests shouldn't depend on each other
5. **Fast Execution**: Unit tests should run quickly (mock slow operations)
6. **Comprehensive Coverage**: Aim for >80% code coverage
7. **Document Why**: Comments should explain non-obvious test scenarios

## Common Pitfalls to Avoid

- **Don't mock what you're testing**: Never mock the agent you're testing
- **Don't test implementation details**: Test behavior, not internals
- **Don't ignore error cases**: Always test error handling
- **Don't skip type validation**: Ensure proper types are returned
- **Don't assume API responses**: Always validate response structure
- **Don't use real API keys**: Use test keys or mock the calls

## Example: Using MockLLMAgent Helper

```python
from tests.test_helpers import MockLLMAgent

async def test_with_helper(clarification_agent):
    # Create helper for this agent
    mock_helper = MockLLMAgent(clarification_agent.agent)

    # Use the helper to mock responses
    expected_output = ClarifyWithUser(...)

    with mock_helper.mock_response(expected_output):
        result = await clarification_agent.agent.run(query, deps=deps)
        assert result.output.needs_clarification is True

    # Check call history
    last_call = mock_helper.get_last_call()
    assert "AI" in last_call["args"][0]
```

## Testing Checklist

Before submitting a PR, ensure:

- [ ] All tests pass locally
- [ ] New features have unit tests
- [ ] Edge cases are tested
- [ ] Error handling is tested
- [ ] Tests use real agents with mocked dependencies
- [ ] No entire agents are mocked
- [ ] Test coverage is maintained or improved
- [ ] Tests follow the AAA pattern
- [ ] Test names are descriptive

## Future Improvements

- Add property-based testing for edge cases
- Implement contract tests for agent interfaces
- Create performance benchmarks
- Add mutation testing to verify test quality
- Build test data factories for complex scenarios
- Add snapshot testing for complex outputs
