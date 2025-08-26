# Three-Phase Clarification System Testing Framework

This testing framework provides comprehensive validation for the three-phase clarification improvement system implemented in this project.

## Overview

The testing framework validates:
- **Phase 1**: Enhanced Clarification Assessment
- **Phase 2**: Query Transformation
- **Phase 3**: Enhanced Brief Generation
- **Integration**: End-to-end workflow orchestration

## Test Structure

### Core Test Files

1. **`test_three_phase_integration.py`**: End-to-end integration tests
   - Complete workflow testing
   - Data flow validation between phases
   - Metadata schema consistency
   - Non-interactive (HTTP) mode testing

2. **`test_performance_validation.py`**: Performance and accuracy tests
   - Algorithm accuracy validation (>90% target)
   - Response time benchmarking
   - Concurrent execution testing
   - Memory usage validation
   - Edge case handling

3. **`conftest.py`**: Pytest configuration and fixtures
   - Mock components and dependencies
   - Standard test data sets
   - Performance test configurations

### Existing Test Files (Enhanced)

4. **`test_new_agents.py`**: Individual agent testing
5. **`test_workflow_integration.py`**: Basic workflow integration

## Running Tests

### Using the Test Runner (Recommended)
From the project root directory:
```bash
# Full test suite with the test runner
python tests/run_tests.py

# Quick validation tests
python tests/run_tests.py --quick

# Integration tests only
python tests/run_tests.py --integration

# Performance tests only
python tests/run_tests.py --performance

# Algorithm accuracy tests only
python tests/run_tests.py --accuracy

# Verbose output for debugging
python tests/run_tests.py --verbose
```

### Direct pytest Execution
```bash
# Full test suite
LOGFIRE_IGNORE_NO_CONFIG=1 uv run python -m pytest tests/ -v

# Integration tests only
LOGFIRE_IGNORE_NO_CONFIG=1 uv run python -m pytest tests/test_three_phase_integration.py -v

# Performance tests only
LOGFIRE_IGNORE_NO_CONFIG=1 uv run python -m pytest tests/test_performance_validation.py -v

# Accuracy validation
LOGFIRE_IGNORE_NO_CONFIG=1 uv run python -m pytest tests/test_performance_validation.py::TestPerformanceValidation::test_clarification_algorithm_accuracy -v
```

### Test Markers
```bash
# Slow tests (>30s)
LOGFIRE_IGNORE_NO_CONFIG=1 uv run python -m pytest -m "slow" -v

# Performance tests
LOGFIRE_IGNORE_NO_CONFIG=1 uv run python -m pytest -m "performance" -v

# Algorithm accuracy tests
LOGFIRE_IGNORE_NO_CONFIG=1 uv run python -m pytest -m "accuracy" -v
```

## Test Categories

### 1. Integration Tests (`test_three_phase_integration.py`)

- **`test_specific_query_minimal_processing()`**: Validates that specific queries require minimal clarification
- **`test_broad_query_processing()`**: Validates that broad queries go through all three phases
- **`test_data_flow_between_phases()`**: Validates data flows correctly between all phases
- **`test_error_handling_and_fallback()`**: Tests graceful error handling and fallback mechanisms
- **`test_metadata_schema_consistency()`**: Validates metadata schema remains consistent
- **`test_non_interactive_http_mode_simulation()`**: Tests HTTP mode behavior
- **`test_query_categorization_accuracy()`**: Basic query categorization validation
- **`test_workflow_performance_basic()`**: Basic performance characteristics

### 2. Performance Tests (`test_performance_validation.py`)

- **`test_clarification_algorithm_accuracy()`**: Algorithm accuracy validation (>90% target)
- **`test_workflow_response_time()`**: Response time benchmarking
- **`test_concurrent_workflow_execution()`**: Concurrent execution validation
- **`test_memory_usage_validation()`**: Memory usage monitoring
- **`test_workflow_interruption_recovery()`**: Interruption and recovery testing

### 3. Edge Case Tests (`TestEdgeCaseHandling`)

- **`test_empty_query_handling()`**: Empty/whitespace query handling
- **`test_very_long_query_handling()`**: Extremely long query handling
- **`test_special_characters_handling()`**: Special characters and formatting

## Performance Targets

### Algorithm Accuracy
- **Target**: >90% accuracy for broad vs. specific query classification
- **Test**: `test_clarification_algorithm_accuracy()`
- **Dataset**: 8+ curated test cases covering broad and specific queries

### Response Time
- **Target**: <45s average for planning phase (realistic production target)
- **Target**: <90s maximum for individual queries
- **Test**: `test_workflow_response_time()`

### Memory Usage
- **Target**: <100MB memory increase for 3 workflow iterations (realistic leak detection)
- **Test**: `test_memory_usage_validation()`

### Concurrent Execution
- **Target**: Multiple workflows should execute without interference
- **Test**: `test_concurrent_workflow_execution()`

## Test Data

### Algorithm Accuracy Dataset
The testing framework includes expanded curated test cases for accuracy validation (22 total cases):

**Specific Queries (Expected: No Clarification) - 10 cases**
- Simple facts: "What is 2+2?"
- Data requests: "Current stock price of AAPL"
- Technical comparisons: "Compare React hooks useState vs useReducer for TypeScript"
- Implementation guides: "How to implement JWT authentication in Node.js Express"
- Regulatory/compliance: "Current FDA approval status for Alzheimer's drug aducanumab"

**Intermediate Queries (Edge Cases) - 2 cases**
- Moderate concepts: "How does solar energy work?"
- Technical overviews: "Benefits of microservices architecture"

**Broad Queries (Expected: Needs Clarification) - 10 cases**
- Broad concepts: "What is artificial intelligence?"
- Domain-wide: "What should I know about healthcare?"
- Abstract/philosophical: "What is the meaning of innovation?"
- Impact analysis: "What are the implications of climate change?"

### Performance Test Queries
Standard set of queries for performance benchmarking across different complexity levels.

## Mock Components

The `MockWorkflowComponents` class provides standardized mock data:

```python
mock_components = MockWorkflowComponents()

# Mock clarification response
clarification = mock_components.create_mock_clarification_response(
    needs_clarification=True,
    question="What specific aspect interests you?"
)

# Mock transformation data
transformation = mock_components.create_mock_transformation_data(
    specificity_score=0.8
)

# Mock brief result
brief = mock_components.create_mock_brief_result(
    confidence=0.9
)
```

## Configuration

Tests use the following configuration:
- **Async Testing**: pytest-asyncio with STRICT mode
- **Logging**: Reduced verbosity during tests
- **Timeouts**: Reasonable timeouts for development testing
- **Mocking**: External dependencies mocked (HTTP clients, API keys)

## Coverage Expectations

The testing framework aims to cover:
- ✅ **Integration**: Complete three-phase workflow
- ✅ **Performance**: Algorithm accuracy and response times
- ✅ **Error Handling**: Graceful failure and recovery
- ✅ **Edge Cases**: Malformed and extreme inputs
- ✅ **Concurrency**: Multi-user and concurrent execution
- ✅ **Data Validation**: Schema consistency and integrity

## Adding New Tests

When adding new tests:

1. **Use appropriate fixtures**: Leverage `conftest.py` fixtures
2. **Follow naming conventions**: `test_*` for test functions
3. **Add appropriate markers**: `@pytest.mark.slow`, `@pytest.mark.performance`, etc.
4. **Include assertions with messages**: Clear failure messages
5. **Mock external dependencies**: Avoid real API calls in unit tests
6. **Document expected behavior**: Clear docstrings explaining test purpose

## Troubleshooting

### Common Issues

1. **Tests timeout**: Tests may take time due to LLM agent execution
   - Use mocks for unit tests
   - Increase timeouts for integration tests

2. **Import errors**: Ensure `src/` is in Python path
   - Add `sys.path.append('src')` or run with proper PYTHONPATH

3. **Agent initialization errors**: Agents require proper configuration
   - Use `LOGFIRE_IGNORE_NO_CONFIG=1` prefix
   - Ensure mock dependencies are provided

4. **Async test issues**: Ensure proper async test setup
   - Use `@pytest.mark.asyncio` decorator
   - Properly await async functions

### Performance Considerations

- **Real LLM Calls**: Integration tests make real agent calls (slower)
- **Mock vs Real**: Balance between test speed and accuracy
- **Resource Usage**: Monitor memory and CPU during testing
- **Parallel Execution**: Some tests can run in parallel, others cannot

This testing framework provides comprehensive validation of the three-phase clarification improvement system while maintaining reasonable test execution times and resource usage.
