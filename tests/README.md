# Testing Framework for Open Deep Research with Pydantic AI

This comprehensive testing framework provides validation for the research agents, with a particular focus on the clarification agent system.

## Test Structure Overview

```
tests/
├── unit/                        # Unit tests for individual components
│   ├── agents/
│   │   └── test_clarification_agent_unit.py
│   └── models/
│       └── test_clarification_models.py
├── integration/                 # Integration and workflow tests
│   └── test_clarification_workflows.py
├── evals/                       # LLM evaluation framework
│   ├── clarification_evals.py   # Specialized clarification evaluations
│   ├── multi_judge_evaluation.py # Multi-judge consensus system
│   ├── domain_specific_evals.py # Domain-specific evaluations
│   ├── regression_tracker.py    # Performance tracking
│   └── evaluation_runner.py     # Unified evaluation orchestration
├── acceptance/                  # End-to-end acceptance tests
│   └── test_final_validation.py
└── conftest.py                  # Pytest configuration and fixtures
```

## Testing the Clarification Agent

### 1. Model Tests

Test the data models used by the clarification agent:

```bash
# Test clarification models (ClarificationRequest, ClarificationResponse, etc.)
uv run pytest tests/unit/models/test_clarification_models.py -v

# Quick validation of model functionality
uv run pytest tests/unit/models/test_clarification_models.py::TestClarificationQuestion -v
```

### 2. Unit Tests

Test agent logic in isolation:

```bash
# Run all unit tests for the clarification agent
uv run pytest tests/unit/agents/test_clarification_agent.py -v

# Test specific functionality
uv run pytest tests/unit/agents/test_clarification_agent_unit.py::TestClarificationAgentUnit::test_response_structure_validation -v
```

### 3. Integration Tests

Test workflow integration and dependencies:

```bash
# Run integration tests
uv run pytest tests/integration/test_clarification_workflows.py -v

# Test with real dependencies
source .env && uv run pytest tests/integration/test_clarification_workflows.py::test_real_clarification_workflow -v
```

### 4. LLM Evaluation Framework

Run comprehensive evaluations using the Pydantic AI evaluation framework:

```bash
# Run the comprehensive evaluation suite
source .env && uv run python tests/evals/run_clarification_eval.py

# Run specific evaluations
source .env && uv run python -c "
import asyncio
from tests.evals.clarification_evals import ClarificationEvaluator
from src.agents.base import ResearchDependencies
from src.models.api_models import APIKeys
from pydantic import SecretStr
import os

async def main():
    deps = ResearchDependencies(
        api_keys=APIKeys(openai=SecretStr(os.getenv('OPENAI_API_KEY'))),
        research_state=None
    )
    evaluator = ClarificationEvaluator(deps)
    evaluator.create_comprehensive_test_cases()
    results = await evaluator.run_evaluation_suite()
    print(f'Overall accuracy: {results[\"summary\"][\"overall_metrics\"][\"accuracy\"]:.2%}')

asyncio.run(main())
"

# Run multi-judge evaluation
source .env && uv run python -c "
import asyncio
from tests.evals.multi_judge_evaluation import MultiJudgeEvaluator
from src.agents.base import ResearchDependencies
from src.models.api_models import APIKeys
from pydantic import SecretStr
import os

async def main():
    deps = ResearchDependencies(
        api_keys=APIKeys(openai=SecretStr(os.getenv('OPENAI_API_KEY'))),
        research_state=None
    )
    evaluator = MultiJudgeEvaluator(deps)
    evaluator.create_default_test_cases()
    results = await evaluator.run_evaluation_suite()
    print(f'Average score: {results[\"summary\"][\"avg_overall_score\"]:.2f}/5')

asyncio.run(main())
"

# Run domain-specific evaluation
source .env && uv run python -c "
import asyncio
from tests.evals.domain_specific_evals import DomainEvaluationOrchestrator
from src.agents.base import ResearchDependencies
from src.models.api_models import APIKeys
from pydantic import SecretStr
import os

async def main():
    deps = ResearchDependencies(
        api_keys=APIKeys(openai=SecretStr(os.getenv('OPENAI_API_KEY'))),
        research_state=None
    )
    orchestrator = DomainEvaluationOrchestrator(deps)
    orchestrator.create_default_test_cases()
    results = await orchestrator.run_all_evaluations()
    print(f'Pass rate: {results[\"pass_rate\"]:.2%}')

asyncio.run(main())
"
```

### 5. Regression Testing

Track performance over time and detect regressions:

```bash
# Establish a baseline
source .env && uv run python -c "
import asyncio
from tests.evals.regression_tracker import establish_new_baseline
from src.agents.base import ResearchDependencies
from src.models.api_models import APIKeys
from pydantic import SecretStr
import os

async def main():
    deps = ResearchDependencies(
        api_keys=APIKeys(openai=SecretStr(os.getenv('OPENAI_API_KEY'))),
        research_state=None
    )
    baseline = await establish_new_baseline(deps, 'v1.0.0')
    print(f'Baseline established with {baseline.summary_stats[\"pass_rate\"]:.2%} pass rate')

asyncio.run(main())
"

# Run regression check against baseline
source .env && uv run python -c "
import asyncio
from tests.evals.regression_tracker import run_quick_regression_check
from src.agents.base import ResearchDependencies
from src.models.api_models import APIKeys
from pydantic import SecretStr
import os

async def main():
    deps = ResearchDependencies(
        api_keys=APIKeys(openai=SecretStr(os.getenv('OPENAI_API_KEY'))),
        research_state=None
    )
    results = await run_quick_regression_check(deps, baseline_version='v1.0.0')
    if 'baseline_comparison' in results:
        comp = results['baseline_comparison']['comparison_summary']
        print(f'Regression check: {comp[\"improved\"]} improved, {comp[\"degraded\"]} degraded')

asyncio.run(main())
"
```

### 6. Comprehensive Evaluation Runner

Run the complete evaluation suite with reporting:

```bash
# Run all evaluations with comprehensive reporting
source .env && uv run python tests/evals/evaluation_runner.py --suites all --output-formats console markdown json

# Run specific evaluation suites
source .env && uv run python tests/evals/evaluation_runner.py --suites clarification domain_specific --output-formats console

# Run with regression detection
source .env && uv run python tests/evals/evaluation_runner.py --suites regression --baseline-version v1.0.0
```

### 7. Acceptance Tests

Run final validation tests:

```bash
# Run acceptance tests
uv run pytest tests/acceptance/test_final_validation.py -v

# Run system health checks
uv run pytest tests/acceptance/test_final_validation.py::TestSystemHealth -v

# Run critical user journeys
source .env && uv run pytest tests/acceptance/test_final_validation.py::TestCriticalUserJourneys -v
```

## Quick Test Commands

### Minimal Testing (No API Keys Required)

```bash
# Unit tests only
uv run pytest tests/unit/ -v

# Model tests
uv run pytest tests/unit/models/ -v

# System health checks
uv run pytest tests/acceptance/test_final_validation.py::TestSystemHealth -v
```

### Standard Testing (API Keys Required)

```bash
# Set up environment
source .env

# Run core evaluation suite
uv run python tests/evals/run_clarification_eval.py

# Run regression tests
uv run python tests/evals/evaluation_runner.py --suites regression clarification
```

### Comprehensive Testing

```bash
# Run everything
source .env && uv run python tests/evals/evaluation_runner.py --suites all --output-formats console markdown json html
```

## Test Coverage Areas

### Clarification Agent Coverage

- **Unit Tests**: Agent initialization, response structure, input validation, error handling
- **Integration Tests**: Workflow integration, dependency injection, performance monitoring
- **Model Tests**: Data model validation, serialization, consistency checks
- **LLM Evaluations**:
  - Clarification accuracy (>90% target)
  - Question quality assessment
  - Domain-specific performance
  - Multi-judge consensus validation
- **Regression Testing**: Performance tracking, regression detection, baseline comparisons
- **Acceptance Tests**: End-to-end validation, deployment readiness

## Evaluation Metrics

### Core Metrics

- **Accuracy**: Correct clarification decisions (target: >90%)
- **Precision**: Of clarifications requested, how many were needed
- **Recall**: Of needed clarifications, how many were requested
- **F1 Score**: Harmonic mean of precision and recall
- **Response Time**: Average processing time (target: <2s)

### Quality Metrics

- **Question Quality**: 1-5 scale rating of clarification questions
- **Domain Accuracy**: Performance on domain-specific queries
- **Edge Case Handling**: Robustness score for edge cases
- **Multi-Judge Agreement**: Consensus among multiple evaluators

## Configuration

### Environment Variables

```bash
# Required for LLM evaluations
OPENAI_API_KEY=your_api_key

# Optional configuration
LOGFIRE_IGNORE_NO_CONFIG=1  # Suppress Logfire warnings
```

### Test Markers

```bash
# Run specific test categories
uv run pytest -m "unit" -v        # Unit tests only
uv run pytest -m "integration" -v  # Integration tests only
uv run pytest -m "slow" -v         # Long-running tests
uv run pytest -m "performance" -v  # Performance tests
```

## Adding New Tests

When adding tests for the clarification agent:

1. **Unit Tests (Agents)**: Add to `tests/unit/agents/test_clarification_agent_unit.py`
2. **Unit Tests (Models)**: Add to `tests/unit/models/test_clarification_models.py`
3. **Integration Tests**: Add to `tests/integration/test_clarification_workflows.py`
4. **Evaluation Cases**: Add to `tests/evals/clarification_evals.py`

Follow these patterns:

- Use fixtures from `conftest.py`
- Mock external dependencies in unit tests
- Use real agents for integration/evaluation tests
- Add appropriate markers (@pytest.mark.unit, @pytest.mark.slow, etc.)

## Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Ensure proper Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **API Key Issues**

   ```bash
   # Check API key is set
   echo $OPENAI_API_KEY

   # Run without API key (unit tests only)
   uv run pytest tests/unit/ -v
   ```

3. **Performance Test Failures**

   - Increase timeouts for slower systems
   - Run with `--verbose` for detailed output
   - Check resource usage with performance monitoring

4. **Evaluation Failures**
   - Verify API keys are valid
   - Check network connectivity
   - Review error logs in `tests/evals/reports/`

## CI/CD Integration

For continuous integration:

```yaml
# Example GitHub Actions workflow
- name: Run Unit Tests
  run: uv run pytest tests/unit/ -v

- name: Run Model Tests
  run: uv run pytest tests/unit/models/ -v

- name: Run Evaluations
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    uv run python tests/evals/evaluation_runner.py \
      --suites clarification regression \
      --fail-on-regression \
      --output-formats json
```

## Future Enhancements

Planned improvements for the testing framework:

- [ ] Automated performance benchmarking
- [ ] Visual evaluation dashboards
- [ ] Comparative analysis across model versions
- [ ] Automated test case generation
- [ ] Integration with monitoring systems
