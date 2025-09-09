# ClarificationAgent Evaluation Framework

This directory contains a comprehensive evaluation framework for testing the ClarificationAgent's real capabilities without mocking.

## Overview

The evaluation framework provides a hierarchical testing structure from quick smoke tests to comprehensive multi-LLM evaluations:

### 📊 Evaluation Files Hierarchy

| File | Purpose | Runtime | Use Case |
|------|---------|---------|----------|
| `test_eval_quick.py` | Quick smoke test (4 cases) | < 1 min | Pre-commit validation |
| `run_clarification_eval.py` | Standard evaluation (20+ cases) | 5-10 min | Feature development |
| `clarification_evals.py` | Pydantic Evals framework | 10-15 min | Detailed analysis |
| `evaluation_runner.py` | Master orchestrator (all suites) | 20-30 min | Release validation |
| `multi_judge_evaluation.py` | Multi-LLM judge evaluation | 15-20 min | Quality assurance |
| `domain_specific_evals.py` | Domain-specific scenarios | 10-15 min | Specialized testing |
| `regression_tracker.py` | Performance tracking | Continuous | CI/CD monitoring |

## Key Features

### 1. Real Testing (No Mocks)
Unlike unit tests that mock all LLM calls, this framework actually runs the agent to test:
- Binary correctness: Does it correctly identify when clarification is needed?
- Question quality: Are the clarification questions relevant and helpful?
- Dimension coverage: Does it follow the 4-dimension framework?
- Consistency: Does it produce similar results across runs?

### 2. Evaluation Metrics

#### Quantitative Metrics
- **Accuracy**: Binary classification accuracy
- **Precision/Recall**: For clarification detection
- **F1 Score**: Harmonic mean of precision and recall
- **Dimension Coverage**: How well the 4-dimension framework is utilized

#### Qualitative Metrics
- **Question Relevance**: Evaluated via semantic analysis
- **LLM-as-Judge**: Another LLM evaluates clarification quality
- **Consistency Score**: Similarity across multiple runs

### 3. Test Categories

#### Clear Queries (Should NOT need clarification)
- Specific financial queries: "What is the current Bitcoin price?"
- Technical comparisons: "Compare ResNet-50 vs VGG-16"
- Clear programming tasks: "Implement quicksort in Python"

#### Ambiguous Queries (SHOULD need clarification)
- Broad topics: "What is AI?"
- Ambiguous terms: "Tell me about Python" (language or snake?)
- Vague research: "Research climate change"
- Missing context: "How does it work?"

#### Edge Cases
- Minimal queries: "?"
- Multiple questions in one
- Very long queries
- Non-English queries

## 🚀 Running Evaluations

### Level 1: Quick Validation (< 1 minute)
Use for rapid smoke testing before commits or during development:

```bash
# Run quick test with 4 essential test cases
source .env && uv run python tests/evals/test_eval_quick.py
```

**When to use:**
- Pre-commit hooks
- Quick validation during development
- Sanity checks after code changes

### Level 2: Standard Evaluation (5-10 minutes)
Use for thorough testing of core functionality:

```bash
# Run standard evaluation with 20+ diverse test cases
source .env && uv run python tests/evals/run_clarification_eval.py
```

**When to use:**
- After implementing new features
- Before creating pull requests
- Daily development testing

### Level 3: Comprehensive Evaluation Suite (20-30 minutes)
Use for complete system validation:

```bash
# Run ALL evaluation suites
source .env && uv run python tests/evals/evaluation_runner.py --suites all

# Run specific evaluation suites
source .env && uv run python tests/evals/evaluation_runner.py --suites llm_evaluations,multi_judge

# Generate detailed reports in multiple formats
source .env && uv run python tests/evals/evaluation_runner.py --suites all --output-formats console,html,json
```

**Available suites:**
- `unit_tests` - Unit test execution
- `integration_tests` - Integration test execution
- `llm_evaluations` - LLM-based evaluations
- `multi_judge` - Multi-LLM judge evaluation
- `domain_specific` - Domain-specific scenarios
- `regression_tracking` - Performance regression detection
- `performance_benchmarks` - Speed and resource benchmarks
- `all` - Run everything

### Level 4: Specialized Evaluations

#### Pydantic Evals Framework
For detailed custom evaluations with multiple evaluators:

```bash
# Run comprehensive Pydantic-based evaluation
source .env && uv run python tests/evals/clarification_evals.py
```

#### Multi-Judge Evaluation
For quality assessment using multiple LLMs as judges:

```bash
# Run multi-LLM judge evaluation
source .env && uv run python tests/evals/multi_judge_evaluation.py
```

#### Domain-Specific Evaluation
For testing performance in specific domains:

```bash
# Run domain-specific test scenarios
source .env && uv run python tests/evals/domain_specific_evals.py
```

### Integration Tests
For testing with real LLM calls using pytest:

```bash
# Run all integration tests
pytest tests/integration/test_clarification_real.py -v

# Run specific test
pytest tests/integration/test_clarification_real.py::TestClarificationAgentReal::test_golden_dataset -v
```

## 📈 Evaluation Decision Tree

```
Start Here
    │
    ├─ Need quick validation? (< 1 min)
    │   └─► Run: test_eval_quick.py
    │
    ├─ Testing new feature? (5-10 min)
    │   └─► Run: run_clarification_eval.py
    │
    ├─ Pre-release validation? (20-30 min)
    │   └─► Run: evaluation_runner.py --suites all
    │
    └─ Specific concern?
        ├─ Quality assurance?
        │   └─► Run: multi_judge_evaluation.py
        ├─ Domain performance?
        │   └─► Run: domain_specific_evals.py
        └─ Regression detection?
            └─► Run: evaluation_runner.py --suites regression_tracking
```

## 📊 Test Coverage Matrix

| Evaluation Aspect | Quick | Standard | Pydantic | Multi-Judge | Domain | Runner |
|------------------|-------|----------|----------|-------------|---------|---------|
| Binary Accuracy | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Multi-Question Support | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dimension Coverage | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Question Quality | ❌ | ⚡ | ✅ | ✅ | ✅ | ✅ |
| Response Time | ⚡ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Consistency Check | ❌ | ❌ | ✅ | ✅ | ⚡ | ✅ |
| LLM-as-Judge | ❌ | ❌ | ✅ | ✅ | ⚡ | ✅ |
| Domain Specific | ❌ | ❌ | ⚡ | ⚡ | ✅ | ✅ |
| Regression Tracking | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| CI/CD Integration | ⚡ | ✅ | ✅ | ⚡ | ⚡ | ✅ |

**Legend:** ✅ Full support | ⚡ Partial support | ❌ Not supported

### Using Pydantic Evals Framework Programmatically

```python
from tests.evals.clarification_evals import (
    create_clarification_dataset,
    run_clarification_evaluation,
    generate_evaluation_report
)

# Run evaluation
report = await run_clarification_evaluation()

# Generate human-readable report
print(generate_evaluation_report(report))
```

## Custom Evaluators

The framework includes several custom evaluators:

### BinaryAccuracyEvaluator
Evaluates if the clarification decision (yes/no) is correct.

### DimensionCoverageEvaluator
Checks coverage of the 4-dimension framework:
- Audience Level & Purpose
- Scope & Focus Areas
- Source & Quality Requirements
- Deliverable Specifications

### QuestionRelevanceEvaluator
Evaluates the relevance and quality of clarification questions.

### ConsistencyEvaluator
Runs the same query multiple times to check consistency.

### LLMJudgeEvaluator
Uses another LLM to judge clarification quality based on:
- Relevance to original query
- Identification of key ambiguities
- Helpfulness for providing better answers
- Clarity and specificity

## Test Dataset Format

The `clarification_dataset.yaml` file organizes test cases by category:

```yaml
cases:
  clear_queries:
    - name: bitcoin_price
      input:
        query: "What is the current Bitcoin price?"
      expected:
        need_clarification: false

  ambiguous_queries:
    - name: broad_ai
      input:
        query: "What is AI?"
      expected:
        need_clarification: true
        dimension_categories:
          - audience_level
          - scope_focus
        key_themes:
          - "artificial intelligence"
          - "specific aspect"
```

## ⚙️ Configuration

### API Keys
Set your API keys in the `.env` file:
```bash
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here  # Optional for multi-judge evaluation
MODEL_NAME=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
```

### Evaluation Runner Configuration
Configure comprehensive evaluation settings:

```bash
# Run with specific configuration
source .env && uv run python tests/evals/evaluation_runner.py \
    --suites llm_evaluations,regression_tracking \
    --output-formats console,html,json \
    --output-dir ./evaluation_results \
    --baseline-days 30 \
    --fail-on-regression \
    --fail-on-accuracy-drop 0.05
```

**Configuration Options:**
- `--suites`: Choose evaluation suites to run
- `--output-formats`: Output format (console, html, json, markdown, dashboard)
- `--output-dir`: Directory for saving results
- `--baseline-days`: Days of historical data for regression detection
- `--fail-on-regression`: Exit with error code if regression detected
- `--fail-on-accuracy-drop`: Threshold for accuracy drop (0.05 = 5%)
- `--include-trends`: Include trend analysis in reports
- `--include-recommendations`: Generate improvement recommendations

### Evaluation Parameters
Customize evaluation behavior in individual scripts:

```python
# In test files, adjust these parameters:
TEMPERATURE = 0  # For consistent outputs
NUM_RUNS = 3  # For consistency testing
TIMEOUT = 30  # API timeout in seconds
MAX_RETRIES = 2  # Retry failed API calls
```

### CI/CD Integration
Example GitHub Actions workflow:

```yaml
name: Evaluation Suite
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Quick Validation
        run: |
          source .env
          uv run python tests/evals/test_eval_quick.py
      - name: Standard Evaluation (PR only)
        if: github.event_name == 'pull_request'
        run: |
          source .env
          uv run python tests/evals/run_clarification_eval.py
      - name: Comprehensive Evaluation (main branch)
        if: github.ref == 'refs/heads/main'
        run: |
          source .env
          uv run python tests/evals/evaluation_runner.py \
            --suites all \
            --fail-on-regression \
            --output-formats json \
            --output-dir ./results
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: evaluation-results
          path: ./results/
```

## Output

The evaluation generates:
1. **Console Report**: Summary metrics and per-category performance
2. **JSON Results**: Detailed results saved to `evaluation_results.json`
3. **Metrics**:
   - Overall accuracy
   - Confusion matrix
   - Per-category breakdown
   - Common failure patterns
   - Dimension distribution

## 📖 Interpreting Results

### Success Criteria

| Metric | Excellent | Good | Needs Improvement |
|--------|-----------|------|-------------------|
| Binary Accuracy | > 95% | 85-95% | < 85% |
| Precision | > 90% | 80-90% | < 80% |
| Recall | > 90% | 80-90% | < 80% |
| F1 Score | > 0.90 | 0.80-0.90 | < 0.80 |
| Response Time | < 2s | 2-5s | > 5s |
| Consistency | > 95% | 85-95% | < 85% |
| Question Quality* | > 4.5/5 | 3.5-4.5/5 | < 3.5/5 |

*As judged by LLM evaluators

### Common Failure Patterns

1. **False Positives**: Agent asks for clarification when query is clear
   - Usually indicates overly cautious prompting
   - Check dimension detection thresholds

2. **False Negatives**: Agent misses ambiguous queries
   - May need more training examples
   - Review edge cases in dataset

3. **Poor Question Quality**: Questions are irrelevant or unhelpful
   - Review question generation templates
   - Check dimension mapping logic

4. **Inconsistent Results**: Same query gets different responses
   - Reduce temperature setting
   - Add more structured prompting

## Best Practices

1. **Start Small, Scale Up**: Begin with quick tests, progressively run more comprehensive evaluations
2. **Use Semantic Similarity**: Don't expect exact matches for LLM outputs
3. **Allow Variation**: Set reasonable thresholds (e.g., 80% similarity)
4. **Run Multiple Times**: Check consistency across runs (use NUM_RUNS=3)
5. **Monitor Performance**: Track metrics over time to catch regressions
6. **Update Golden Dataset**: Add new edge cases as discovered
7. **Version Control Results**: Save evaluation results with git commits for tracking
8. **Automate in CI/CD**: Integrate appropriate evaluation levels in your pipeline

## Extending the Framework

To add new test cases:
1. Add entries to `clarification_dataset.yaml`
2. Or modify the test lists in the Python files

To add new evaluators:
1. Create a class inheriting from `Evaluator`
2. Implement the `evaluate()` method
3. Add to the evaluator list in dataset creation

## Troubleshooting

- **Timeouts**: Reduce the number of test cases or increase timeout
- **API Errors**: Check your API keys and rate limits
- **Inconsistent Results**: Use `temperature=0` for more deterministic outputs
- **Import Errors**: Ensure you're running from the project root
