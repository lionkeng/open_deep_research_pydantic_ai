# ClarificationAgent Evaluation Framework

This directory contains a comprehensive evaluation framework for testing the ClarificationAgent's real capabilities without mocking.

## Overview

The evaluation framework provides multiple approaches to test the ClarificationAgent:

1. **Integration Tests** (`test_clarification_real.py`) - Real agent tests with actual LLM calls
2. **Pydantic Evals Framework** (`clarification_evals.py`) - Structured evaluation with custom evaluators
3. **Test Dataset** (`clarification_dataset.yaml`) - Curated test cases organized by category
4. **Simple Runner** (`run_clarification_eval.py`) - Standalone evaluation script

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

## Usage

### Running Integration Tests

```bash
# Run with pytest
pytest tests/integration/test_clarification_real.py -v

# Run specific test
pytest tests/integration/test_clarification_real.py::TestClarificationAgentReal::test_golden_dataset -v
```

### Running Evaluation Suite

```bash
# Simple evaluation runner
python tests/evals/run_clarification_eval.py

# Quick test with minimal cases
python tests/evals/test_eval_quick.py
```

### Using Pydantic Evals Framework

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

## Configuration

### API Keys
Set your API keys in the `.env` file:
```bash
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here  # Optional
```

### Evaluation Settings
Adjust evaluation parameters in the code:
- `temperature=0` for more consistent outputs
- `num_runs=3` for consistency testing
- Custom weights for different evaluators

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

## Best Practices

1. **Use Semantic Similarity**: Don't expect exact matches for LLM outputs
2. **Allow Variation**: Set reasonable thresholds (e.g., 80% similarity)
3. **Run Multiple Times**: Check consistency across runs
4. **Monitor Performance**: Track metrics over time to catch regressions
5. **Update Golden Dataset**: Add new edge cases as discovered

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
