# Evaluation Framework Documentation

## Overview

This evaluation framework provides comprehensive testing for both Clarification and Query Transformation agents using the pydantic-ai evaluation patterns.

## Structure

### 1. YAML Datasets (`evaluation_datasets/`)

Both agents use YAML-based datasets for evaluation:

- **`clarification_dataset.yaml`**: Test cases for clarification agent
- **`query_transformation_dataset.yaml`**: Test cases for query transformation agent

#### Dataset Structure

```yaml
metadata:
  name: dataset_name
  version: 1.0.0

golden_standard_cases:
  - id: unique_id
    name: Human-readable name
    inputs:
      query: "User query"
      complexity: simple|medium|complex
      domain: technical|scientific|business
    expected_output:
      min_search_queries: 5
      max_search_queries: 12
      # ... other expectations
    evaluators:
      - EvaluatorName1
      - EvaluatorName2
```

### 2. Evaluators

#### Clarification Agent Evaluators
- `BinaryAccuracyEvaluator`: Validates clarification decision accuracy
- `DimensionCoverageEvaluator`: Checks 4-dimension framework coverage
- `QuestionRelevanceEvaluator`: Evaluates question quality
- `MultiQuestionEvaluator`: Tests multi-question capabilities
- `LLMJudgeEvaluator`: LLM-based quality assessment

#### Query Transformation Evaluators
- `SearchQueryRelevanceEvaluator`: Validates search query relevance
- `ObjectiveCoverageEvaluator`: Checks research objective coverage
- `PlanCoherenceEvaluator`: Evaluates research plan structure
- `QueryDiversityEvaluator`: Measures query diversity
- `TransformationAccuracyEvaluator`: Overall transformation quality

### 3. Dataset Loaders

- **`query_transformation_dataset_loader.py`**: Utilities for loading YAML datasets
  - `load_dataset_from_yaml()`: Load complete dataset
  - `load_golden_standard_dataset()`: Load only golden cases
  - `create_filtered_dataset()`: Create custom filtered datasets

### 4. Running Evaluations

#### Quick Test
```bash
# Test clarification agent
uv run python tests/evals/run_clarification_eval.py

# Test query transformation agent
uv run python tests/evals/run_query_transformation_eval.py
```

#### Load Specific Categories
```python
from tests.evals.query_transformation_dataset_loader import (
    load_golden_standard_dataset,
    load_technical_dataset,
    load_edge_cases_dataset
)

# Load only golden standard cases
dataset = load_golden_standard_dataset()

# Load only technical cases
dataset = load_technical_dataset()
```

#### Create Custom Filtered Dataset
```python
from tests.evals.query_transformation_dataset_loader import create_filtered_dataset

# Get only medium/complex technical cases
dataset = create_filtered_dataset(
    min_complexity="medium",
    domains=["technical"],
    max_cases=10
)
```

## Dataset Categories

### Query Transformation Dataset

1. **Golden Standard Cases** (3 cases)
   - Complex AI in Healthcare
   - Database Comparison for E-commerce
   - Climate Change Research

2. **Technical Cases** (4 cases)
   - Microservices Best Practices
   - Database Query Optimization
   - RESTful API Design
   - Cloud Migration Strategy

3. **Scientific Cases** (3 cases)
   - Quantum Computing Research
   - Vaccine Development Process
   - Renewable Energy Analysis

4. **Business Cases** (3 cases)
   - EV Market Analysis
   - Startup Strategy Development
   - Competitive Analysis

5. **Edge Cases** (5 cases)
   - Minimal Query
   - Ultra-Specific Query
   - Multi-Topic Query
   - Ambiguous Query
   - Contradictory Requirements

6. **Cross-Domain Cases** (2 cases)
   - AI Ethics and Law
   - FinTech Machine Learning

7. **Performance Cases** (2 cases)
   - Simple Performance Test
   - Complex Performance Test

**Total: 22 test cases**

## Evaluation Metrics

### Thresholds
- `relevance_threshold`: 0.7
- `coverage_threshold`: 0.75
- `coherence_threshold`: 0.7
- `accuracy_threshold`: 0.75
- `diversity_threshold`: 0.7

### Execution Configuration
- `max_concurrency`: 5
- `timeout_seconds`: 30
- `retry_attempts`: 2

## Best Practices

1. **Adding New Test Cases**: Add them to the appropriate YAML file section
2. **Creating New Evaluators**: Extend `pydantic_evals.evaluators.Evaluator`
3. **Custom Datasets**: Use the dataset loader utilities for filtering
4. **Performance Testing**: Use the performance category cases
5. **Edge Case Testing**: Use edge cases for robustness testing

## Integration with CI/CD

The evaluation framework can be integrated into CI/CD pipelines:

```bash
# Run all evaluations
uv run pytest tests/evals/

# Run specific evaluation
uv run python tests/evals/run_query_transformation_eval.py

# Run with custom dataset
EVAL_DATASET=golden_standard uv run python tests/evals/run_query_transformation_eval.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root
2. **YAML Parse Errors**: Validate YAML syntax
3. **Evaluator Not Found**: Check EVALUATOR_MAP in dataset_loader.py
4. **Timeout Issues**: Adjust timeout_seconds in evaluation_config

### Debug Mode

```python
# Enable debug logging
import logfire
logfire.configure(send_to_logfire=False, console=True)
```

## Future Enhancements

1. **Multi-model evaluation**: Test across different LLM models
2. **Regression testing**: Track performance over time
3. **A/B testing**: Compare different agent versions
4. **Custom metrics**: Add domain-specific evaluators
5. **Automated dataset generation**: Use LLMs to generate test cases
