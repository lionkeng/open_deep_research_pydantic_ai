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
      query: 'User query'
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

The Query Transformation evaluation framework includes both original and enhanced evaluators that provide comprehensive behavioral coverage of the agent.

##### **SearchQueryRelevanceEvaluator**

- **Goal**: Ensure generated search queries maintain semantic connection to the original query
- **What it measures**:
  - Word overlap between original and transformed queries
  - Coverage of expected search themes
  - Relevance scores for each generated query
- **Why it matters**: Prevents query drift and ensures searches remain on-topic
- **Scoring approach**: Combines lexical overlap analysis with theme coverage validation

##### **ObjectiveCoverageEvaluator**

- **Goal**: Ensure comprehensive, specific, diverse, and aligned research objectives
- **What it measures**:
  - Objective count validation (within min/max bounds)
  - Specificity through action verbs ("analyze", "evaluate", "compare")
  - Diversity to avoid redundant objectives
  - Alignment with original query intent
- **Why it matters**: Well-defined objectives guide effective research execution
- **Scoring approach**: Multi-factor assessment of count, quality, diversity, and alignment

##### **PlanCoherenceEvaluator**

- **Goal**: Ensure well-structured, methodologically sound research plans
- **What it measures**:
  - Methodology quality (approach, data sources, analysis methods, quality criteria)
  - Deliverables clarity and appropriateness
  - Success metrics definition
- **Why it matters**: Coherent plans lead to systematic, reproducible research
- **Scoring approach**: Evaluates structural completeness and logical consistency

##### **QueryDiversityEvaluator**

- **Goal**: Ensure search queries explore different aspects and perspectives
- **What it measures**:
  - Lexical diversity (unique word ratio)
  - Query type diversity (factual, analytical, comparative)
  - Length variation for broad and specific searches
- **Why it matters**: Diverse queries capture comprehensive information
- **Scoring approach**: Statistical analysis of lexical, type, and length variations

##### **TransformationAccuracyEvaluator**

- **Goal**: Ensure accurate, complete, and internally consistent transformations
- **What it measures**:
  - Query preservation (original intent maintained)
  - Completeness (all required components present)
  - Internal consistency (balanced objectives-to-queries ratio)
- **Why it matters**: Validates overall transformation quality and coherence
- **Scoring approach**: Holistic assessment of preservation, completeness, and consistency

##### **AssumptionQualityEvaluator**

- **Goal**: Evaluate the quality and reasonableness of assumptions made during transformation
- **What it measures**:
  - Assumption count appropriateness relative to identified gaps
  - Clarity and explicitness of assumptions
  - Gap coverage by assumptions
  - Risk assessment of assumptions made
- **Why it matters**: Ensures assumptions are reasonable and well-documented
- **Scoring approach**: Multi-factor assessment of count, clarity, coverage, and risk

##### **PriorityDistributionEvaluator**

- **Goal**: Evaluate the distribution and appropriateness of search query priorities
- **What it measures**:
  - Priority distribution balance (HIGH/MEDIUM/LOW)
  - Alignment between query priorities and objective importance
  - Critical query prioritization
- **Why it matters**: Ensures efficient research execution order
- **Scoring approach**: Statistical analysis of distribution and alignment

##### **ClarificationIntegrationEvaluator**

- **Goal**: Measure how well clarification responses are integrated into transformation
- **What it measures**:
  - Coverage of user-provided answers in transformation
  - Preservation of clarified intent
  - Handling of skipped or partial responses
- **Why it matters**: Ensures user clarifications are properly utilized
- **Scoring approach**: Term coverage and ambiguity resolution analysis

##### **QueryDecompositionEvaluator**

- **Goal**: Evaluate the quality of query decomposition into sub-components
- **What it measures**:
  - Hierarchical structure quality (PRIMARY/SECONDARY/TERTIARY)
  - Component independence
  - Coverage completeness of original query
- **Why it matters**: Ensures systematic and comprehensive research approach
- **Scoring approach**: Structural analysis and term coverage

##### **SupportingQuestionsEvaluator**

- **Goal**: Evaluate the quality of supporting questions in research objectives
- **What it measures**:
  - Question relevance to objectives
  - Question specificity and clarity
  - Question diversity
- **Why it matters**: Ensures thorough exploration of research topics
- **Scoring approach**: Relevance, specificity, and diversity metrics

##### **SuccessCriteriaMeasurabilityEvaluator**

- **Goal**: Evaluate the measurability of success criteria
- **What it measures**:
  - Presence of quantifiable metrics
  - Clarity of completion indicators
  - Achievability assessment
- **Why it matters**: Ensures research goals are measurable and achievable
- **Scoring approach**: Pattern matching for quantifiable terms and achievability

##### **TemporalGeographicScopeEvaluator**

- **Goal**: Evaluate appropriateness of temporal and geographic scope definitions
- **What it measures**:
  - Temporal boundaries relevance
  - Geographic scope necessity
  - Scope constraint consistency
- **Why it matters**: Ensures appropriate research boundaries
- **Scoring approach**: Keyword detection and scope-objective alignment

##### **SearchSourceSelectionEvaluator**

- **Goal**: Evaluate the appropriateness of search source selections
- **What it measures**:
  - Source diversity appropriateness
  - Source-query type alignment
  - Domain-specific source usage
- **Why it matters**: Ensures reliable and relevant information sources
- **Scoring approach**: Domain mapping and source-type alignment

##### **ConfidenceCalibrationEvaluator**

- **Goal**: Evaluate the calibration of confidence scores
- **What it measures**:
  - Confidence vs. assumption count correlation
  - Confidence vs. gap count correlation
  - Confidence vs. complexity correlation
- **Why it matters**: Ensures realistic confidence assessment
- **Scoring approach**: Statistical correlation analysis

##### **ExecutionStrategyEvaluator**

- **Goal**: Evaluate execution strategy selection appropriateness
- **What it measures**:
  - Strategy appropriateness for query batch
  - Dependency handling in HIERARCHICAL mode
  - Parallelization efficiency
- **Why it matters**: Ensures optimal query execution approach
- **Scoring approach**: Strategy-context alignment and efficiency analysis

### 3. Dataset Loading

Dataset loading is now integrated directly into the evaluation runners:

- **`run_query_transformation_eval.py`**: Contains `load_dataset_from_yaml()` method
- **`run_clarification_eval.py`**: Contains dataset loading logic
- Both runners support category filtering for selective evaluation

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
# Run evaluation with specific categories
uv run python tests/evals/run_query_transformation_eval.py --categories golden_standard

# Run enhanced evaluator tests
uv run python tests/evals/run_query_transformation_eval.py --categories enhanced_evaluator

# In code, load specific categories
evaluator = QueryTransformationEvaluator()
dataset = evaluator.load_dataset_from_yaml(categories=["technical", "enhanced_evaluator"])
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

8. **Enhanced Evaluator Cases** (8 cases)
   - Test Assumption Quality
   - Test Temporal Scope
   - Test Clarification Integration
   - Test Query Decomposition
   - Test Medical Domain Sources
   - Test Geographic Scope
   - Test Priority Distribution
   - Comprehensive Enhanced Evaluation

**Total: 30 test cases**

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
