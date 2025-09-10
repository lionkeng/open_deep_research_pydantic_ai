# Generalized Multi-Judge Evaluation Framework

## Overview

The Multi-Judge Evaluation Framework provides a sophisticated, agent-agnostic system for evaluating AI agent outputs using multiple LLM judges with consensus mechanisms. This framework has been generalized from the original ClarificationAgent-specific implementation to support any agent type.

## Architecture

### Core Components

1. **`base_multi_judge.py`** - The generalized framework
   - `BaseMultiJudgeEvaluator`: Main evaluator class with consensus logic
   - `AgentEvaluationAdapter`: Protocol for agent-specific adapters
   - `VotingMethod`: Different consensus mechanisms
   - `JudgeConfiguration`: Individual judge setup
   - `ConsensusResult`: Evaluation results with agreement analysis

2. **Agent-Specific Adapters**
   - `clarification_multi_judge_adapter.py`: Adapter for ClarificationAgent
   - `query_transformation_multi_judge_adapter.py`: Adapter for QueryTransformationAgent

### Key Features

- **Multiple Voting Methods**:
  - `MAJORITY`: Median-based consensus for numeric scores
  - `WEIGHTED_AVERAGE`: Simple weighted averaging
  - `CONFIDENCE_WEIGHTED`: Weights by judge confidence scores
  - `EXPERT_WEIGHTED`: Weights by judge expertise configuration

- **Judge Expertise Levels**:
  - `GENERAL`: Broad evaluation perspective
  - `TECHNICAL`: Programming and technical focus
  - `SCIENTIFIC`: Research methodology expertise
  - `BUSINESS`: Commercial and practical focus
  - `CREATIVE`: Innovation and exploration focus

- **Consensus Analysis**:
  - Agreement scoring across dimensions
  - Variance analysis for disagreement detection
  - Configurable consensus thresholds
  - Dimension-specific weighting

## Usage

### Basic Evaluation

```python
from tests.evals.base_multi_judge import BaseMultiJudgeEvaluator, VotingMethod
from tests.evals.clarification_multi_judge_adapter import ClarificationMultiJudgeAdapter

# Create adapter for your agent type
adapter = ClarificationMultiJudgeAdapter()

# Initialize evaluator
evaluator = BaseMultiJudgeEvaluator(
    adapter=adapter,
    voting_method=VotingMethod.CONFIDENCE_WEIGHTED,
    consensus_threshold=0.7,
    max_disagreement_std=2.0
)

# Evaluate agent output
consensus_result = await evaluator.evaluate(
    input=user_query,
    output=agent_output,
    context={"domain": "technical"}
)

# Access results
print(f"Final Score: {consensus_result.final_score}")
print(f"Consensus Reached: {consensus_result.consensus_reached}")
print(f"Dimension Scores: {consensus_result.dimension_scores}")
```

### Custom Judge Configuration

```python
from tests.evals.base_multi_judge import JudgeConfiguration, JudgeExpertise

custom_judges = [
    JudgeConfiguration(
        model="openai:gpt-4o",
        expertise=JudgeExpertise.SCIENTIFIC,
        weight=1.5,  # Higher weight for this judge
        temperature=0.1
    ),
    JudgeConfiguration(
        model="anthropic:claude-3-sonnet-20240229",
        expertise=JudgeExpertise.TECHNICAL,
        weight=1.2,
        temperature=0.0
    )
]

evaluator = BaseMultiJudgeEvaluator(
    adapter=adapter,
    judges=custom_judges,
    voting_method=VotingMethod.EXPERT_WEIGHTED
)
```

### Pairwise Comparison

```python
# Compare two outputs
comparison_result = await evaluator.compare_outputs(
    input=query,
    output_a=first_output,
    output_b=second_output,
    context={"test_type": "A/B"}
)

print(f"Winner: {comparison_result['winner']}")
print(f"Vote Breakdown: {comparison_result['vote_breakdown']}")
```

## Creating New Agent Adapters

To add support for a new agent type, implement the `AgentEvaluationAdapter` protocol:

```python
from typing import List, Dict, Any, Optional
from tests.evals.base_multi_judge import AgentEvaluationAdapter, EvaluationDimension

class MyAgentAdapter(AgentEvaluationAdapter[InputType, OutputType]):

    def get_evaluation_dimensions(self) -> List[EvaluationDimension]:
        """Define evaluation dimensions for your agent."""
        return [
            EvaluationDimension(
                name="accuracy",
                description="How accurate is the agent's output?",
                weight=1.3
            ),
            # Add more dimensions...
        ]

    def format_output_for_evaluation(self, output: OutputType) -> str:
        """Convert output to string for evaluation."""
        # Format your agent's output
        return formatted_string

    def create_evaluation_prompt(
        self,
        input: InputType,
        output: OutputType,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create the evaluation prompt for judges."""
        # Build evaluation prompt
        return prompt

    def is_output_valid(self, output: OutputType) -> bool:
        """Check if output is valid for evaluation."""
        # Validate output
        return True

    def get_expertise_context(self, expertise: JudgeExpertise) -> str:
        """Provide expertise-specific context."""
        # Return context based on expertise
        return context_string
```

## Running the Demo

```bash
# Ensure API keys are set
export OPENAI_API_KEY='your-key-here'
export ANTHROPIC_API_KEY='your-key-here'  # Optional

# Run the comprehensive demo
uv run python tests/evals/run_multi_judge_demo.py
```

## Evaluation Dimensions

### ClarificationAgent Dimensions
- `relevance`: Question relevance to query
- `ambiguity_detection`: Identification of key ambiguities
- `helpfulness`: Utility for better research
- `clarity`: Question clarity and specificity
- `completeness`: Coverage of all ambiguities
- `framework_adherence`: Following 4-dimension framework

### QueryTransformationAgent Dimensions
- `search_query_relevance`: Search query quality
- `objective_coverage`: Research objective completeness
- `plan_coherence`: Logical structure
- `query_diversity`: Query comprehensiveness
- `methodology_quality`: Research approach
- `transformation_completeness`: Aspect coverage
- `actionability`: Executability of plan

## Benefits

1. **Consistency**: Same evaluation framework for all agents
2. **Flexibility**: Multiple voting methods and configurations
3. **Reliability**: Multi-judge consensus reduces individual judge bias
4. **Transparency**: Detailed disagreement analysis
5. **Extensibility**: Easy to add new agent types
6. **Comparability**: Pairwise comparison for A/B testing

## Integration with Existing Evaluations

The multi-judge framework complements existing pydantic-ai evaluations:

```python
# Use alongside pydantic-ai evaluators
from pydantic_evals import Dataset, Case

# Traditional pydantic-ai evaluation
dataset = Dataset(cases=[...])
results = await evaluate_dataset(dataset)

# Add multi-judge consensus for critical cases
for case in critical_cases:
    consensus = await multi_judge_evaluator.evaluate(
        input=case.inputs,
        output=case.output
    )
    # Combine results...
```

## Best Practices

1. **Judge Selection**: Use diverse models for better consensus
2. **Dimension Weights**: Adjust weights based on evaluation priorities
3. **Voting Method**: Choose based on your confidence in judge capabilities
4. **Consensus Threshold**: Higher thresholds for critical evaluations
5. **Context Provision**: Include relevant context for better evaluation

## Future Enhancements

- [ ] Add support for streaming evaluations
- [ ] Implement caching for repeated evaluations
- [ ] Add visualization for disagreement patterns
- [ ] Support for custom voting algorithms
- [ ] Integration with evaluation databases
- [ ] Automated judge calibration
