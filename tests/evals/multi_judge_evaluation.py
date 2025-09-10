"""Multi-Judge Evaluation System for ClarificationAgent.

This module provides advanced multi-judge consensus evaluation using multiple LLM models
with sophisticated voting mechanisms, confidence weighting, and disagreement analysis.
Implements 2024 best practices for LLM-as-a-Judge evaluation systems.
"""

import asyncio
import json
import os
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_evals import Evaluator

from src.agents.clarification import ClarificationAgent, ClarifyWithUser
from src.agents.base import ResearchDependencies
from src.models.core import ResearchState, ResearchStage
from src.models.metadata import ResearchMetadata
from src.models.api_models import APIKeys
from pydantic import SecretStr


class VotingMethod(Enum):
    """Different voting methods for multi-judge consensus."""
    MAJORITY = "majority"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    EXPERT_WEIGHTED = "expert_weighted"


class JudgeExpertise(Enum):
    """Judge expertise levels for different domains."""
    GENERAL = "general"
    TECHNICAL = "technical"
    SCIENTIFIC = "scientific"
    BUSINESS = "business"
    CREATIVE = "creative"


@dataclass
class JudgeConfiguration:
    """Configuration for a single judge."""
    model: str
    expertise: JudgeExpertise
    weight: float = 1.0
    temperature: float = 0.0
    system_prompt_override: Optional[str] = None


@dataclass
class EvaluationDimension:
    """Represents a single evaluation dimension."""
    name: str
    description: str
    scale: Tuple[int, int] = (0, 10)
    weight: float = 1.0


class JudgmentResult(BaseModel):
    """Result from a single judge evaluation."""
    judge_id: str
    model: str
    expertise: JudgeExpertise
    scores: Dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=5.0, ge=0, le=10)
    reasoning: str = Field(default="")
    execution_time: float = Field(default=0.0)
    success: bool = Field(default=True)
    error_message: Optional[str] = None


class ConsensusResult(BaseModel):
    """Result from multi-judge consensus evaluation."""
    final_score: float
    consensus_reached: bool
    agreement_score: float
    voting_method: VotingMethod
    judge_results: List[JudgmentResult] = Field(default_factory=list)
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    disagreement_analysis: Dict[str, Any] = Field(default_factory=dict)
    execution_metadata: Dict[str, Any] = Field(default_factory=dict)


class AdvancedMultiJudgeEvaluator:
    """Advanced multi-judge evaluation system with sophisticated consensus mechanisms."""

    def __init__(
        self,
        judges: List[JudgeConfiguration] = None,
        dimensions: List[EvaluationDimension] = None,
        voting_method: VotingMethod = VotingMethod.CONFIDENCE_WEIGHTED,
        consensus_threshold: float = 0.7,
        max_disagreement_std: float = 2.0
    ):
        """Initialize the multi-judge evaluator.

        Args:
            judges: List of judge configurations
            dimensions: List of evaluation dimensions
            voting_method: Method for combining judge votes
            consensus_threshold: Minimum agreement for consensus
            max_disagreement_std: Maximum standard deviation for agreement
        """
        self.judges = judges or self._create_default_judges()
        self.dimensions = dimensions or self._create_default_dimensions()
        self.voting_method = voting_method
        self.consensus_threshold = consensus_threshold
        self.max_disagreement_std = max_disagreement_std

        # Initialize judge agents
        self.judge_agents = {}
        for judge in self.judges:
            self.judge_agents[judge.model] = self._create_judge_agent(judge)

    def _create_default_judges(self) -> List[JudgeConfiguration]:
        """Create default judge configuration."""
        return [
            JudgeConfiguration(
                model="openai:gpt-5",
                expertise=JudgeExpertise.GENERAL,
                weight=1.2,  # Higher weight for more capable model
                temperature=0.1
            ),
            JudgeConfiguration(
                model="openai:gpt-5-mini",
                expertise=JudgeExpertise.GENERAL,
                weight=1.0,
                temperature=0.0
            ),
            JudgeConfiguration(
                model="anthropic:claude-3-sonnet-20240229",
                expertise=JudgeExpertise.TECHNICAL,
                weight=1.1,
                temperature=0.1
            ),
            JudgeConfiguration(
                model="anthropic:claude-3-haiku-20240307",
                expertise=JudgeExpertise.GENERAL,
                weight=0.9,
                temperature=0.0
            )
        ]

    def _create_default_dimensions(self) -> List[EvaluationDimension]:
        """Create default evaluation dimensions."""
        return [
            EvaluationDimension(
                name="relevance",
                description="How relevant are the clarification questions to the original query?",
                weight=1.2
            ),
            EvaluationDimension(
                name="ambiguity_detection",
                description="How well does the agent identify key ambiguities that need clarification?",
                weight=1.3  # Highest weight - core functionality
            ),
            EvaluationDimension(
                name="helpfulness",
                description="Would answering these clarification questions lead to better research results?",
                weight=1.1
            ),
            EvaluationDimension(
                name="clarity",
                description="Are the clarification questions clear, specific, and well-formulated?",
                weight=1.0
            ),
            EvaluationDimension(
                name="completeness",
                description="Does the clarification cover all major ambiguities in the query?",
                weight=1.1
            )
        ]

    def _create_judge_agent(self, judge: JudgeConfiguration) -> Agent:
        """Create a judge agent with appropriate system prompt."""

        # Create dimension-specific instructions
        dimension_instructions = []
        for dim in self.dimensions:
            dimension_instructions.append(f"- {dim.name.title()}: {dim.description} (Scale: {dim.scale[0]}-{dim.scale[1]})")

        expertise_context = ""
        if judge.expertise == JudgeExpertise.TECHNICAL:
            expertise_context = "You have particular expertise in technical and programming-related queries. "
        elif judge.expertise == JudgeExpertise.SCIENTIFIC:
            expertise_context = "You have particular expertise in scientific research and academic queries. "
        elif judge.expertise == JudgeExpertise.BUSINESS:
            expertise_context = "You have particular expertise in business and commercial queries. "
        elif judge.expertise == JudgeExpertise.CREATIVE:
            expertise_context = "You have particular expertise in creative and artistic queries. "

        system_prompt = judge.system_prompt_override or f"""You are an expert evaluator of AI-generated clarification questions.

        {expertise_context}Your task is to evaluate the quality of clarification questions based on these dimensions:

        {chr(10).join(dimension_instructions)}

        For each dimension, provide a score on the specified scale and include your confidence level (0-10) in this evaluation.

        Return your evaluation as a JSON object with the following structure:
        {{
            "scores": {{
                "relevance": X,
                "ambiguity_detection": X,
                "helpfulness": X,
                "clarity": X,
                "completeness": X
            }},
            "confidence": X,
            "reasoning": "Brief explanation of your evaluation focusing on key strengths and weaknesses"
        }}

        Be precise, objective, and consider the specific context of each query when evaluating."""

        return Agent(
            model=judge.model,
            system_prompt=system_prompt,
            temperature=judge.temperature
        )

    async def evaluate_clarification(
        self,
        query: str,
        output: ClarifyWithUser,
        context: Optional[Dict[str, Any]] = None
    ) -> ConsensusResult:
        """Perform multi-judge evaluation of clarification output."""

        if not output.needs_clarification or not output.request:
            return ConsensusResult(
                final_score=0.0,
                consensus_reached=True,
                agreement_score=1.0,
                voting_method=self.voting_method,
                execution_metadata={"applicable": False, "reason": "No clarification needed"}
            )

        # Prepare evaluation context
        questions_text = "\n".join([
            f"- {q.question} (Type: {q.question_type}, Required: {q.is_required})"
            for q in output.request.questions
        ])

        evaluation_prompt = f"""
        Original Query: {query}

        Clarification Response:
        Questions:
        {questions_text}

        Missing Dimensions Identified: {', '.join(output.missing_dimensions) if output.missing_dimensions else 'None'}
        Agent's Reasoning: {output.assessment_reasoning if hasattr(output, 'assessment_reasoning') else 'Not provided'}

        Context: {json.dumps(context) if context else 'None'}

        Please evaluate this clarification response according to the dimensions specified in your system prompt.
        """

        # Collect judgments from all judges
        judge_tasks = []
        for judge in self.judges:
            task = self._get_single_judgment(judge, evaluation_prompt)
            judge_tasks.append(task)

        # Execute all judgments concurrently
        import time
        start_time = time.time()
        judgment_results = await asyncio.gather(*judge_tasks, return_exceptions=True)
        execution_time = time.time() - start_time

        # Process judgment results
        valid_judgments = []
        failed_judgments = []

        for i, result in enumerate(judgment_results):
            if isinstance(result, Exception):
                failed_judgments.append(JudgmentResult(
                    judge_id=f"judge_{i}",
                    model=self.judges[i].model,
                    expertise=self.judges[i].expertise,
                    success=False,
                    error_message=str(result)
                ))
            else:
                valid_judgments.append(result)

        # Calculate consensus
        consensus_result = self._calculate_consensus(valid_judgments, failed_judgments)
        consensus_result.execution_metadata = {
            "total_execution_time": execution_time,
            "num_judges": len(self.judges),
            "successful_judges": len(valid_judgments),
            "failed_judges": len(failed_judgments)
        }

        return consensus_result

    async def _get_single_judgment(
        self,
        judge: JudgeConfiguration,
        prompt: str
    ) -> JudgmentResult:
        """Get judgment from a single judge."""
        import time

        start_time = time.time()

        try:
            agent = self.judge_agents[judge.model]
            result = await agent.run(prompt)
            execution_time = time.time() - start_time

            # Parse the result
            if isinstance(result.output, str):
                eval_data = json.loads(result.output)
            else:
                eval_data = result.output

            return JudgmentResult(
                judge_id=judge.model,
                model=judge.model,
                expertise=judge.expertise,
                scores=eval_data.get("scores", {}),
                confidence=eval_data.get("confidence", 5.0),
                reasoning=eval_data.get("reasoning", ""),
                execution_time=execution_time,
                success=True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return JudgmentResult(
                judge_id=judge.model,
                model=judge.model,
                expertise=judge.expertise,
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    def _calculate_consensus(
        self,
        valid_judgments: List[JudgmentResult],
        failed_judgments: List[JudgmentResult]
    ) -> ConsensusResult:
        """Calculate consensus from valid judgments."""

        if not valid_judgments:
            return ConsensusResult(
                final_score=0.0,
                consensus_reached=False,
                agreement_score=0.0,
                voting_method=self.voting_method,
                judge_results=failed_judgments,
                execution_metadata={"error": "No valid judgments received"}
            )

        # Calculate scores for each dimension
        dimension_scores = {}
        overall_scores = []

        for dimension in self.dimensions:
            dim_name = dimension.name
            dim_scores = []
            dim_weights = []

            for judgment in valid_judgments:
                if dim_name in judgment.scores:
                    score = judgment.scores[dim_name]

                    # Apply voting method weighting
                    if self.voting_method == VotingMethod.CONFIDENCE_WEIGHTED:
                        weight = judgment.confidence / 10.0  # Normalize confidence to 0-1
                    elif self.voting_method == VotingMethod.EXPERT_WEIGHTED:
                        judge_config = next(j for j in self.judges if j.model == judgment.model)
                        weight = judge_config.weight
                    else:
                        weight = 1.0

                    dim_scores.append(score)
                    dim_weights.append(weight)

            if dim_scores:
                if self.voting_method == VotingMethod.MAJORITY:
                    # For majority voting on numeric scores, use median
                    dimension_score = statistics.median(dim_scores)
                else:
                    # Weighted average
                    dimension_score = sum(s * w for s, w in zip(dim_scores, dim_weights)) / sum(dim_weights)

                dimension_scores[dim_name] = dimension_score

                # Add to overall score with dimension weight
                overall_scores.append(dimension_score * dimension.weight)

        # Calculate final score
        total_weight = sum(dim.weight for dim in self.dimensions if dim.name in dimension_scores)
        final_score = sum(overall_scores) / total_weight if total_weight > 0 else 0.0

        # Normalize to 0-1 scale
        max_possible_score = max(dim.scale[1] for dim in self.dimensions)
        final_score_normalized = final_score / max_possible_score

        # Calculate agreement metrics
        agreement_analysis = self._analyze_agreement(valid_judgments, dimension_scores)

        return ConsensusResult(
            final_score=final_score_normalized,
            consensus_reached=agreement_analysis["consensus_reached"],
            agreement_score=agreement_analysis["agreement_score"],
            voting_method=self.voting_method,
            judge_results=valid_judgments + failed_judgments,
            dimension_scores=dimension_scores,
            disagreement_analysis=agreement_analysis
        )

    def _analyze_agreement(
        self,
        valid_judgments: List[JudgmentResult],
        dimension_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze agreement between judges."""

        if len(valid_judgments) < 2:
            return {
                "consensus_reached": True,  # Single judge or no disagreement possible
                "agreement_score": 1.0,
                "score_variance": 0.0,
                "dimension_agreements": {}
            }

        # Calculate variance for each dimension
        dimension_variances = {}
        dimension_agreements = {}

        for dimension in self.dimensions:
            dim_name = dimension.name
            dim_scores = []

            for judgment in valid_judgments:
                if dim_name in judgment.scores:
                    dim_scores.append(judgment.scores[dim_name])

            if len(dim_scores) > 1:
                variance = statistics.variance(dim_scores)
                dimension_variances[dim_name] = variance

                # Agreement score based on inverse of variance
                max_variance = (dimension.scale[1] - dimension.scale[0]) ** 2 / 4  # Max possible variance
                agreement = 1.0 - min(variance / max_variance, 1.0)
                dimension_agreements[dim_name] = agreement

        # Overall agreement score
        if dimension_agreements:
            overall_agreement = statistics.mean(dimension_agreements.values())
            overall_variance = statistics.mean(dimension_variances.values())
        else:
            overall_agreement = 1.0
            overall_variance = 0.0

        # Determine if consensus is reached
        consensus_reached = (
            overall_agreement >= self.consensus_threshold and
            overall_variance <= self.max_disagreement_std ** 2
        )

        return {
            "consensus_reached": consensus_reached,
            "agreement_score": overall_agreement,
            "score_variance": overall_variance,
            "dimension_agreements": dimension_agreements,
            "dimension_variances": dimension_variances
        }


class PairwiseComparisonEvaluator:
    """Evaluates two clarification responses using pairwise comparison with multiple judges."""

    def __init__(self, multi_judge_evaluator: AdvancedMultiJudgeEvaluator):
        self.multi_judge_evaluator = multi_judge_evaluator

    async def compare_clarifications(
        self,
        query: str,
        response_a: ClarifyWithUser,
        response_b: ClarifyWithUser,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compare two clarification responses using multi-judge consensus."""

        # Create comparison prompt for judges
        questions_a = "\n".join([f"- {q.question}" for q in response_a.request.questions]) if response_a.request else "No clarification questions"
        questions_b = "\n".join([f"- {q.question}" for q in response_b.request.questions]) if response_b.request else "No clarification questions"

        comparison_prompt = f"""
        Original Query: {query}

        Response A:
        Needs Clarification: {response_a.needs_clarification}
        Questions: {questions_a}

        Response B:
        Needs Clarification: {response_b.needs_clarification}
        Questions: {questions_b}

        Please compare these two responses and determine which is better overall.
        Consider relevance, completeness, clarity, and helpfulness.

        Return: {{"winner": "A" or "B" or "tie", "confidence": 0-10, "reasoning": "explanation"}}
        """

        # Get comparisons from all judges
        comparison_tasks = []
        for judge in self.multi_judge_evaluator.judges:
            agent = self.multi_judge_evaluator.judge_agents[judge.model]
            comparison_tasks.append(agent.run(comparison_prompt))

        comparison_results = await asyncio.gather(*comparison_tasks, return_exceptions=True)

        # Analyze comparison results
        votes = {"A": 0, "B": 0, "tie": 0}
        judge_comparisons = []

        for i, result in enumerate(comparison_results):
            if not isinstance(result, Exception):
                try:
                    if isinstance(result.output, str):
                        comp_data = json.loads(result.output)
                    else:
                        comp_data = result.output

                    winner = comp_data.get("winner", "tie").upper()
                    if winner in votes:
                        votes[winner] += 1

                    judge_comparisons.append({
                        "judge": self.multi_judge_evaluator.judges[i].model,
                        "winner": winner,
                        "confidence": comp_data.get("confidence", 5),
                        "reasoning": comp_data.get("reasoning", "")
                    })

                except Exception:
                    votes["tie"] += 1  # Default to tie on parse error

        # Determine overall winner
        max_votes = max(votes.values())
        winners = [k for k, v in votes.items() if v == max_votes]

        if len(winners) == 1:
            overall_winner = winners[0]
            confidence = votes[overall_winner] / sum(votes.values())
        else:
            overall_winner = "tie"
            confidence = 0.5

        return {
            "winner": overall_winner,
            "confidence": confidence,
            "vote_breakdown": votes,
            "judge_comparisons": judge_comparisons,
            "total_judges": len(self.multi_judge_evaluator.judges)
        }


async def run_advanced_evaluation_demo():
    """Demonstration of the advanced multi-judge evaluation system."""

    # Create evaluator
    evaluator = AdvancedMultiJudgeEvaluator()

    # Mock clarification output for demo
    from src.models.clarification import ClarificationRequest, ClarificationQuestion

    mock_output = ClarifyWithUser(
        needs_clarification=True,
        request=ClarificationRequest(
            questions=[
                ClarificationQuestion(
                    question="What specific aspect of machine learning are you most interested in?",
                    question_type="choice",
                    choices=["supervised learning", "unsupervised learning", "reinforcement learning"],
                    is_required=True
                ),
                ClarificationQuestion(
                    question="What is your technical background level?",
                    question_type="choice",
                    choices=["beginner", "intermediate", "advanced"],
                    is_required=True
                )
            ]
        ),
        missing_dimensions=["scope", "audience_level"],
        assessment_reasoning="The query is too broad and doesn't specify the user's background or specific interests."
    )

    # Run evaluation
    result = await evaluator.evaluate_clarification(
        query="Tell me about machine learning",
        output=mock_output,
        context={"domain": "educational"}
    )

    print("=== Advanced Multi-Judge Evaluation Results ===")
    print(f"Final Score: {result.final_score:.3f}")
    print(f"Consensus Reached: {result.consensus_reached}")
    print(f"Agreement Score: {result.agreement_score:.3f}")
    print(f"Voting Method: {result.voting_method.value}")

    print("\nDimension Scores:")
    for dim, score in result.dimension_scores.items():
        print(f"  {dim}: {score:.2f}")

    print("\nJudge Results:")
    for judge_result in result.judge_results:
        if judge_result.success:
            print(f"  {judge_result.model}: Score {judge_result.confidence:.1f}, Confidence {judge_result.confidence:.1f}")
        else:
            print(f"  {judge_result.model}: FAILED - {judge_result.error_message}")

    print(f"\nExecution Time: {result.execution_metadata.get('total_execution_time', 0):.2f}s")


if __name__ == "__main__":
    asyncio.run(run_advanced_evaluation_demo())
