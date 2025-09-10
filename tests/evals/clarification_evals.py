"""
Evaluation framework for ClarificationAgent using Pydantic Evals.

This module provides comprehensive evaluation capabilities for the clarification agent,
including custom evaluators, metrics, and LLM-as-judge patterns.
"""

import asyncio
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import httpx

from pydantic import BaseModel, Field, SecretStr
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_ai import Agent

from src.agents.clarification import ClarificationAgent, ClarifyWithUser
from src.agents.base import ResearchDependencies
from src.models.metadata import ResearchMetadata
from src.models.core import ResearchState, ResearchStage
from src.models.clarification import ClarificationQuestion, ClarificationRequest
from src.models.api_models import APIKeys


class ClarificationInput(BaseModel):
    """Input model for clarification evaluation."""
    query: str = Field(description="User query to evaluate")
    context: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional conversation context"
    )


class ClarificationExpectedOutput(BaseModel):
    """Expected output for clarification evaluation."""
    needs_clarification: bool = Field(description="Whether clarification is needed")
    min_questions: Optional[int] = Field(
        default=None,
        description="Minimum number of questions expected"
    )
    max_questions: Optional[int] = Field(
        default=None,
        description="Maximum number of questions expected"
    )
    dimension_categories: Optional[List[str]] = Field(
        default=None,
        description="Expected dimension categories from 4-framework"
    )
    key_themes: Optional[List[str]] = Field(
        default=None,
        description="Key themes that should appear in clarification questions"
    )
    expected_question_types: Optional[List[str]] = Field(
        default=None,
        description="Expected question types (text, choice, multi_choice)"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Expected confidence score for the clarification decision (0.0-1.0)"
    )
    expected_questions: Optional[List[str]] = Field(
        default=None,
        description="Sample expected clarification questions for golden standard cases"
    )
    max_response_time: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Maximum acceptable response time in seconds"
    )
    domain: Optional[str] = Field(
        default=None,
        description="Domain classification for domain-specific evaluation (technical, scientific, business, etc.)"
    )


class BinaryAccuracyEvaluator(Evaluator):
    """Evaluates binary correctness of clarification decision."""

    def evaluate(self, output: ClarifyWithUser, expected: ClarificationExpectedOutput) -> Dict[str, Any]:
        """Evaluate if clarification decision matches expected."""
        correct = output.needs_clarification == expected.needs_clarification
        return {
            "score": 1.0 if correct else 0.0,
            "correct": correct,
            "predicted": output.needs_clarification,
            "expected": expected.needs_clarification
        }


class DimensionCoverageEvaluator(Evaluator):
    """Evaluates coverage of the 4-dimension framework."""

    DIMENSION_KEYWORDS = {
        "audience_level": ["audience", "level", "technical", "background", "expertise", "beginner", "expert"],
        "scope_focus": ["scope", "focus", "aspect", "specific", "broad", "area", "domain"],
        "source_quality": ["source", "credibility", "academic", "industry", "quality", "reliability"],
        "deliverable": ["deliverable", "format", "output", "report", "summary", "presentation", "depth"]
    }

    def evaluate(self, output: ClarifyWithUser, expected: ClarificationExpectedOutput) -> Dict[str, Any]:
        """Evaluate dimension framework coverage."""
        if not output.needs_clarification or not output.request:
            # If no clarification needed, this evaluator is not applicable
            return {"score": None, "applicable": False}

        # Combine all text for analysis
        questions_text = " ".join([q.question for q in output.request.questions])
        all_text = " ".join([
            questions_text,
            " ".join(output.missing_dimensions),
            output.assessment_reasoning
        ]).lower()

        # Check which dimensions are covered
        covered_dimensions = []
        for dimension, keywords in self.DIMENSION_KEYWORDS.items():
            if any(keyword in all_text for keyword in keywords):
                covered_dimensions.append(dimension)

        # Calculate coverage score
        coverage_score = len(covered_dimensions) / len(self.DIMENSION_KEYWORDS)

        # Check against expected dimensions if provided
        dimension_match_score = 1.0
        if expected.dimension_categories:
            matched = sum(1 for exp_dim in expected.dimension_categories
                         if any(exp_dim in dim for dim in covered_dimensions))
            dimension_match_score = matched / len(expected.dimension_categories) if expected.dimension_categories else 0

        final_score = (coverage_score + dimension_match_score) / 2

        return {
            "score": final_score,
            "covered_dimensions": covered_dimensions,
            "coverage_rate": coverage_score,
            "dimension_match_score": dimension_match_score,
            "total_dimensions": len(self.DIMENSION_KEYWORDS)
        }


class QuestionRelevanceEvaluator(Evaluator):
    """Evaluates relevance and quality of clarification questions."""

    def evaluate(self, output: ClarifyWithUser, expected: ClarificationExpectedOutput) -> Dict[str, Any]:
        """Evaluate question relevance and quality."""
        if not output.needs_clarification or not output.request or not output.request.questions:
            return {"score": None, "applicable": False}

        scores = []
        question_scores = []

        for question in output.request.questions:
            q_scores = []

            # Check if question is not empty and substantial
            question_length_score = min(len(question.question) / 100, 1.0)  # Normalize to 0-1
            q_scores.append(question_length_score)

            # Check if question ends with question mark (basic quality)
            has_question_mark = 1.0 if question.question.strip().endswith("?") else 0.5
            q_scores.append(has_question_mark)

            # Check theme coverage if expected themes provided
            if expected.key_themes:
                question_lower = question.question.lower()
                theme_matches = sum(1 for theme in expected.key_themes
                                  if theme.lower() in question_lower)
                theme_score = theme_matches / len(expected.key_themes) if expected.key_themes else 0
                q_scores.append(theme_score)

            question_scores.append(sum(q_scores) / len(q_scores))

        # Average score across all questions
        avg_question_score = sum(question_scores) / len(question_scores) if question_scores else 0
        scores.append(avg_question_score)

        # Check if reasoning is provided
        reasoning_score = min(len(output.assessment_reasoning) / 100, 1.0) if output.assessment_reasoning else 0
        scores.append(reasoning_score)

        # Check question count expectations
        if expected.min_questions or expected.max_questions:
            count = len(output.request.questions)
            count_score = 1.0
            if expected.min_questions and count < expected.min_questions:
                count_score = count / expected.min_questions
            elif expected.max_questions and count > expected.max_questions:
                count_score = expected.max_questions / count
            scores.append(count_score)

        final_score = sum(scores) / len(scores)

        return {
            "score": final_score,
            "num_questions": len(output.request.questions),
            "has_reasoning": bool(output.assessment_reasoning),
            "question_scores": question_scores,
            "theme_coverage": theme_score if expected.key_themes else None
        }


class ConsistencyEvaluator(Evaluator):
    """Evaluates consistency across multiple runs of the same query."""

    def __init__(self, num_runs: int = 3):
        self.num_runs = num_runs

    async def evaluate_async(self, agent: ClarificationAgent, query: str) -> Dict[str, Any]:
        """Run multiple times and check consistency."""
        results = []

        # Create dependencies
        async with httpx.AsyncClient() as http_client:
            for _ in range(self.num_runs):
                state = ResearchState(
                    request_id=f"consistency-test-{_}",
                    user_query=query
                )
                deps = ResearchDependencies(
                    http_client=http_client,
                    api_keys=APIKeys(
                        openai=SecretStr(openai_key) if (openai_key := os.getenv("OPENAI_API_KEY")) else None
                    ),
                    research_state=state
                )

                result = await agent.agent.run(query, deps=deps)
                results.append(result.output)

        # Check consistency of binary decision
        decisions = [r.needs_clarification for r in results]
        decision_consistency = all(d == decisions[0] for d in decisions)

        # Check consistency of dimensions (if clarification needed)
        dimension_consistency = 1.0
        if decisions[0]:  # If clarification is needed
            all_dimensions = [set(r.missing_dimensions) for r in results]
            if all_dimensions:
                # Calculate Jaccard similarity between dimension sets
                intersection = set.intersection(*all_dimensions) if all_dimensions else set()
                union = set.union(*all_dimensions) if all_dimensions else set()
                dimension_consistency = len(intersection) / len(union) if union else 1.0

        consistency_score = (1.0 if decision_consistency else 0.5) * dimension_consistency

        return {
            "score": consistency_score,
            "decision_consistency": decision_consistency,
            "dimension_consistency": dimension_consistency,
            "num_runs": self.num_runs,
            "all_decisions": decisions
        }


class MultiQuestionEvaluator(Evaluator):
    """Evaluates multi-question clarification capabilities."""

    def evaluate(self, output: ClarifyWithUser, expected: ClarificationExpectedOutput) -> Dict[str, Any]:
        """Evaluate multi-question generation and diversity."""
        if not output.needs_clarification or not output.request:
            return {"score": None, "applicable": False}

        questions = output.request.questions
        scores = []

        # 1. Question count score
        count = len(questions)
        count_score = 1.0
        if expected.min_questions and count < expected.min_questions:
            count_score = count / expected.min_questions
        elif expected.max_questions and count > expected.max_questions:
            count_score = expected.max_questions / count
        scores.append(count_score)

        # 2. Question type diversity
        question_types = set(q.question_type for q in questions)
        expected_types = expected.expected_question_types or ["text", "choice", "multi_choice"]
        type_coverage = len(question_types.intersection(expected_types)) / len(expected_types)
        scores.append(type_coverage)

        # 3. Required vs optional balance
        required_count = sum(1 for q in questions if q.is_required)
        optional_count = len(questions) - required_count
        balance_score = 1.0
        if required_count == 0 or optional_count == 0:
            balance_score = 0.5  # Penalize if all questions are same type
        scores.append(balance_score)

        # 4. Question uniqueness (no duplicate questions)
        unique_questions = len(set(q.question.lower().strip() for q in questions))
        uniqueness_score = unique_questions / len(questions) if questions else 0
        scores.append(uniqueness_score)

        # 5. Question ordering
        ordered_questions = output.request.get_sorted_questions()
        ordering_score = 1.0 if ordered_questions == sorted(questions, key=lambda q: q.order) else 0.5
        scores.append(ordering_score)

        final_score = sum(scores) / len(scores)

        return {
            "score": final_score,
            "num_questions": count,
            "question_types": list(question_types),
            "required_count": required_count,
            "optional_count": optional_count,
            "uniqueness_score": uniqueness_score,
            "type_coverage": type_coverage,
            "balance_score": balance_score
        }


class LLMJudgeEvaluator(Evaluator):
    """Uses an LLM to judge the quality of clarification questions."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.judge_agent = Agent(
            model=model,
            system_prompt="""You are an expert evaluator of clarification questions.
            Evaluate the quality of clarification questions based on:
            1. Relevance to the original query
            2. Identification of key ambiguities
            3. Helpfulness for providing better answers
            4. Clarity and specificity of the question"""
        )

    async def evaluate_async(
        self,
        query: str,
        output: ClarifyWithUser,
        expected: Optional[ClarificationExpectedOutput] = None
    ) -> Dict[str, Any]:
        """Use LLM to judge clarification quality."""

        if not output.needs_clarification or not output.request:
            return {"score": None, "applicable": False}

        questions_text = "\n".join([f"- {q.question} (Type: {q.question_type}, Required: {q.is_required})"
                                    for q in output.request.questions])

        evaluation_prompt = f"""
        Original Query: {query}

        Clarification Response:
        - Questions:\n{questions_text}
        - Missing Dimensions: {', '.join(output.missing_dimensions)}
        - Reasoning: {output.assessment_reasoning}

        Please evaluate this clarification on a scale of 0-10 for:
        1. Relevance (0-10): How relevant is the clarification to the query?
        2. Ambiguity Detection (0-10): How well does it identify the key ambiguities?
        3. Helpfulness (0-10): Would the answer help provide better research?
        4. Clarity (0-10): Is the clarification question clear and specific?

        Provide your evaluation as a JSON object with these scores and a brief explanation.
        """

        result = await self.judge_agent.run(evaluation_prompt)

        # Parse the LLM's evaluation (assuming it returns structured JSON)
        try:
            eval_data = json.loads(result.output) if isinstance(result.output, str) else result.output

            scores = [
                eval_data.get("relevance", 0) / 10,
                eval_data.get("ambiguity_detection", 0) / 10,
                eval_data.get("helpfulness", 0) / 10,
                eval_data.get("clarity", 0) / 10
            ]

            final_score = sum(scores) / len(scores)

            return {
                "score": final_score,
                "relevance": eval_data.get("relevance"),
                "ambiguity_detection": eval_data.get("ambiguity_detection"),
                "helpfulness": eval_data.get("helpfulness"),
                "clarity": eval_data.get("clarity"),
                "explanation": eval_data.get("explanation", "")
            }
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "score": None,
                "error": f"Failed to parse LLM evaluation: {e}"
            }


class MultiJudgeConsensusEvaluator(Evaluator):
    """Enhanced LLM judge evaluator with multi-judge consensus voting."""

    def __init__(
        self,
        models: List[str] = None,
        consensus_threshold: float = 0.6,
        weight_by_confidence: bool = True
    ):
        """Initialize multi-judge evaluator.

        Args:
            models: List of model names to use as judges
            consensus_threshold: Minimum agreement threshold for consensus
            weight_by_confidence: Whether to weight votes by confidence scores
        """
        self.models = models or [
            "openai:gpt-4o-mini",
            "openai:gpt-4o",
            "anthropic:claude-3-haiku-20240307"
        ]
        self.consensus_threshold = consensus_threshold
        self.weight_by_confidence = weight_by_confidence

        # Create judge agents for each model
        self.judges = {}
        for model in self.models:
            self.judges[model] = Agent(
                model=model,
                system_prompt="""You are an expert evaluator of clarification questions.
                Evaluate the quality of clarification questions based on:
                1. Relevance to the original query (0-10)
                2. Identification of key ambiguities (0-10)
                3. Helpfulness for providing better answers (0-10)
                4. Clarity and specificity of the question (0-10)
                5. Confidence in your evaluation (0-10)

                Return a JSON object with numeric scores and brief reasoning."""
            )

    async def evaluate_async(
        self,
        query: str,
        output: ClarifyWithUser,
        expected: Optional[ClarificationExpectedOutput] = None
    ) -> Dict[str, Any]:
        """Use multiple LLM judges with consensus voting."""

        if not output.needs_clarification or not output.request:
            return {"score": None, "applicable": False}

        questions_text = "\n".join([f"- {q.question} (Type: {q.question_type}, Required: {q.is_required})"
                                    for q in output.request.questions])

        evaluation_prompt = f"""
        Original Query: {query}

        Clarification Response:
        - Questions:\n{questions_text}
        - Missing Dimensions: {', '.join(output.missing_dimensions)}
        - Reasoning: {output.assessment_reasoning}

        Evaluate this clarification on a scale of 0-10 for:
        1. Relevance: How relevant is the clarification to the query?
        2. Ambiguity Detection: How well does it identify key ambiguities?
        3. Helpfulness: Would this help provide better research?
        4. Clarity: Are the clarification questions clear and specific?
        5. Confidence: How confident are you in this evaluation?

        Return JSON: {{"relevance": X, "ambiguity_detection": X, "helpfulness": X, "clarity": X, "confidence": X, "reasoning": "brief explanation"}}
        """

        # Collect evaluations from all judges
        judge_evaluations = []
        for model, judge in self.judges.items():
            try:
                result = await judge.run(evaluation_prompt)
                eval_data = json.loads(result.output) if isinstance(result.output, str) else result.output

                evaluation = {
                    "model": model,
                    "relevance": eval_data.get("relevance", 0),
                    "ambiguity_detection": eval_data.get("ambiguity_detection", 0),
                    "helpfulness": eval_data.get("helpfulness", 0),
                    "clarity": eval_data.get("clarity", 0),
                    "confidence": eval_data.get("confidence", 5),
                    "reasoning": eval_data.get("reasoning", ""),
                    "individual_score": sum([
                        eval_data.get("relevance", 0),
                        eval_data.get("ambiguity_detection", 0),
                        eval_data.get("helpfulness", 0),
                        eval_data.get("clarity", 0)
                    ]) / 40  # Normalize to 0-1
                }
                judge_evaluations.append(evaluation)

            except Exception as e:
                # If a judge fails, record the failure
                judge_evaluations.append({
                    "model": model,
                    "error": str(e),
                    "individual_score": None
                })

        # Calculate consensus metrics
        valid_evaluations = [e for e in judge_evaluations if e.get("individual_score") is not None]

        if not valid_evaluations:
            return {"score": None, "error": "All judges failed", "judge_evaluations": judge_evaluations}

        # Calculate weighted or simple average
        if self.weight_by_confidence:
            total_weight = sum(e["confidence"] for e in valid_evaluations)
            if total_weight > 0:
                consensus_score = sum(
                    e["individual_score"] * e["confidence"] for e in valid_evaluations
                ) / total_weight
            else:
                consensus_score = sum(e["individual_score"] for e in valid_evaluations) / len(valid_evaluations)
        else:
            consensus_score = sum(e["individual_score"] for e in valid_evaluations) / len(valid_evaluations)

        # Calculate agreement metrics
        scores = [e["individual_score"] for e in valid_evaluations]
        score_variance = sum((s - consensus_score) ** 2 for s in scores) / len(scores)
        agreement_score = 1 / (1 + score_variance)  # Higher variance = lower agreement

        # Check consensus threshold
        consensus_reached = agreement_score >= self.consensus_threshold

        return {
            "score": consensus_score,
            "consensus_reached": consensus_reached,
            "agreement_score": agreement_score,
            "num_judges": len(valid_evaluations),
            "failed_judges": len(judge_evaluations) - len(valid_evaluations),
            "judge_evaluations": judge_evaluations,
            "score_variance": score_variance,
            "weighted_by_confidence": self.weight_by_confidence
        }


class SemanticSimilarityEvaluator(Evaluator):
    """Evaluates semantic similarity of clarification questions across runs."""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.embeddings_cache = {}

    async def get_embedding(self, text: str) -> List[float]:
        """Get text embedding using OpenAI's embedding model."""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]

        # In a real implementation, you would use OpenAI's embedding API
        # For now, we'll simulate with a simple hash-based approach
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        # Simulate 1536-dimensional embedding
        embedding = [float(ord(c)) / 255.0 for c in hash_obj.hexdigest()[:768]] * 2
        self.embeddings_cache[text] = embedding
        return embedding

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        return dot_product / (magnitude_a * magnitude_b)

    async def evaluate_similarity_to_baseline(
        self,
        output: ClarifyWithUser,
        baseline_questions: List[str]
    ) -> Dict[str, Any]:
        """Evaluate similarity to baseline clarification questions."""

        if not output.needs_clarification or not output.request:
            return {"score": None, "applicable": False}

        current_questions = [q.question for q in output.request.questions]

        # Get embeddings for all questions
        current_embeddings = []
        baseline_embeddings = []

        for question in current_questions:
            embedding = await self.get_embedding(question)
            current_embeddings.append(embedding)

        for question in baseline_questions:
            embedding = await self.get_embedding(question)
            baseline_embeddings.append(embedding)

        # Calculate maximum similarity for each current question to any baseline
        max_similarities = []
        for curr_emb in current_embeddings:
            max_sim = max(
                self.cosine_similarity(curr_emb, base_emb)
                for base_emb in baseline_embeddings
            ) if baseline_embeddings else 0.0
            max_similarities.append(max_sim)

        # Overall similarity score
        avg_similarity = sum(max_similarities) / len(max_similarities) if max_similarities else 0.0
        meets_threshold = avg_similarity >= self.similarity_threshold

        return {
            "score": avg_similarity,
            "meets_threshold": meets_threshold,
            "question_similarities": max_similarities,
            "num_questions": len(current_questions),
            "num_baseline": len(baseline_questions)
        }


class PerformanceBenchmarkEvaluator(Evaluator):
    """Evaluates performance metrics like response time and resource usage."""

    def __init__(self, max_response_time: float = 10.0):
        self.max_response_time = max_response_time
        self.performance_history = []

    def evaluate_performance(
        self,
        response_time: float,
        token_count: int = None,
        memory_usage: float = None
    ) -> Dict[str, Any]:
        """Evaluate performance metrics."""

        # Response time score (1.0 if under max, decreasing exponentially)
        time_score = min(1.0, self.max_response_time / max(response_time, 0.1))

        # Token efficiency (if available)
        token_efficiency = None
        if token_count is not None:
            # Assume reasonable token count is 500-1000 for clarification
            optimal_tokens = 750
            token_efficiency = min(1.0, optimal_tokens / max(token_count, 1))

        # Memory efficiency (if available)
        memory_efficiency = None
        if memory_usage is not None:
            # Assume reasonable memory usage is under 100MB
            optimal_memory = 100
            memory_efficiency = min(1.0, optimal_memory / max(memory_usage, 1))

        # Overall performance score
        scores = [time_score]
        if token_efficiency is not None:
            scores.append(token_efficiency)
        if memory_efficiency is not None:
            scores.append(memory_efficiency)

        performance_score = sum(scores) / len(scores)

        # Record for trend analysis
        performance_data = {
            "response_time": response_time,
            "token_count": token_count,
            "memory_usage": memory_usage,
            "time_score": time_score,
            "token_efficiency": token_efficiency,
            "memory_efficiency": memory_efficiency,
            "overall_score": performance_score
        }
        self.performance_history.append(performance_data)

        return {
            "score": performance_score,
            "response_time": response_time,
            "time_score": time_score,
            "token_efficiency": token_efficiency,
            "memory_efficiency": memory_efficiency,
            "meets_time_threshold": response_time <= self.max_response_time
        }


class RobustnessEvaluator(Evaluator):
    """Evaluates robustness across edge cases and error conditions."""

    def __init__(self):
        self.test_cases = [
            {"type": "empty_query", "input": ""},
            {"type": "single_word", "input": "AI"},
            {"type": "very_long", "input": "What is machine learning? " * 50},
            {"type": "special_chars", "input": "What is 机器学习 café? 🤖"},
            {"type": "code_snippet", "input": "Fix this: ```python\ndef foo(): return x```"},
        ]

    async def evaluate_robustness(
        self,
        agent,
        dependencies
    ) -> Dict[str, Any]:
        """Test agent robustness across various edge cases."""

        results = []

        for test_case in self.test_cases:
            try:
                # Update dependencies for this test
                dependencies.research_state.user_query = test_case["input"]

                # Run agent
                result = await agent.agent.run(test_case["input"], deps=dependencies)

                # Check if result is properly structured
                is_valid = (
                    hasattr(result, 'data') and
                    hasattr(result.data, 'need_clarification') and
                    isinstance(result.data.need_clarification, bool)
                )

                results.append({
                    "test_type": test_case["type"],
                    "success": True,
                    "valid_structure": is_valid,
                    "needs_clarification": result.data.need_clarification if is_valid else None
                })

            except Exception as e:
                results.append({
                    "test_type": test_case["type"],
                    "success": False,
                    "error": str(e),
                    "valid_structure": False
                })

        # Calculate robustness score
        successful_tests = sum(1 for r in results if r["success"] and r["valid_structure"])
        robustness_score = successful_tests / len(results)

        return {
            "score": robustness_score,
            "successful_tests": successful_tests,
            "total_tests": len(results),
            "test_results": results
        }


class DimensionFrameworkEvaluator(Evaluator):
    """Enhanced evaluator for the 4-dimension framework with detailed analysis."""

    ENHANCED_DIMENSION_KEYWORDS = {
        "audience_level": {
            "primary": ["audience", "level", "technical", "background", "expertise"],
            "secondary": ["beginner", "expert", "intermediate", "novice", "advanced"],
            "weight": 1.0
        },
        "scope_focus": {
            "primary": ["scope", "focus", "aspect", "specific", "broad"],
            "secondary": ["area", "domain", "field", "topic", "subject"],
            "weight": 1.2  # Higher weight as it's often most important
        },
        "source_quality": {
            "primary": ["source", "credibility", "academic", "industry", "quality"],
            "secondary": ["reliability", "peer-reviewed", "authoritative", "recent"],
            "weight": 0.8
        },
        "deliverable": {
            "primary": ["deliverable", "format", "output", "report", "summary"],
            "secondary": ["presentation", "document", "analysis", "findings"],
            "weight": 1.0
        }
    }

    def evaluate(self, output: ClarifyWithUser, expected: ClarificationExpectedOutput) -> Dict[str, Any]:
        """Enhanced dimension framework evaluation with weighted scoring."""

        if not output.needs_clarification or not output.request:
            return {"score": None, "applicable": False}

        # Combine all text for analysis
        questions_text = " ".join([q.question for q in output.request.questions])
        all_text = " ".join([
            questions_text,
            " ".join(output.missing_dimensions),
            output.assessment_reasoning
        ]).lower()

        # Detailed dimension analysis
        dimension_analysis = {}
        total_weighted_score = 0
        total_weight = 0

        for dimension, keywords in self.ENHANCED_DIMENSION_KEYWORDS.items():
            primary_matches = sum(1 for kw in keywords["primary"] if kw in all_text)
            secondary_matches = sum(1 for kw in keywords["secondary"] if kw in all_text)

            # Calculate dimension score
            primary_score = min(1.0, primary_matches / len(keywords["primary"]))
            secondary_score = min(1.0, secondary_matches / len(keywords["secondary"]))
            dimension_score = (primary_score * 0.8) + (secondary_score * 0.2)

            # Apply weight
            weighted_score = dimension_score * keywords["weight"]
            total_weighted_score += weighted_score
            total_weight += keywords["weight"]

            dimension_analysis[dimension] = {
                "score": dimension_score,
                "weighted_score": weighted_score,
                "primary_matches": primary_matches,
                "secondary_matches": secondary_matches,
                "keywords_found": [kw for kw in keywords["primary"] + keywords["secondary"] if kw in all_text]
            }

        # Overall framework coverage
        framework_coverage = total_weighted_score / total_weight if total_weight > 0 else 0

        # Check expected dimensions
        expected_coverage = 1.0
        if expected and expected.dimension_categories:
            covered_expected = sum(
                1 for exp_dim in expected.dimension_categories
                if dimension_analysis.get(exp_dim, {}).get("score", 0) > 0
            )
            expected_coverage = covered_expected / len(expected.dimension_categories)

        # Bonus for comprehensive coverage (covering 3+ dimensions)
        comprehensive_bonus = 0.1 if sum(
            1 for analysis in dimension_analysis.values()
            if analysis["score"] > 0.3
        ) >= 3 else 0

        final_score = min(1.0, (framework_coverage + expected_coverage + comprehensive_bonus) / 2)

        return {
            "score": final_score,
            "framework_coverage": framework_coverage,
            "expected_coverage": expected_coverage,
            "comprehensive_bonus": comprehensive_bonus,
            "dimension_analysis": dimension_analysis,
            "dimensions_covered": sum(1 for analysis in dimension_analysis.values() if analysis["score"] > 0),
            "strong_dimensions": [dim for dim, analysis in dimension_analysis.items() if analysis["score"] > 0.5]
        }


def create_clarification_dataset() -> Dataset:
    """Create comprehensive evaluation dataset for clarification agent with golden standards and domain-specific scenarios."""

    # Golden Standard Cases - Expert validated scenarios with clear expected outcomes
    golden_standard_cases = [
        # GOLDEN: Clear queries (should NOT need clarification)
        Case(
            name="golden_bitcoin_price",
            inputs=ClarificationInput(query="What is the current Bitcoin price in USD?"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=False,
                confidence_score=0.9
            ),
            evaluators=[BinaryAccuracyEvaluator(), PerformanceBenchmarkEvaluator()]
        ),
        Case(
            name="golden_specific_code",
            inputs=ClarificationInput(query="Implement quicksort in Python with O(n log n) complexity"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=False,
                confidence_score=0.95
            ),
            evaluators=[BinaryAccuracyEvaluator(), PerformanceBenchmarkEvaluator()]
        ),
        Case(
            name="golden_factual_query",
            inputs=ClarificationInput(query="What year was the iPhone first released?"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=False,
                confidence_score=0.98
            ),
            evaluators=[BinaryAccuracyEvaluator()]
        ),
        Case(
            name="golden_calculation",
            inputs=ClarificationInput(query="Calculate the area of a circle with radius 5 meters"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=False,
                confidence_score=0.95
            ),
            evaluators=[BinaryAccuracyEvaluator()]
        ),

        # GOLDEN: Ambiguous queries (SHOULD need clarification)
        Case(
            name="golden_broad_ai",
            inputs=ClarificationInput(query="What is AI?"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["audience_level", "scope_focus", "deliverable"],
                key_themes=["artificial intelligence", "specific", "aspect", "level"],
                confidence_score=0.9,
                expected_questions=["What aspect of AI?", "For what audience level?", "What format do you need?"]
            ),
            evaluators=[
                BinaryAccuracyEvaluator(),
                DimensionCoverageEvaluator(),
                QuestionRelevanceEvaluator(),
                MultiJudgeConsensusEvaluator()
            ]
        ),
        Case(
            name="golden_ambiguous_python",
            inputs=ClarificationInput(query="Tell me about Python"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["scope_focus"],
                key_themes=["programming", "language", "snake", "specific", "aspect"],
                confidence_score=0.85,
                expected_questions=["Do you mean Python programming language or the snake?"]
            ),
            evaluators=[
                BinaryAccuracyEvaluator(),
                DimensionCoverageEvaluator(),
                QuestionRelevanceEvaluator()
            ]
        ),
        Case(
            name="golden_vague_research",
            inputs=ClarificationInput(query="Research climate change"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["scope_focus", "deliverable", "source_quality", "audience_level"],
                key_themes=["aspect", "focus", "specific", "purpose", "timeframe"],
                confidence_score=0.88,
                expected_questions=["What specific aspect of climate change?", "What type of deliverable?", "What time period?"]
            ),
            evaluators=[
                BinaryAccuracyEvaluator(),
                DimensionCoverageEvaluator(),
                QuestionRelevanceEvaluator(),
                SemanticSimilarityEvaluator()
            ]
        )
    ]

    # Technical Domain-Specific Cases
    technical_cases = [
        Case(
            name="tech_ambiguous_optimization",
            inputs=ClarificationInput(query="How do I optimize my code?"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["scope_focus", "source_quality"],
                key_themes=["programming language", "performance metric", "specific bottleneck"],
                domain="technical"
            ),
            evaluators=[BinaryAccuracyEvaluator(), DimensionFrameworkEvaluator()]
        ),
        Case(
            name="tech_database_design",
            inputs=ClarificationInput(query="Design a database for my application"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["scope_focus", "deliverable", "audience_level"],
                key_themes=["application type", "data volume", "requirements"],
                domain="technical"
            ),
            evaluators=[BinaryAccuracyEvaluator(), DimensionCoverageEvaluator()]
        ),
        Case(
            name="tech_specific_api",
            inputs=ClarificationInput(query="Write a REST API endpoint for user authentication with JWT tokens using FastAPI"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=False,
                confidence_score=0.9,
                domain="technical"
            ),
            evaluators=[BinaryAccuracyEvaluator()]
        ),
        Case(
            name="tech_microservices",
            inputs=ClarificationInput(query="Explain microservices architecture"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["audience_level", "scope_focus", "deliverable"],
                key_themes=["specific aspect", "comparison", "implementation"],
                domain="technical"
            ),
            evaluators=[BinaryAccuracyEvaluator(), QuestionRelevanceEvaluator()]
        )
    ]

    # Scientific Domain-Specific Cases
    scientific_cases = [
        Case(
            name="sci_broad_quantum",
            inputs=ClarificationInput(query="Explain quantum computing"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["audience_level", "scope_focus", "deliverable"],
                key_themes=["specific aspect", "mathematical level", "applications"],
                domain="scientific"
            ),
            evaluators=[BinaryAccuracyEvaluator(), DimensionCoverageEvaluator()]
        ),
        Case(
            name="sci_specific_experiment",
            inputs=ClarificationInput(query="Design a double-blind placebo-controlled trial for testing antidepressant efficacy in adults aged 18-65 with major depressive disorder"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=False,
                confidence_score=0.95,
                domain="scientific"
            ),
            evaluators=[BinaryAccuracyEvaluator()]
        ),
        Case(
            name="sci_climate_research",
            inputs=ClarificationInput(query="Study the impact of ocean acidification"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["scope_focus", "deliverable", "source_quality"],
                key_themes=["specific organisms", "geographic region", "timeframe"],
                domain="scientific"
            ),
            evaluators=[BinaryAccuracyEvaluator(), DimensionFrameworkEvaluator()]
        )
    ]

    # Business Domain-Specific Cases
    business_cases = [
        Case(
            name="biz_market_analysis",
            inputs=ClarificationInput(query="Analyze the market for our product"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["scope_focus", "deliverable", "audience_level"],
                key_themes=["product type", "target market", "geographic scope", "timeframe"],
                domain="business"
            ),
            evaluators=[BinaryAccuracyEvaluator(), DimensionCoverageEvaluator()]
        ),
        Case(
            name="biz_roi_calculation",
            inputs=ClarificationInput(query="Calculate ROI for a $100K marketing campaign that generated $300K in revenue over 6 months"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=False,
                confidence_score=0.9,
                domain="business"
            ),
            evaluators=[BinaryAccuracyEvaluator()]
        ),
        Case(
            name="biz_strategy_consulting",
            inputs=ClarificationInput(query="Help me develop a business strategy"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["scope_focus", "deliverable", "audience_level", "source_quality"],
                key_themes=["industry", "company stage", "specific goals", "timeframe"],
                domain="business"
            ),
            evaluators=[BinaryAccuracyEvaluator(), MultiJudgeConsensusEvaluator()]
        )
    ]

    # Edge Cases and Robustness Testing
    edge_cases = [
        Case(
            name="edge_minimal_query",
            inputs=ClarificationInput(query="?"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                key_themes=["question", "help", "clarify"]
            ),
            evaluators=[BinaryAccuracyEvaluator(), RobustnessEvaluator()]
        ),
        Case(
            name="edge_empty_query",
            inputs=ClarificationInput(query=""),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                key_themes=["empty", "help", "question"]
            ),
            evaluators=[BinaryAccuracyEvaluator(), RobustnessEvaluator()]
        ),
        Case(
            name="edge_very_long_query",
            inputs=ClarificationInput(query="What is machine learning and how does it work and what are the applications and can you tell me about neural networks and deep learning and artificial intelligence and data science and big data and analytics and statistics and algorithms and programming languages like Python and R and frameworks like TensorFlow and PyTorch and scikit-learn? " * 5),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["scope_focus", "deliverable", "audience_level"]
            ),
            evaluators=[BinaryAccuracyEvaluator(), RobustnessEvaluator()]
        ),
        Case(
            name="edge_special_characters",
            inputs=ClarificationInput(query="What is 机器学习 and how does it relate to café? 🤖"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["audience_level", "scope_focus"]
            ),
            evaluators=[BinaryAccuracyEvaluator(), RobustnessEvaluator()]
        ),
        Case(
            name="edge_contradictory",
            inputs=ClarificationInput(query="I need a simple but comprehensive and brief yet detailed explanation of everything about AI but nothing specific"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                dimension_categories=["scope_focus", "deliverable", "audience_level"],
                key_themes=["contradictory", "clarification needed"]
            ),
            evaluators=[BinaryAccuracyEvaluator(), QuestionRelevanceEvaluator()]
        )
    ]

    # Performance Benchmark Cases
    performance_cases = [
        Case(
            name="perf_concurrent_simple",
            inputs=ClarificationInput(query="What is the capital of France?"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=False,
                max_response_time=2.0
            ),
            evaluators=[BinaryAccuracyEvaluator(), PerformanceBenchmarkEvaluator()]
        ),
        Case(
            name="perf_concurrent_complex",
            inputs=ClarificationInput(query="Analyze the geopolitical implications of renewable energy adoption on global trade relationships"),
            expected_output=ClarificationExpectedOutput(
                needs_clarification=True,
                max_response_time=10.0
            ),
            evaluators=[BinaryAccuracyEvaluator(), PerformanceBenchmarkEvaluator(), MultiJudgeConsensusEvaluator()]
        )
    ]

    # Combine all cases
    all_cases = (
        golden_standard_cases +
        technical_cases +
        scientific_cases +
        business_cases +
        edge_cases +
        performance_cases
    )

    return Dataset(
        name="clarification_agent_comprehensive_evaluation",
        cases=all_cases,
        description="Comprehensive evaluation dataset with golden standards, domain-specific scenarios, edge cases, and performance benchmarks for ClarificationAgent"
    )


async def run_clarification_evaluation():
    """Run complete evaluation of clarification agent."""

    # Create agent
    agent = ClarificationAgent()

    # Create dataset
    dataset = create_clarification_dataset()

    # Define the task function that will be evaluated
    async def clarification_task(inputs: ClarificationInput) -> ClarifyWithUser:
        """Task function for evaluation."""
        async with httpx.AsyncClient() as http_client:
            state = ResearchState(
                request_id="eval-test",
                user_id="test-user",
                session_id="test-session",
                user_query=inputs.query,
                current_stage=ResearchStage.CLARIFICATION,
                metadata=ResearchMetadata()
            )
            deps = ResearchDependencies(
                http_client=http_client,
                api_keys=APIKeys(openai=SecretStr(openai_key) if (openai_key := os.getenv("OPENAI_API_KEY")) else None),
                research_state=state
            )

            result = await agent.agent.run(inputs.query, deps=deps)
            return result.output

    # Run evaluation
    report = await dataset.evaluate(clarification_task)

    return report


def generate_evaluation_report(report: Report) -> str:
    """Generate human-readable evaluation report."""

    output = ["=" * 60]
    output.append("CLARIFICATION AGENT EVALUATION REPORT")
    output.append("=" * 60)

    # Overall metrics
    output.append("\nOVERALL METRICS:")
    output.append("-" * 40)

    total_cases = len(report.cases)
    binary_scores = [case.evaluations.get("BinaryAccuracyEvaluator", {}).get("score", 0)
                     for case in report.cases
                     if "BinaryAccuracyEvaluator" in case.evaluations]

    if binary_scores:
        accuracy = sum(binary_scores) / len(binary_scores)
        output.append(f"Binary Accuracy: {accuracy:.2%}")

    # Per-category breakdown
    output.append("\nPER-CATEGORY PERFORMANCE:")
    output.append("-" * 40)

    for case in report.cases:
        output.append(f"\n{case.name}:")
        for evaluator_name, evaluation in case.evaluations.items():
            if evaluation.get("score") is not None:
                output.append(f"  {evaluator_name}: {evaluation['score']:.2f}")
                if "explanation" in evaluation:
                    output.append(f"    {evaluation['explanation']}")

    # Common patterns
    output.append("\nCOMMON PATTERNS:")
    output.append("-" * 40)

    clarification_needed = sum(1 for case in report.cases
                              if case.output and case.output.needs_clarification)
    output.append(f"Cases needing clarification: {clarification_needed}/{total_cases}")

    # Dimension coverage analysis
    all_dimensions = []
    for case in report.cases:
        if case.output and case.output.needs_clarification:
            all_dimensions.extend(case.output.missing_dimensions)

    if all_dimensions:
        from collections import Counter
        dimension_counts = Counter(all_dimensions)
        output.append("\nMost common missing dimensions:")
        for dim, count in dimension_counts.most_common(5):
            output.append(f"  - {dim}: {count} times")

    return "\n".join(output)


if __name__ == "__main__":
    # Run evaluation
    report = asyncio.run(run_clarification_evaluation())
    print(generate_evaluation_report(report))
