"""Research Executor evaluation utilities.

This module wraps Pydantic-Evals to score `ResearchResults` objects using a mix of
rule-based and LLM-as-judge evaluators. It mirrors the production quality metrics,
covering relevance, completeness, and synthesis quality.
"""

import asyncio
import concurrent.futures
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_ai import Agent
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

from agents.base import ResearchDependencies
from agents.research_executor import ResearchExecutorAgent
from models.api_models import APIKeys
from models.core import ResearchStage, ResearchState
from models.metadata import ResearchMetadata
from models.research_executor import HierarchicalFinding, ResearchResults, ResearchSource

# Constants for evaluation scoring weights
LLM_RELEVANCE_WEIGHT = 0.8
CATEGORY_COVERAGE_WEIGHT = 0.2
FALLBACK_CONFIDENCE_WEIGHT = 0.5
FALLBACK_EVIDENCE_WEIGHT = 0.5
FALLBACK_SCORE_MULTIPLIER = 0.7
INSIGHT_DEPTH_WEIGHT = 0.5
INSIGHT_ACTIONABILITY_WEIGHT = 0.5
FALLBACK_INSIGHT_MULTIPLIER = 0.6


class ResearchExecutorInput(BaseModel):
    """Input model for research executor evaluation."""

    query: str = Field(description="Research query to execute")
    research_brief: str | None = Field(
        default=None, description="Research plan or brief for execution"
    )
    methodology: str | None = Field(default=None, description="Research methodology to follow")
    domain: str | None = Field(
        default=None,
        description="Domain classification (technical, scientific, business, medical, etc.)",
    )
    complexity: str = Field(
        default="medium", description="Expected complexity level: simple, medium, complex"
    )
    temporal_relevance: bool | None = Field(
        default=None, description="Whether temporal relevance is important"
    )


class ResearchExecutorExpectedOutput(BaseModel):
    """Expected output for research executor evaluation."""

    model_config = ConfigDict(populate_by_name=True)

    min_findings: int | None = Field(
        default=None, description="Minimum number of findings expected"
    )
    max_findings: int | None = Field(
        default=None, description="Maximum number of findings expected"
    )
    min_sources: int | None = Field(default=None, description="Minimum number of sources expected")
    max_sources: int | None = Field(default=None, description="Maximum number of sources expected")
    expected_categories: list[str] | None = Field(
        default=None, description="Expected finding categories"
    )
    expected_insights_themes: list[str] | None = Field(
        default=None, description="Expected themes in key insights"
    )
    expected_gaps: list[str] | None = Field(
        default=None, description="Expected data gaps to be identified"
    )
    min_overall_quality_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum expected overall quality score",
        alias="min_quality_score",
    )
    source_credibility_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum average source credibility"
    )
    confidence_calibration: str | None = Field(
        default=None,
        description=(
            "Expected confidence calibration: well-calibrated, overconfident, "
            "underconfident"
        ),
    )
    max_response_time: float | None = Field(
        default=None, gt=0.0, description="Maximum acceptable response time in seconds"
    )


class FindingsRelevanceEvaluator(Evaluator):
    """Evaluates relevance of research findings to the original query using LLM as judge."""

    def __init__(self, model: str = "openai:gpt-4o-mini"):
        """Initialize the evaluator with an LLM judge.

        Args:
            model: The model to use for evaluation (default: gpt-4o-mini for cost efficiency)
        """
        self.model = model
        self.judge_agent = Agent(
            model=model,
            system_prompt="""You are an expert evaluator of research relevance.

Your task is to evaluate how relevant and on-topic research findings are to the original query.

For each finding, consider:
1. **Direct Relevance**: Does the finding directly address the query?
2. **Semantic Relevance**: Is the finding about the same topic/domain even if using different terms?
3. **Contextual Value**: Does it provide important context or background?
4. **Evidence Quality**: Is the finding well-supported with evidence?

Score each finding from 0.0 to 1.0 where:
- 1.0 = Directly answers or addresses the query
- 0.8-0.9 = Highly relevant, same topic/domain
- 0.6-0.7 = Relevant context or related information
- 0.4-0.5 = Tangentially related
- 0.2-0.3 = Minimal relevance
- 0.0-0.1 = Off-topic or irrelevant

Return a JSON object with:
{
    "findings_scores": [list of scores for each finding],
    "overall_relevance": average score,
    "reasoning": "brief explanation of the scoring"
}""",
        )

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate findings relevance using LLM judgment."""
        output = ctx.output
        expected = ctx.expected_output or ResearchExecutorExpectedOutput()

        if not output.findings:
            return 0.0

        # Use asyncio to execute the async evaluation

        async def run_evaluation():
            # Format findings for evaluation
            formatted_findings = []
            for index, finding in enumerate(output.findings):
                evidence_summary = (
                    ", ".join(finding.supporting_evidence)
                    if finding.supporting_evidence
                    else "None"
                )
                confidence_summary = (
                    f"{finding.confidence_score:.2f}"
                    if finding.confidence_score is not None
                    else "N/A"
                )
                formatted_findings.append(
                    "\n".join(
                        [
                            f"Finding {index + 1}:",
                            f"- Content: {finding.finding}",
                            f"- Evidence: {evidence_summary}",
                            f"- Confidence: {confidence_summary}",
                            f"- Category: {finding.category or 'uncategorized'}",
                        ]
                    )
                )

            findings_text = "\n\n".join(formatted_findings)

            evaluation_prompt = f"""
Query: {output.query}

Research Findings:
{findings_text}

Please evaluate the relevance of each finding to the original query.
Consider semantic similarity, not just exact word matches.
"""

            try:
                result = await self.judge_agent.run(evaluation_prompt)

                # Parse the LLM's evaluation
                if isinstance(result.output, str):
                    import json

                    eval_data = json.loads(result.output)
                else:
                    eval_data = result.output

                # Get the overall relevance score
                llm_relevance = eval_data.get("overall_relevance", 0.5)

                # Check expected categories if provided
                category_coverage = 1.0
                if expected.expected_categories:
                    finding_categories = {
                        f.category for f in output.findings if f.category
                    }
                    covered_categories = finding_categories.intersection(
                        set(expected.expected_categories)
                    )
                    category_coverage = len(covered_categories) / len(expected.expected_categories)

                # Combine LLM relevance score with category coverage
                final_score = (
                    llm_relevance * LLM_RELEVANCE_WEIGHT
                    + category_coverage * CATEGORY_COVERAGE_WEIGHT
                )

                return final_score

            except Exception:
                # Fallback to a simple heuristic if LLM evaluation fails
                # Check if findings have good confidence and evidence
                avg_confidence = sum(f.confidence_score or 0.5 for f in output.findings) / len(
                    output.findings
                )
                has_evidence = sum(1 for f in output.findings if f.supporting_evidence) / len(
                    output.findings
                )
                return (
                    avg_confidence * FALLBACK_CONFIDENCE_WEIGHT
                    + has_evidence * FALLBACK_EVIDENCE_WEIGHT
                ) * FALLBACK_SCORE_MULTIPLIER  # Conservative score

        # Run the async evaluation
        # Check if there's already an event loop running
        try:
            asyncio.get_running_loop()

            # If we're in an async context, create a new event loop in a thread
            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(run_evaluation())
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_new_loop)
                return future.result()
        except RuntimeError:
            # No event loop running, we can use asyncio.run directly
            return asyncio.run(run_evaluation())


class SourceCredibilityEvaluator(Evaluator):
    """Evaluates the credibility and diversity of research sources."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate source credibility."""
        output = ctx.output
        expected = ctx.expected_output or ResearchExecutorExpectedOutput()

        if not output.sources:
            return 0.0

        # Calculate average source credibility
        credibility_scores = [s.relevance_score for s in output.sources if s.relevance_score]
        avg_credibility = (
            sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.5
        )

        # Check against threshold
        threshold_score = 1.0
        if expected.source_credibility_threshold:
            threshold_score = (
                1.0
                if avg_credibility >= expected.source_credibility_threshold
                else avg_credibility / expected.source_credibility_threshold
            )

        # Evaluate source diversity
        diversity_score = self._evaluate_source_diversity(output.sources)

        # Check source count expectations
        count_score = 1.0
        source_count = len(output.sources)
        if expected.min_sources and source_count < expected.min_sources:
            count_score = source_count / expected.min_sources
        elif expected.max_sources and source_count > expected.max_sources:
            count_score = expected.max_sources / source_count

        # Check for recent sources if temporal relevance matters
        recency_score = 1.0
        if ctx.input.temporal_relevance:
            recency_score = self._evaluate_source_recency(output.sources)

        final_score = (
            avg_credibility * 0.3
            + threshold_score * 0.2
            + diversity_score * 0.2
            + count_score * 0.15
            + recency_score * 0.15
        )

        return final_score

    def _evaluate_source_diversity(self, sources: list[ResearchSource]) -> float:
        """Evaluate diversity of sources."""
        if not sources:
            return 0.0

        # Check URL diversity (different domains)
        domains = set()
        for source in sources:
            if source.url:
                # Extract domain from URL
                parts = source.url.split("/")
                if len(parts) > 2:
                    domain = parts[2]
                    domains.add(domain)

        # More unique domains = better diversity
        diversity_ratio = len(domains) / len(sources) if sources else 0
        return min(1.0, diversity_ratio * 1.5)  # Boost slightly as perfect diversity is rare

    def _evaluate_source_recency(self, sources: list[ResearchSource]) -> float:
        """Evaluate recency of sources."""
        recent_sources = 0
        total_dated_sources = 0

        for source in sources:
            if source.date:
                total_dated_sources += 1
                # Check if source is within last 2 years
                if source.date > datetime.now(UTC) - timedelta(days=730):
                    recent_sources += 1

        if total_dated_sources == 0:
            return 0.5  # Neutral score if no dates available

        return recent_sources / total_dated_sources


class InsightQualityEvaluator(Evaluator):
    """Evaluates the quality, depth, and actionability of key insights using LLM as judge."""

    def __init__(self, model: str = "openai:gpt-4o-mini"):
        """Initialize the evaluator with an LLM judge.

        Args:
            model: The model to use for evaluation (default: gpt-4o-mini for cost efficiency)
        """
        self.model = model
        self.judge_agent = Agent(
            model=model,
            system_prompt="""You are an expert evaluator of research insights and synthesis quality.

Your task is to evaluate the quality of key insights extracted from research findings.

For each insight, consider:
1. **Depth and Comprehensiveness**: Does it go beyond surface-level observations?
2. **Actionability**: Does it provide clear guidance or recommendations?
3. **Synthesis Quality**: Does it effectively combine multiple findings into coherent understanding?
4. **Strategic Value**: Does it offer strategic perspective or forward-looking analysis?
5. **Clarity and Specificity**: Is it clear, specific, and well-articulated?
6. **Evidence Grounding**: Is it well-supported by the research findings?

Score each insight from 0.0 to 1.0 where:
- 1.0 = Exceptional insight: Deep, actionable, strategic, well-synthesized
- 0.8-0.9 = Strong insight: Clear value, actionable, good synthesis
- 0.6-0.7 = Good insight: Useful but could be deeper or more actionable
- 0.4-0.5 = Basic insight: Surface-level, limited actionability
- 0.2-0.3 = Weak insight: Too obvious or generic
- 0.0-0.1 = Poor insight: No real value or insight provided

IMPORTANT: This is different from relevance evaluation. Focus on QUALITY not RELEVANCE.
Even if an insight is relevant to the query, it should score low if it lacks depth,
actionability, or synthesis.

Return a JSON object with:
{
    "insight_scores": [list of scores for each insight],
    "overall_quality": average score,
    "strengths": "what makes the insights valuable",
    "weaknesses": "what could be improved",
    "reasoning": "brief explanation of the scoring"
}""",
        )

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate insight quality using LLM judgment."""
        output = ctx.output
        expected = ctx.expected_output or ResearchExecutorExpectedOutput()

        if not output.key_insights:
            return 0.0

        # Use asyncio to execute the async evaluation

        async def run_evaluation():
            # Format insights for evaluation
            insights_text = "\n\n".join(
                [f"Insight {i + 1}: {insight}" for i, insight in enumerate(output.key_insights)]
            )

            # Also include findings summary for context
            findings_summary = (
                f"Based on {len(output.findings)} research findings"
                if output.findings
                else "No findings available"
            )
            sources_summary = (
                f"From {len(output.sources)} sources" if output.sources else "No sources cited"
            )

            evaluation_prompt = f"""
Query: {output.query}

Research Context:
- {findings_summary}
- {sources_summary}

Key Insights to Evaluate:
{insights_text}

Please evaluate the quality of these insights.
Focus on depth, actionability, synthesis quality, and strategic value.
Remember: This is about QUALITY not RELEVANCE. A relevant but shallow insight should score low.
"""

            try:
                result = await self.judge_agent.run(evaluation_prompt)

                # Parse the LLM's evaluation
                if isinstance(result.output, str):
                    import json

                    eval_data = json.loads(result.output)
                else:
                    eval_data = result.output

                # Get the overall quality score
                llm_quality = eval_data.get("overall_quality", 0.5)

                # Check theme coverage if expected
                theme_bonus = 0.0
                if expected.expected_insights_themes:
                    all_insights_text = " ".join(output.key_insights).lower()
                    theme_coverage = sum(
                        1
                        for theme in expected.expected_insights_themes
                        if theme.lower() in all_insights_text
                    )
                    theme_bonus = (theme_coverage / len(expected.expected_insights_themes)) * 0.1

                # Adjust for insight count (too few or too many is suboptimal)
                count_adjustment = 1.0
                insight_count = len(output.key_insights)
                if insight_count < 2:
                    count_adjustment = 0.8  # Too few insights
                elif insight_count > 10:
                    count_adjustment = 0.9  # Too many, might lack synthesis

                # Combine scores with adjustments
                final_score = min(1.0, (llm_quality * count_adjustment) + theme_bonus)

                return final_score

            except Exception:
                # Fallback to a simple heuristic if LLM evaluation fails
                # Check for basic quality indicators
                avg_length = sum(len(insight.split()) for insight in output.key_insights) / len(
                    output.key_insights
                )
                length_score = min(1.0, avg_length / 30)  # Prefer insights with ~30 words

                # Check for actionability keywords
                actionable_keywords = ["recommend", "should", "must", "consider", "implement"]
                actionability = sum(
                    1
                    for insight in output.key_insights
                    if any(kw in insight.lower() for kw in actionable_keywords)
                )
                actionability_score = actionability / len(output.key_insights)

                return (
                    length_score * INSIGHT_DEPTH_WEIGHT
                    + actionability_score * INSIGHT_ACTIONABILITY_WEIGHT
                ) * FALLBACK_INSIGHT_MULTIPLIER  # Conservative score

        # Run the async evaluation
        # Check if there's already an event loop running
        try:
            asyncio.get_running_loop()

            # If we're in an async context, create a new event loop in a thread
            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(run_evaluation())
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_new_loop)
                return future.result()
        except RuntimeError:
            # No event loop running, we can use asyncio.run directly
            return asyncio.run(run_evaluation())


class DataGapIdentificationEvaluator(Evaluator):
    """Evaluates the identification and articulation of data gaps."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate data gap identification."""
        output = ctx.output
        expected = ctx.expected_output or ResearchExecutorExpectedOutput()

        scores = []

        # Check if gaps were identified
        has_gaps = len(output.data_gaps) > 0
        scores.append(1.0 if has_gaps else 0.3)

        if output.data_gaps:
            # Evaluate gap specificity
            specificity_scores = []
            for gap in output.data_gaps:
                word_count = len(gap.split())
                if word_count < 5:
                    specificity_scores.append(0.3)
                elif word_count < 15:
                    specificity_scores.append(0.7)
                else:
                    specificity_scores.append(1.0)

            avg_specificity = sum(specificity_scores) / len(specificity_scores)
            scores.append(avg_specificity)

            # Check expected gaps coverage
            if expected.expected_gaps:
                gap_text = " ".join(output.data_gaps).lower()
                covered_gaps = sum(
                    1 for expected_gap in expected.expected_gaps if expected_gap.lower() in gap_text
                )
                gap_coverage = covered_gaps / len(expected.expected_gaps)
                scores.append(gap_coverage)

            # Evaluate reasonableness of gap count
            gap_count = len(output.data_gaps)
            if gap_count == 0:
                count_score = 0.0
            elif gap_count <= 3:
                count_score = 0.8
            elif gap_count <= 6:
                count_score = 1.0
            else:
                count_score = 0.6  # Too many gaps might indicate poor research
            scores.append(count_score)

        return sum(scores) / len(scores) if scores else 0.0


class ComprehensiveEvaluator(Evaluator):
    """Evaluates overall research completeness and coherence."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate research comprehensiveness."""
        output = ctx.output
        expected = ctx.expected_output or ResearchExecutorExpectedOutput()

        completeness_checks = {
            "has_findings": len(output.findings) > 0,
            "has_sources": len(output.sources) > 0,
            "has_insights": len(output.key_insights) > 0,
            "has_quality_score": output.overall_quality_score > 0,
            "findings_have_evidence": any(f.supporting_evidence for f in output.findings),
            "findings_have_confidence": all(
                f.confidence_score is not None for f in output.findings
            ),
            "sources_have_relevance": any(s.relevance_score is not None for s in output.sources),
            "metadata_present": bool(output.metadata),
        }

        completeness_score = sum(completeness_checks.values()) / len(completeness_checks)

        # Check finding count expectations
        finding_count_score = 1.0
        finding_count = len(output.findings)
        if expected.min_findings and finding_count < expected.min_findings:
            finding_count_score = finding_count / expected.min_findings
        elif expected.max_findings and finding_count > expected.max_findings:
            finding_count_score = expected.max_findings / finding_count

        # Check quality score expectations
        quality_score_check = 1.0
        if (
            expected.min_overall_quality_score
            and output.overall_quality_score < expected.min_overall_quality_score
        ):
            quality_score_check = (
                output.overall_quality_score / expected.min_overall_quality_score
            )

        # Evaluate coherence between findings and insights
        coherence_score = self._evaluate_coherence(output)

        final_score = (
            completeness_score * 0.3
            + finding_count_score * 0.2
            + quality_score_check * 0.2
            + coherence_score * 0.3
        )

        return final_score

    def _evaluate_coherence(self, output: ResearchResults) -> float:
        """Evaluate coherence between findings and insights."""
        if not output.findings or not output.key_insights:
            return 0.5

        # Check if insights reference findings
        findings_text = " ".join(f.finding for f in output.findings).lower()
        insights_text = " ".join(output.key_insights).lower()

        # Simple keyword overlap check
        finding_words = set(findings_text.split())
        insight_words = set(insights_text.split())

        overlap = finding_words.intersection(insight_words)
        coherence = (
            len(overlap) / min(len(finding_words), len(insight_words))
            if finding_words and insight_words
            else 0
        )

        return min(1.0, coherence * 2)  # Boost as perfect overlap is rare


class ConfidenceCalibrationEvaluator(Evaluator):
    """Evaluates the calibration of confidence scores across findings."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate confidence calibration."""
        output = ctx.output
        expected = ctx.expected_output or ResearchExecutorExpectedOutput()

        if not output.findings:
            return 0.5

        confidence_scores = [
            f.confidence_score for f in output.findings if f.confidence_score is not None
        ]

        if not confidence_scores:
            return 0.3  # No confidence levels provided

        # Calculate confidence statistics
        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        # Check for appropriate confidence distribution
        distribution_score = 1.0
        if all(c > 0.9 for c in confidence_scores):
            distribution_score = 0.5  # Likely overconfident
        elif all(c < 0.5 for c in confidence_scores):
            distribution_score = 0.5  # Likely underconfident
        elif 0.4 <= avg_confidence <= 0.8:
            distribution_score = 1.0  # Well-calibrated range

        # Check correlation with evidence quality
        evidence_correlation_score = self._evaluate_evidence_correlation(output.findings)

        # Check expected calibration
        calibration_match = 1.0
        if expected.confidence_calibration:
            if (
                expected.confidence_calibration == "well-calibrated"
                and 0.4 <= avg_confidence <= 0.8
            ):
                calibration_match = 1.0
            elif expected.confidence_calibration == "overconfident" and avg_confidence > 0.8:
                calibration_match = 1.0
            elif expected.confidence_calibration == "underconfident" and avg_confidence < 0.4:
                calibration_match = 1.0
            else:
                calibration_match = 0.3

        # Check variance (should have some variation)
        if len(confidence_scores) > 1:
            variance = sum((c - avg_confidence) ** 2 for c in confidence_scores) / len(
                confidence_scores
            )
            variance_score = min(1.0, variance * 10)  # Some variance is good
        else:
            variance_score = 0.5

        final_score = (
            distribution_score * 0.3
            + evidence_correlation_score * 0.3
            + calibration_match * 0.2
            + variance_score * 0.2
        )

        return final_score

    def _evaluate_evidence_correlation(self, findings: list[HierarchicalFinding]) -> float:
        """Check if confidence correlates with evidence quality."""
        correlations = []

        for finding in findings:
            if finding.confidence_score is not None:
                evidence_count = len(finding.supporting_evidence)
                has_source = finding.source is not None

                # Higher confidence should have more evidence
                if finding.confidence_score > 0.7:
                    if evidence_count > 0 and has_source:
                        correlations.append(1.0)
                    elif evidence_count > 0 or has_source:
                        correlations.append(0.7)
                    else:
                        correlations.append(0.3)
                elif finding.confidence_score > 0.4:
                    if evidence_count > 0 or has_source:
                        correlations.append(1.0)
                    else:
                        correlations.append(0.5)
                else:
                    # Low confidence is fine with little evidence
                    correlations.append(1.0)

        return sum(correlations) / len(correlations) if correlations else 0.5


class EvidenceSupportEvaluator(Evaluator):
    """Evaluates the quality and sufficiency of supporting evidence."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate evidence support quality."""
        output = ctx.output

        if not output.findings:
            return 0.0

        evidence_scores = []

        for finding in output.findings:
            score = 0.0

            # Check for evidence presence
            if finding.supporting_evidence:
                score += 0.4

                # Check evidence quantity
                evidence_count = len(finding.supporting_evidence)
                if evidence_count >= 3:
                    score += 0.3
                elif evidence_count >= 2:
                    score += 0.2
                elif evidence_count >= 1:
                    score += 0.1

                # Check evidence quality (length as proxy)
                avg_evidence_length = (
                    sum(len(e.split()) for e in finding.supporting_evidence) / evidence_count
                )
                if avg_evidence_length > 20:
                    score += 0.2
                elif avg_evidence_length > 10:
                    score += 0.1

            # Check for source attribution
            if finding.source:
                score += 0.1
                if finding.source.relevance_score and finding.source.relevance_score > 0.7:
                    score = min(1.0, score * 1.1)

            evidence_scores.append(score)

        return sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.0


class CategoryCoverageEvaluator(Evaluator):
    """Evaluates the coverage and balance of finding categories."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate category coverage."""
        output = ctx.output
        expected = ctx.expected_output or ResearchExecutorExpectedOutput()

        if not output.findings:
            return 0.0

        # Get category distribution
        categories = {}
        for finding in output.findings:
            if finding.category:
                categories[finding.category] = categories.get(finding.category, 0) + 1

        if not categories:
            return 0.3  # No categorization

        scores = []

        # Check expected categories coverage
        if expected.expected_categories:
            covered = sum(1 for cat in expected.expected_categories if cat in categories)
            coverage_score = covered / len(expected.expected_categories)
            scores.append(coverage_score)

        # Check category diversity
        num_categories = len(categories)
        if num_categories == 1:
            diversity_score = 0.3
        elif num_categories == 2:
            diversity_score = 0.6
        elif num_categories <= 5:
            diversity_score = 1.0
        else:
            diversity_score = 0.8  # Too many categories might indicate poor categorization
        scores.append(diversity_score)

        # Check balance (no category should dominate too much)
        total_findings = sum(categories.values())
        max_category_ratio = max(categories.values()) / total_findings
        if max_category_ratio > 0.8:
            balance_score = 0.3  # Too dominated by one category
        elif max_category_ratio > 0.6:
            balance_score = 0.7
        else:
            balance_score = 1.0
        scores.append(balance_score)

        return sum(scores) / len(scores)


# DEPRECATED: TemporalRelevanceEvaluator has been removed as temporal relevance can be arbitrary
# Some subjects have longevity and don't require recent sources.
# The LLM-based evaluators can consider temporal aspects when actually relevant to the query.
#
# class TemporalRelevanceEvaluator(Evaluator):
#     """Evaluates temporal relevance and recency of research."""
#
#     def evaluate(self, ctx: EvaluatorContext) -> float:
#         """Evaluate temporal relevance."""
#         output = ctx.output
#
#         # Only evaluate if temporal relevance is important
#         if not ctx.input.temporal_relevance:
#             return 1.0  # Not applicable, return neutral
#
#         scores = []
#
#         # Check source recency
#         if output.sources:
#             recent_sources = 0
#             dated_sources = 0
#
#             for source in output.sources:
#                 if source.date:
#                     dated_sources += 1
#                     # Within last year
#                     if source.date > datetime.now(timezone.utc) - timedelta(days=365):
#                         recent_sources += 1
#
#             if dated_sources > 0:
#                 recency_ratio = recent_sources / dated_sources
#                 scores.append(recency_ratio)
#
#         # Check if findings mention temporal aspects
#         temporal_keywords = [
#             "recent", "current", "latest", "2024", "2023", "trend",
#             "emerging", "new", "updated", "modern", "contemporary"
#         ]
#
#         temporal_findings = 0
#         for finding in output.findings:
#             if any(keyword in finding.finding.lower() for keyword in temporal_keywords):
#                 temporal_findings += 1
#
#         if output.findings:
#             temporal_ratio = temporal_findings / len(output.findings)
#             scores.append(min(1.0, temporal_ratio * 2))
#             # Boost as not all findings need temporal references
#
#         # Check execution time is recent
#         if output.execution_time:
#             time_diff = datetime.now(timezone.utc) - output.execution_time
#             if time_diff.days == 0:
#                 scores.append(1.0)
#             elif time_diff.days <= 1:
#                 scores.append(0.8)
#             else:
#                 scores.append(0.5)
#
#         return sum(scores) / len(scores) if scores else 0.5


class CrossReferenceEvaluator(Evaluator):
    """Evaluates cross-referencing and verification across sources."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate cross-referencing quality."""
        output = ctx.output

        if not output.findings or not output.sources:
            return 0.3

        scores = []

        # Check if multiple sources are referenced
        source_count = len(output.sources)
        if source_count >= 5:
            source_diversity_score = 1.0
        elif source_count >= 3:
            source_diversity_score = 0.7
        elif source_count >= 2:
            source_diversity_score = 0.5
        else:
            source_diversity_score = 0.2
        scores.append(source_diversity_score)

        # Check if findings reference different sources
        findings_with_sources = sum(1 for f in output.findings if f.source)
        if output.findings:
            source_attribution_ratio = findings_with_sources / len(output.findings)
            scores.append(source_attribution_ratio)

        # Check for conflicting or corroborating evidence mentions
        verification_keywords = [
            "confirm",
            "corroborate",
            "agree",
            "consistent",
            "conflict",
            "contradict",
            "disagree",
            "inconsistent",
            "however",
            "although",
            "despite",
            "whereas",
        ]

        verification_mentions = 0
        all_text = " ".join(f.finding for f in output.findings).lower()
        for keyword in verification_keywords:
            if keyword in all_text:
                verification_mentions += 1

        verification_score = min(
            1.0, verification_mentions / 3
        )  # Expect at least 3 verification mentions
        scores.append(verification_score)

        return sum(scores) / len(scores)


class LLMJudgeEvaluator(Evaluator):
    """Uses an LLM to judge the quality of research execution."""

    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model
        self.judge_agent = Agent(
            model=model,
            system_prompt="""You are an expert evaluator of research quality.
            Evaluate the research execution based on:
            1. Comprehensiveness of findings
            2. Quality and credibility of sources
            3. Depth and actionability of insights
            4. Identification of data gaps
            5. Overall coherence and synthesis
            6. Evidence quality and support
            7. Appropriate confidence calibration""",
        )

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Synchronous evaluate method required by Evaluator."""
        # For now, return a placeholder. In practice, you'd call evaluate_async
        return 0.8

    async def evaluate_async(
        self,
        query: str,
        output: ResearchResults,
        expected: ResearchExecutorExpectedOutput | None = None,
    ) -> dict[str, Any]:
        """Use LLM to judge research quality."""

        # Format findings
        findings_lines: list[str] = []
        for finding in output.findings:
            confidence_text = (
                f"{finding.confidence_score:.2f}" if finding.confidence_score else "N/A"
            )
            category_text = finding.category or "uncategorized"
            findings_lines.append(
                f"- {finding.finding} (Confidence: {confidence_text}, Category: {category_text})"
            )
        findings_text = "\n".join(findings_lines)

        # Format sources
        sources_lines: list[str] = []
        for source in output.sources:
            relevance = (
                f"{source.relevance_score:.2f}"
                if source.relevance_score is not None
                else "unknown"
            )
            sources_lines.append(
                f"- {source.title} ({source.url or 'no url'}, Relevance: {relevance})"
            )
        sources_text = "\n".join(sources_lines)

        # Format insights
        insights_text = "\n".join([f"- {insight}" for insight in output.key_insights])

        # Format gaps
        gaps_text = "\n".join([f"- {gap}" for gap in output.data_gaps])

        evaluation_prompt = f"""
        Research Query: {query}

        Research Results:

        Findings ({len(output.findings)} total):
{findings_text}

        Sources ({len(output.sources)} total):
{sources_text}

        Key Insights:
{insights_text}

        Data Gaps Identified:
{gaps_text}

        Overall Quality Score: {output.overall_quality_score:.2f}

        Please evaluate this research on a scale of 0-10 for:
        1. Comprehensiveness (0-10): How thoroughly was the query addressed?
        2. Source Quality (0-10): How credible and diverse are the sources?
        3. Finding Quality (0-10): How relevant and well-supported are the findings?
        4. Insight Depth (0-10): How actionable and valuable are the insights?
        5. Gap Identification (0-10): How well were data gaps identified?
        6. Synthesis Quality (0-10): How well was information synthesized?
        7. Confidence Calibration (0-10): How appropriate are the confidence levels?

        Provide your evaluation as a JSON object with these scores and a brief explanation.
        """

        result = await self.judge_agent.run(evaluation_prompt)

        try:
            eval_data = (
                json.loads(result.output) if isinstance(result.output, str) else result.output
            )

            scores = [
                eval_data.get("comprehensiveness", 0) / 10,
                eval_data.get("source_quality", 0) / 10,
                eval_data.get("finding_quality", 0) / 10,
                eval_data.get("insight_depth", 0) / 10,
                eval_data.get("gap_identification", 0) / 10,
                eval_data.get("synthesis_quality", 0) / 10,
                eval_data.get("confidence_calibration", 0) / 10,
            ]

            final_score = sum(scores) / len(scores)

            return {
                "score": final_score,
                "comprehensiveness": eval_data.get("comprehensiveness"),
                "source_quality": eval_data.get("source_quality"),
                "finding_quality": eval_data.get("finding_quality"),
                "insight_depth": eval_data.get("insight_depth"),
                "gap_identification": eval_data.get("gap_identification"),
                "synthesis_quality": eval_data.get("synthesis_quality"),
                "confidence_calibration": eval_data.get("confidence_calibration"),
                "explanation": eval_data.get("explanation", ""),
            }
        except (json.JSONDecodeError, KeyError) as e:
            return {"score": None, "error": f"Failed to parse LLM evaluation: {e}"}


class MultiJudgeConsensusEvaluator(Evaluator):
    """Enhanced LLM judge evaluator with multi-judge consensus voting."""

    def __init__(
        self,
        models: list[str] = None,
        consensus_threshold: float = 0.6,
        weight_by_confidence: bool = True,
    ):
        """Initialize multi-judge evaluator.

        Args:
            models: List of model names to use as judges
            consensus_threshold: Minimum agreement threshold for consensus
            weight_by_confidence: Whether to weight votes by confidence scores
        """
        self.models = models or ["openai:gpt-5-mini", "openai:gpt-5"]
        self.consensus_threshold = consensus_threshold
        self.weight_by_confidence = weight_by_confidence

        # Create judge agents for each model
        self.judges = {}
        for model in self.models:
            self.judges[model] = Agent(
                model=model,
                system_prompt="""You are an expert evaluator of research quality.
                Evaluate the research execution based on:
                1. Comprehensiveness (0-10)
                2. Source Quality (0-10)
                3. Finding Relevance (0-10)
                4. Insight Value (0-10)
                5. Synthesis Quality (0-10)
                6. Confidence in evaluation (0-10)

                Return a JSON object with numeric scores and brief reasoning.""",
            )

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Synchronous evaluate method required by Evaluator."""
        return 0.85  # Placeholder

    async def evaluate_async(
        self,
        query: str,
        output: ResearchResults,
        expected: ResearchExecutorExpectedOutput | None = None,
    ) -> dict[str, Any]:
        """Use multiple LLM judges with consensus voting."""

        # Format research output for evaluation
        if output.findings:
            confidence_values = [
                f.confidence_score for f in output.findings if f.confidence_score
            ]
            average_confidence = (
                sum(confidence_values) / len(output.findings)
                if confidence_values
                else 0.0
            )
            findings_summary = (
                f"{len(output.findings)} findings with average confidence "
                f"{average_confidence:.2f}"
            )
        else:
            findings_summary = "No findings"
        sources_summary = f"{len(output.sources)} sources" if output.sources else "No sources"
        insights_summary = (
            f"{len(output.key_insights)} key insights" if output.key_insights else "No insights"
        )
        gaps_summary = (
            f"{len(output.data_gaps)} data gaps identified"
            if output.data_gaps
            else "No gaps identified"
        )

        evaluation_prompt = f"""
        Research Query: {query}

        Research Summary:
        - {findings_summary}
        - {sources_summary}
        - {insights_summary}
        - {gaps_summary}
        - Overall Quality Score: {output.overall_quality_score:.2f}

        Sample Finding: {output.findings[0].finding if output.findings else "None"}
        Sample Insight: {output.key_insights[0] if output.key_insights else "None"}

        Evaluate this research on:
        1. Comprehensiveness: How thoroughly was the query addressed?
        2. Source Quality: How credible are the sources?
        3. Finding Relevance: How relevant are the findings?
        4. Insight Value: How valuable are the insights?
        5. Synthesis Quality: How well was information synthesized?
        6. Confidence: Your confidence in this evaluation

        Return JSON: {"comprehensiveness": X, "source_quality": X,
        "finding_relevance": X, "insight_value": X, "synthesis_quality": X,
        "confidence": X, "reasoning": "brief explanation"}
        """

        # Collect evaluations from all judges
        judge_evaluations = []
        for model, judge in self.judges.items():
            try:
                result = await judge.run(evaluation_prompt)
                eval_data = (
                    json.loads(result.output) if isinstance(result.output, str) else result.output
                )

                evaluation = {
                    "model": model,
                    "comprehensiveness": eval_data.get("comprehensiveness", 0),
                    "source_quality": eval_data.get("source_quality", 0),
                    "finding_relevance": eval_data.get("finding_relevance", 0),
                    "insight_value": eval_data.get("insight_value", 0),
                    "synthesis_quality": eval_data.get("synthesis_quality", 0),
                    "confidence": eval_data.get("confidence", 5),
                    "reasoning": eval_data.get("reasoning", ""),
                    "individual_score": sum(
                        [
                            eval_data.get("comprehensiveness", 0),
                            eval_data.get("source_quality", 0),
                            eval_data.get("finding_relevance", 0),
                            eval_data.get("insight_value", 0),
                            eval_data.get("synthesis_quality", 0),
                        ]
                    )
                    / 50,  # Normalize to 0-1
                }
                judge_evaluations.append(evaluation)

            except Exception as e:
                judge_evaluations.append(
                    {"model": model, "error": str(e), "individual_score": None}
                )

        # Calculate consensus metrics
        valid_evaluations = [e for e in judge_evaluations if e.get("individual_score") is not None]

        if not valid_evaluations:
            return {
                "score": None,
                "error": "All judges failed",
                "judge_evaluations": judge_evaluations,
            }

        # Calculate weighted or simple average
        if self.weight_by_confidence:
            total_weight = sum(e["confidence"] for e in valid_evaluations)
            if total_weight > 0:
                consensus_score = (
                    sum(e["individual_score"] * e["confidence"] for e in valid_evaluations)
                    / total_weight
                )
            else:
                consensus_score = sum(e["individual_score"] for e in valid_evaluations) / len(
                    valid_evaluations
                )
        else:
            consensus_score = sum(e["individual_score"] for e in valid_evaluations) / len(
                valid_evaluations
            )

        # Calculate agreement metrics
        scores = [e["individual_score"] for e in valid_evaluations]
        score_variance = sum((s - consensus_score) ** 2 for s in scores) / len(scores)
        agreement_score = 1 / (1 + score_variance)

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
            "weighted_by_confidence": self.weight_by_confidence,
        }


def create_research_executor_dataset() -> Dataset:
    """Create comprehensive evaluation dataset for research executor agent from YAML."""
    from pathlib import Path

    # Try to load from YAML file if it exists
    yaml_path = Path(__file__).parent / "evaluation_datasets" / "research_executor_dataset.yaml"
    if yaml_path.exists():
        try:
            from tests.evals.research_executor_dataset_loader import load_dataset_from_yaml

            return load_dataset_from_yaml(yaml_path)
        except ImportError:
            pass  # Fall back to hardcoded dataset

    # Fallback: Golden Standard Cases - Clear expected outcomes
    golden_cases = [
        Case(
            name="golden_technical_comparison",
            inputs=ResearchExecutorInput(
                query="Compare React vs Angular for enterprise applications",
                domain="technical",
                complexity="medium",
                temporal_relevance=True,
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=5,
                max_findings=15,
                min_sources=3,
                max_sources=10,
                expected_categories=["technical", "business"],
                expected_insights_themes=["performance", "scalability", "ecosystem"],
                min_overall_quality_score=0.7,
            ),
            evaluators=[
                FindingsRelevanceEvaluator(),
                SourceCredibilityEvaluator(),
                InsightQualityEvaluator(),
                ComprehensiveEvaluator(),
                LLMJudgeEvaluator(),
            ],
        ),
        Case(
            name="golden_scientific_research",
            inputs=ResearchExecutorInput(
                query="Latest developments in CRISPR gene editing for cancer treatment",
                domain="scientific",
                complexity="complex",
                temporal_relevance=True,
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=6,
                max_findings=20,
                min_sources=5,
                max_sources=15,
                expected_categories=["scientific", "medical"],
                expected_gaps=["clinical trials", "long-term effects"],
                source_credibility_threshold=0.7,
                min_overall_quality_score=0.75,
            ),
            evaluators=[
                FindingsRelevanceEvaluator(),
                SourceCredibilityEvaluator(),
                DataGapIdentificationEvaluator(),
                CrossReferenceEvaluator(),
                MultiJudgeConsensusEvaluator(),
            ],
        ),
        Case(
            name="golden_business_analysis",
            inputs=ResearchExecutorInput(
                query="Market opportunities for AI-powered customer service solutions",
                domain="business",
                complexity="medium",
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=4,
                max_findings=12,
                expected_categories=["business", "technical", "economic"],
                expected_insights_themes=["roi", "automation", "customer satisfaction"],
                confidence_calibration="well-calibrated",
            ),
            evaluators=[
                FindingsRelevanceEvaluator(),
                InsightQualityEvaluator(),
                CategoryCoverageEvaluator(),
                ConfidenceCalibrationEvaluator(),
            ],
        ),
    ]

    # Domain-specific cases
    technical_cases = [
        Case(
            name="tech_kubernetes_research",
            inputs=ResearchExecutorInput(
                query="Best practices for Kubernetes security in production",
                domain="technical",
                complexity="medium",
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=5,
                expected_categories=["technical", "security"],
                expected_insights_themes=["rbac", "network policies", "secrets management"],
            ),
            evaluators=[FindingsRelevanceEvaluator(), InsightQualityEvaluator()],
        ),
        Case(
            name="tech_database_optimization",
            inputs=ResearchExecutorInput(
                query="PostgreSQL performance tuning for high-volume transactions",
                domain="technical",
                complexity="complex",
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=6, expected_categories=["technical"], source_credibility_threshold=0.6
            ),
            evaluators=[
                FindingsRelevanceEvaluator(),
                SourceCredibilityEvaluator(),
                EvidenceSupportEvaluator(),
            ],
        ),
    ]

    scientific_cases = [
        Case(
            name="sci_climate_research",
            inputs=ResearchExecutorInput(
                query="Impact of microplastics on marine ecosystems",
                domain="scientific",
                complexity="complex",
                temporal_relevance=True,
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=5,
                expected_categories=["scientific", "environmental"],
                expected_gaps=["long-term studies", "mitigation strategies"],
            ),
            evaluators=[FindingsRelevanceEvaluator(), DataGapIdentificationEvaluator()],
        ),
        Case(
            name="sci_quantum_computing",
            inputs=ResearchExecutorInput(
                query="Quantum computing applications in drug discovery",
                domain="scientific",
                complexity="complex",
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=4,
                expected_categories=["scientific", "technical"],
                confidence_calibration="underconfident",  # Emerging field
            ),
            evaluators=[FindingsRelevanceEvaluator(), ConfidenceCalibrationEvaluator()],
        ),
    ]

    business_cases = [
        Case(
            name="biz_ecommerce_trends",
            inputs=ResearchExecutorInput(
                query="Emerging trends in social commerce for Gen Z consumers",
                domain="business",
                complexity="medium",
                temporal_relevance=True,
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=5,
                expected_categories=["business", "social"],
                expected_insights_themes=["tiktok", "instagram", "live shopping"],
            ),
            evaluators=[FindingsRelevanceEvaluator(), InsightQualityEvaluator()],
        ),
        Case(
            name="biz_supply_chain",
            inputs=ResearchExecutorInput(
                query="Blockchain adoption in supply chain management",
                domain="business",
                complexity="medium",
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=4,
                expected_categories=["business", "technical"],
                expected_gaps=["roi data", "scalability concerns"],
            ),
            evaluators=[FindingsRelevanceEvaluator(), DataGapIdentificationEvaluator()],
        ),
    ]

    medical_cases = [
        Case(
            name="med_alzheimers_research",
            inputs=ResearchExecutorInput(
                query="Recent breakthroughs in Alzheimer's disease treatment",
                domain="medical",
                complexity="complex",
                temporal_relevance=True,
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=5,
                min_sources=4,
                expected_categories=["medical", "scientific"],
                source_credibility_threshold=0.8,  # High standard for medical
                expected_gaps=["clinical trial results", "side effects"],
            ),
            evaluators=[
                FindingsRelevanceEvaluator(),
                SourceCredibilityEvaluator(),
                DataGapIdentificationEvaluator(),
                CrossReferenceEvaluator(),
            ],
        ),
        Case(
            name="med_telemedicine",
            inputs=ResearchExecutorInput(
                query="Effectiveness of telemedicine for mental health treatment",
                domain="medical",
                complexity="medium",
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=4,
                expected_categories=["medical", "social"],
                expected_insights_themes=["accessibility", "outcomes", "patient satisfaction"],
            ),
            evaluators=[
                FindingsRelevanceEvaluator(),
                InsightQualityEvaluator(),
                EvidenceSupportEvaluator(),
            ],
        ),
    ]

    # Edge cases
    edge_cases = [
        Case(
            name="edge_minimal_query",
            inputs=ResearchExecutorInput(query="AI", complexity="simple"),
            expected_output=ResearchExecutorExpectedOutput(min_findings=3, max_findings=8),
            evaluators=[FindingsRelevanceEvaluator(), ComprehensiveEvaluator()],
        ),
        Case(
            name="edge_highly_specific",
            inputs=ResearchExecutorInput(
                query="Performance comparison of BERT-base vs RoBERTa-base on GLUE benchmark tasks",
                domain="technical",
                complexity="complex",
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=2, max_findings=6, expected_categories=["technical", "scientific"]
            ),
            evaluators=[FindingsRelevanceEvaluator(), SourceCredibilityEvaluator()],
        ),
        Case(
            name="edge_multi_domain",
            inputs=ResearchExecutorInput(
                query="Legal, ethical, and technical challenges of autonomous vehicles",
                complexity="complex",
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=6,
                expected_categories=["technical", "regulatory", "social"],
                expected_gaps=["liability frameworks", "edge cases"],
            ),
            evaluators=[CategoryCoverageEvaluator(), DataGapIdentificationEvaluator()],
        ),
        Case(
            name="edge_contradictory",
            inputs=ResearchExecutorInput(
                query="Benefits and risks of nuclear energy for climate change mitigation",
                complexity="complex",
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=5,
                expected_categories=["environmental", "technical", "economic"],
                confidence_calibration="well-calibrated",
            ),
            evaluators=[
                FindingsRelevanceEvaluator(),
                CrossReferenceEvaluator(),
                ConfidenceCalibrationEvaluator(),
            ],
        ),
    ]

    # Performance benchmark cases
    performance_cases = [
        Case(
            name="perf_simple_factual",
            inputs=ResearchExecutorInput(
                query="What is the GDP of United States in 2023?", complexity="simple"
            ),
            expected_output=ResearchExecutorExpectedOutput(
                min_findings=1, max_findings=3, max_response_time=5.0
            ),
            evaluators=[FindingsRelevanceEvaluator(), ComprehensiveEvaluator()],
        ),
        Case(
            name="perf_complex_synthesis",
            inputs=ResearchExecutorInput(
                query=(
                    "Comprehensive analysis of global renewable energy adoption trends, "
                    "technological innovations, policy frameworks, and economic impacts "
                    "across developed and developing nations"
                ),
                complexity="complex",
            ),
            expected_output=ResearchExecutorExpectedOutput(min_findings=8, max_response_time=30.0),
            evaluators=[
                FindingsRelevanceEvaluator(),
                ComprehensiveEvaluator(),
                InsightQualityEvaluator(),
            ],
        ),
    ]

    all_cases = (
        golden_cases
        + technical_cases
        + scientific_cases
        + business_cases
        + medical_cases
        + edge_cases
        + performance_cases
    )

    return Dataset(cases=all_cases)


async def run_research_executor_evaluation():
    """Run complete evaluation of research executor agent."""

    # Create agent
    agent = ResearchExecutorAgent()

    # Create dataset
    dataset = create_research_executor_dataset()

    # Define the task function that will be evaluated
    async def research_task(inputs: ResearchExecutorInput) -> ResearchResults:
        """Task function for evaluation."""
        async with httpx.AsyncClient() as http_client:
            state = ResearchState(
                request_id="eval-test",
                user_id="test-user",
                session_id="test-session",
                user_query=inputs.query,
                current_stage=ResearchStage.RESEARCH_EXECUTION,
                metadata=ResearchMetadata(),
            )

            # Add research brief and methodology to metadata if provided
            if inputs.research_brief or inputs.methodology:
                state.metadata.query.transformed_query = {
                    "research_plan": {
                        "brief": inputs.research_brief or "",
                        "methodology": inputs.methodology or "",
                    }
                }

            deps = ResearchDependencies(
                http_client=http_client,
                api_keys=APIKeys(
                    openai=SecretStr(openai_key)
                    if (openai_key := os.getenv("OPENAI_API_KEY"))
                    else None
                ),
                research_state=state,
            )

            result = await agent.agent.run(inputs.query, deps=deps)
            return result.output

    # Run evaluation
    report = await dataset.evaluate(research_task)

    return report


def generate_evaluation_report(report: EvaluationReport) -> str:
    """Generate human-readable evaluation report."""

    output = ["=" * 60]
    output.append("RESEARCH EXECUTOR AGENT EVALUATION REPORT")
    output.append("=" * 60)

    # Overall metrics
    output.append("\nOVERALL METRICS:")
    output.append("-" * 40)

    total_cases = len(report.cases)
    all_scores = []

    for case in report.cases:
        case_scores = [
            eval_result.get("score", 0)
            for eval_result in case.evaluations.values()
            if eval_result.get("score") is not None
        ]
        if case_scores:
            all_scores.extend(case_scores)

    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        output.append(f"Average Score: {avg_score:.2%}")

    # Per-case breakdown
    output.append("\nPER-CASE PERFORMANCE:")
    output.append("-" * 40)

    for case in report.cases:
        output.append(f"\n{case.name}:")
        for evaluator_name, evaluation in case.evaluations.items():
            if evaluation.get("score") is not None:
                output.append(f"  {evaluator_name}: {evaluation['score']:.2f}")

    # Research patterns
    output.append("\nRESEARCH PATTERNS:")
    output.append("-" * 40)

    total_findings = 0
    total_sources = 0
    total_insights = 0
    total_gaps = 0

    for case in report.cases:
        if case.output:
            total_findings += len(case.output.findings)
            total_sources += len(case.output.sources)
            total_insights += len(case.output.key_insights)
            total_gaps += len(case.output.data_gaps)

    if total_cases > 0:
        output.append(f"Average findings per case: {total_findings / total_cases:.1f}")
        output.append(f"Average sources per case: {total_sources / total_cases:.1f}")
        output.append(f"Average insights per case: {total_insights / total_cases:.1f}")
        output.append(f"Average gaps identified per case: {total_gaps / total_cases:.1f}")

    # Category distribution
    category_counts = {}
    for case in report.cases:
        if case.output and case.output.findings:
            for finding in case.output.findings:
                if finding.category:
                    category_counts[finding.category] = category_counts.get(finding.category, 0) + 1

    if category_counts:
        output.append("\nFinding Category Distribution:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            output.append(f"  {category}: {count} findings")

    return "\n".join(output)


if __name__ == "__main__":
    # Run evaluation
    report = asyncio.run(run_research_executor_evaluation())
    print(generate_evaluation_report(report))
