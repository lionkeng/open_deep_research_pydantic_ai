"""Multi-Judge Adapter for Research Executor Agent Evaluation.

This module provides the adapter implementation for evaluating ResearchExecutorAgent
outputs using the generalized multi-judge consensus framework.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.evals.base_multi_judge import (
    AgentEvaluationAdapter,
    EvaluationDimension,
    JudgeExpertise
)
from src.models.research_executor import ResearchResults


class ResearchExecutorMultiJudgeAdapter(AgentEvaluationAdapter[str, ResearchResults]):
    """Adapter for evaluating ResearchExecutorAgent outputs with multi-judge consensus."""

    def get_evaluation_dimensions(self) -> List[EvaluationDimension]:
        """Define evaluation dimensions for research executor outputs."""
        return [
            EvaluationDimension(
                name="finding_accuracy",
                description="How accurate and well-supported are the research findings",
                scale=(0, 10),
                weight=1.5
            ),
            EvaluationDimension(
                name="source_reliability",
                description="How credible and diverse are the research sources",
                scale=(0, 10),
                weight=1.3
            ),
            EvaluationDimension(
                name="insight_depth",
                description="How deep, actionable, and valuable are the key insights",
                scale=(0, 10),
                weight=1.4
            ),
            EvaluationDimension(
                name="gap_identification",
                description="How well are data gaps and limitations identified",
                scale=(0, 10),
                weight=1.0
            ),
            EvaluationDimension(
                name="evidence_quality",
                description="How strong and relevant is the supporting evidence",
                scale=(0, 10),
                weight=1.2
            ),
            EvaluationDimension(
                name="synthesis_coherence",
                description="How well is information synthesized into a coherent narrative",
                scale=(0, 10),
                weight=1.3
            ),
            EvaluationDimension(
                name="comprehensiveness",
                description="How thoroughly does the research address the query",
                scale=(0, 10),
                weight=1.4
            ),
            EvaluationDimension(
                name="confidence_calibration",
                description="How appropriate are the confidence levels assigned",
                scale=(0, 10),
                weight=1.0
            )
        ]

    def format_output_for_evaluation(self, output: ResearchResults) -> str:
        """Format research results into a string for evaluation."""

        # Format findings summary
        findings_text = []
        if output.findings:
            for i, finding in enumerate(output.findings[:5], 1):  # Limit to first 5 for brevity
                conf_text = f"Confidence: {finding.confidence_level:.2f}" if finding.confidence_level else "Confidence: N/A"
                cat_text = f"Category: {finding.category}" if finding.category else "Category: uncategorized"
                findings_text.append(f"  {i}. {finding.finding} ({conf_text}, {cat_text})")

                # Add supporting evidence if available
                if finding.supporting_evidence:
                    for j, evidence in enumerate(finding.supporting_evidence[:2], 1):
                        findings_text.append(f"     Evidence {j}: {evidence[:100]}...")
        else:
            findings_text.append("  No findings generated")

        # Format sources summary
        sources_text = []
        if output.sources:
            for i, source in enumerate(output.sources[:5], 1):  # Limit to first 5
                relevance = f"Relevance: {source.relevance_score:.2f}" if source.relevance_score else "Relevance: N/A"
                sources_text.append(f"  {i}. {source.title or 'Untitled'} ({relevance})")
                if source.url:
                    sources_text.append(f"     URL: {source.url}")
        else:
            sources_text.append("  No sources cited")

        # Format insights
        insights_text = []
        if output.key_insights:
            for i, insight in enumerate(output.key_insights[:5], 1):
                insights_text.append(f"  {i}. {insight}")
        else:
            insights_text.append("  No key insights extracted")

        # Format data gaps
        gaps_text = []
        if output.data_gaps:
            for i, gap in enumerate(output.data_gaps[:5], 1):
                gaps_text.append(f"  {i}. {gap}")
        else:
            gaps_text.append("  No data gaps identified")

        return f"""Research Results for Query: {output.query}

Findings ({len(output.findings)} total):
{chr(10).join(findings_text)}

Sources ({len(output.sources)} total):
{chr(10).join(sources_text)}

Key Insights ({len(output.key_insights)} total):
{chr(10).join(insights_text)}

Data Gaps ({len(output.data_gaps)} total):
{chr(10).join(gaps_text)}

Quality Score: {output.quality_score:.2f}
Execution Time: {output.execution_time.isoformat() if output.execution_time else 'N/A'}"""

    def create_evaluation_prompt(
        self,
        input: str,  # The research query
        output: ResearchResults,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create the evaluation prompt for judges."""

        formatted_output = self.format_output_for_evaluation(output)

        domain_context = ""
        if context and "domain" in context:
            domain_context = f"\nDomain: {context['domain']}"

        complexity_context = ""
        if context and "complexity" in context:
            complexity_context = f"\nComplexity: {context['complexity']}"

        temporal_context = ""
        if context and "temporal_relevance" in context:
            temporal_context = f"\nTemporal Relevance: {'Important' if context['temporal_relevance'] else 'Not critical'}"

        return f"""Evaluate the following research execution output:

Original Query: {input}{domain_context}{complexity_context}{temporal_context}

{formatted_output}

Please evaluate this research output based on the specified dimensions. Consider:

1. **Finding Accuracy**: Are the findings factually correct and properly supported by evidence?
2. **Source Reliability**: Are the sources credible, diverse, and appropriate for the research topic?
3. **Insight Depth**: Do the insights provide value beyond surface-level observations?
4. **Gap Identification**: Are important data gaps and limitations clearly identified?
5. **Evidence Quality**: Is the supporting evidence strong, relevant, and well-documented?
6. **Synthesis Coherence**: Is the information well-synthesized into a coherent narrative?
7. **Comprehensiveness**: Does the research thoroughly address all aspects of the query?
8. **Confidence Calibration**: Are confidence levels appropriately assigned based on evidence quality?

Provide your evaluation as a JSON object with scores for each dimension and your overall confidence."""

    def is_output_valid(self, output: ResearchResults) -> bool:
        """Check if the research results are valid for evaluation."""

        # Must have at least some findings or insights
        has_content = (
            (output.findings and len(output.findings) > 0) or
            (output.key_insights and len(output.key_insights) > 0)
        )

        # Must have a query
        has_query = bool(output.query)

        # Must have a quality score
        has_quality_score = output.quality_score is not None and output.quality_score >= 0

        return has_content and has_query and has_quality_score

    def get_expertise_context(self, expertise: JudgeExpertise) -> str:
        """Get expertise-specific context for the system prompt."""

        contexts = {
            JudgeExpertise.GENERAL: """You bring a balanced perspective to research evaluation,
considering both breadth and depth of coverage. Focus on overall quality and usefulness.

""",
            JudgeExpertise.TECHNICAL: """You specialize in evaluating technical and scientific research.
Pay special attention to technical accuracy, methodology rigor, and evidence quality.

""",
            JudgeExpertise.SCIENTIFIC: """You are an expert in scientific research methodology.
Focus on research design, source credibility, statistical validity, and proper citation.

""",
            JudgeExpertise.BUSINESS: """You evaluate research from a business and practical perspective.
Consider actionability, real-world applicability, and commercial relevance.

""",
            JudgeExpertise.CREATIVE: """You bring a creative and innovative perspective to research evaluation.
Look for novel insights, unexpected connections, and creative synthesis of information.

"""
        }

        return contexts.get(expertise, contexts[JudgeExpertise.GENERAL])

    def aggregate_dimension_feedback(
        self,
        dimension: str,
        scores: List[float],
        judge_reasonings: List[str]
    ) -> str:
        """Aggregate feedback for a specific dimension across judges."""

        avg_score = sum(scores) / len(scores) if scores else 0

        # Identify consensus and disagreement
        if scores:
            std_dev = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5
            if std_dev < 1.0:
                consensus = "strong consensus"
            elif std_dev < 2.0:
                consensus = "moderate consensus"
            else:
                consensus = "significant disagreement"
        else:
            consensus = "no data"

        # Extract key themes from reasonings
        themes = []
        keywords = {
            "finding_accuracy": ["accurate", "correct", "supported", "evidence", "verified"],
            "source_reliability": ["credible", "authoritative", "diverse", "peer-reviewed", "recent"],
            "insight_depth": ["actionable", "valuable", "deep", "surface", "practical"],
            "gap_identification": ["gaps", "limitations", "missing", "incomplete", "unknown"],
            "evidence_quality": ["strong", "weak", "relevant", "documented", "citation"],
            "synthesis_coherence": ["coherent", "organized", "synthesized", "fragmented", "clear"],
            "comprehensiveness": ["thorough", "complete", "comprehensive", "partial", "addressed"],
            "confidence_calibration": ["appropriate", "overconfident", "underconfident", "calibrated"]
        }

        dimension_keywords = keywords.get(dimension, [])
        for reasoning in judge_reasonings:
            reasoning_lower = reasoning.lower()
            for keyword in dimension_keywords:
                if keyword in reasoning_lower:
                    themes.append(keyword)

        # Create aggregated feedback
        theme_summary = f"Key aspects: {', '.join(set(themes[:3]))}" if themes else "Various aspects noted"

        return f"{dimension.replace('_', ' ').title()}: {avg_score:.1f}/10 ({consensus}) - {theme_summary}"

    def create_comparison_prompt(
        self,
        input: str,
        output_a: ResearchResults,
        output_b: ResearchResults,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create prompt for comparing two research outputs."""

        formatted_a = self.format_output_for_evaluation(output_a)
        formatted_b = self.format_output_for_evaluation(output_b)

        return f"""Compare these two research outputs for the query: {input}

OUTPUT A:
{formatted_a}

OUTPUT B:
{formatted_b}

Which output provides better research quality overall? Consider all evaluation dimensions.
Provide your assessment as JSON with structure:
{{
    "winner": "A" or "B",
    "confidence": 0-10,
    "reasoning": "explanation",
    "dimension_comparison": {{
        "finding_accuracy": {{"winner": "A/B", "margin": X}},
        "source_reliability": {{"winner": "A/B", "margin": X}},
        // ... other dimensions
    }}
}}"""
