"""Compression Agent for synthesizing and organizing research findings."""

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from open_deep_research_with_pydantic_ai.agents.base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
    coordinator,
)
from open_deep_research_with_pydantic_ai.core.events import emit_stage_completed
from open_deep_research_with_pydantic_ai.models.research import ResearchFinding, ResearchStage


class CompressedFindings(BaseModel):
    """Compressed and synthesized research findings."""

    summary: str = Field(description="Executive summary of all findings")
    key_insights: list[str] = Field(description="Key insights extracted from findings")
    themes: dict[str, list[str]] = Field(
        description="Findings organized by theme", default_factory=dict
    )
    contradictions: list[str] = Field(
        default_factory=list, description="Any contradictory findings identified"
    )
    gaps: list[str] = Field(default_factory=list, description="Identified gaps in the research")
    consensus_points: list[str] = Field(
        default_factory=list, description="Points where multiple sources agree"
    )
    statistical_data: dict[str, Any] = Field(
        default_factory=dict, description="Key statistics and data points"
    )
    source_quality_summary: str = Field(
        default="", description="Summary of source quality and reliability"
    )


class CompressionAgent(BaseResearchAgent[ResearchDependencies, CompressedFindings]):
    """Agent responsible for compressing and synthesizing research findings."""

    def __init__(self):
        """Initialize the compression agent."""
        config = AgentConfiguration(
            agent_name="compression_agent",
            agent_type="compression",
        )
        super().__init__(config=config)

    def _get_output_type(self) -> type[CompressedFindings]:
        """Get the output type for this agent."""
        return CompressedFindings

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for compression."""
        return """You are a research synthesis specialist. Your role is to:

1. Synthesize multiple research findings into coherent insights
2. Identify patterns, themes, and relationships
3. Resolve or highlight contradictions
4. Extract key takeaways and conclusions
5. Organize information for maximum clarity

When compressing findings:

**Synthesis Approach:**
- Group related findings by theme
- Identify common patterns across sources
- Extract the most important insights
- Combine complementary information
- Preserve important nuances

**Quality Assessment:**
- Evaluate the consistency of findings
- Identify areas of strong consensus
- Note any conflicting information
- Assess overall confidence in conclusions
- Identify gaps that need further research

**Organization Principles:**
- Start with high-level summary
- Group findings thematically
- Highlight key insights prominently
- Note statistical data separately
- Maintain source attribution for key claims

**Output Requirements:**
- Concise yet comprehensive summary
- Clear thematic organization
- Explicit noting of contradictions
- Identification of research gaps
- Actionable insights where applicable

Ensure the compressed findings are more valuable than the sum of their parts."""

    def _register_tools(self) -> None:
        """Register compression-specific tools."""

        @self.agent.tool
        async def identify_themes(
            _ctx: RunContext[ResearchDependencies], findings: list[ResearchFinding]
        ) -> dict[str, list[str]]:
            """Identify themes from research findings.

            Args:
                ctx: Run context with dependencies
                findings: List of research findings

            Returns:
                Dictionary of themes with related findings
            """
            themes: defaultdict[str, list[str]] = defaultdict(list)

            # Common theme categories
            theme_keywords = {
                "Technology": ["technology", "digital", "software", "hardware", "AI", "automation"],
                "Economics": ["cost", "price", "market", "economy", "financial", "investment"],
                "Social Impact": ["society", "community", "people", "social", "cultural", "human"],
                "Environment": ["environment", "climate", "sustainability", "green", "ecological"],
                "Innovation": ["innovation", "new", "novel", "breakthrough", "advancement"],
                "Challenges": ["challenge", "problem", "issue", "difficulty", "obstacle"],
                "Opportunities": [
                    "opportunity",
                    "potential",
                    "possibility",
                    "benefit",
                    "advantage",
                ],
                "Trends": ["trend", "future", "emerging", "growth", "development"],
            }

            # Categorize findings by theme
            for finding in findings:
                content_lower = finding.content.lower()
                categorized = False

                for theme, keywords in theme_keywords.items():
                    if any(keyword in content_lower for keyword in keywords):
                        themes[theme].append(finding.content[:200])
                        categorized = True
                        break

                if not categorized:
                    themes["Other"].append(finding.content[:200])

            return dict(themes)

        @self.agent.tool
        async def find_contradictions(
            _ctx: RunContext[ResearchDependencies], findings: list[ResearchFinding]
        ) -> list[str]:
            """Identify contradictory findings.

            Args:
                ctx: Run context with dependencies
                findings: List of research findings

            Returns:
                List of identified contradictions
            """
            contradictions = []

            # Simple contradiction detection based on opposing terms
            opposing_pairs = [
                ("increase", "decrease"),
                ("positive", "negative"),
                ("growth", "decline"),
                ("success", "failure"),
                ("effective", "ineffective"),
                ("beneficial", "harmful"),
            ]

            # Compare findings pairwise for potential contradictions
            for i, finding1 in enumerate(findings):
                for finding2 in findings[i + 1 :]:
                    content1_lower = finding1.content.lower()
                    content2_lower = finding2.content.lower()

                    for term1, term2 in opposing_pairs:
                        if (term1 in content1_lower and term2 in content2_lower) or (
                            term2 in content1_lower and term1 in content2_lower
                        ):
                            # Check if they're about the same subject
                            words1 = set(content1_lower.split())
                            words2 = set(content2_lower.split())
                            common_words = words1.intersection(words2)

                            if len(common_words) > 5:  # Arbitrary threshold
                                contradiction = (
                                    f"Potential contradiction between: "
                                    f"'{finding1.content[:100]}...' and "
                                    f"'{finding2.content[:100]}...'"
                                )
                                contradictions.append(contradiction)  # type: ignore[arg-type]
                                break

            return contradictions[:5]  # type: ignore[return-value]  # Limit to top 5 contradictions

        @self.agent.tool
        async def extract_consensus_points(
            _ctx: RunContext[ResearchDependencies], findings: list[ResearchFinding]
        ) -> list[str]:
            """Extract points where multiple sources agree.

            Args:
                ctx: Run context with dependencies
                findings: List of research findings

            Returns:
                List of consensus points
            """
            consensus_points: list[str] = []

            # Group findings by source
            from typing import Any
            source_groups: defaultdict[str, list[Any]] = defaultdict(list)
            for finding in findings:
                source_groups[finding.source].append(finding)

            # Find common themes across sources
            if len(source_groups) > 1:
                # Extract key phrases from each source
                source_phrases: dict[str, set[str]] = {}
                for source, source_findings in source_groups.items():
                    phrases: set[str] = set()
                    for finding in source_findings:
                        # Simple phrase extraction (in production, use NLP)
                        words = finding.content.lower().split()
                        for i in range(len(words) - 2):
                            phrase = " ".join(words[i : i + 3])
                            phrases.add(phrase)
                    source_phrases[source] = phrases

                # Find common phrases across sources
                sources = list(source_phrases.keys())
                if len(sources) >= 2:
                    common: set[str] = source_phrases[sources[0]]
                    for source in sources[1:]:
                        common = common.intersection(source_phrases[source])

                    for phrase in list(common)[:10]:
                        consensus_points.append(f"Multiple sources agree on: {phrase}")

            return consensus_points[:5]  # Top 5 consensus points

        @self.agent.tool
        async def identify_gaps(
            _ctx: RunContext[ResearchDependencies],
            findings: list[ResearchFinding],
            research_questions: list[str],
        ) -> list[str]:
            """Identify gaps in the research.

            Args:
                ctx: Run context with dependencies
                findings: List of research findings
                research_questions: Original research questions

            Returns:
                List of identified gaps
            """
            gaps: list[str] = []

            # Check if each research question was adequately addressed
            for question in research_questions:
                question_lower = question.lower()
                question_addressed = False

                for finding in findings:
                    # Simple check - in production, use semantic similarity
                    finding_lower = finding.content.lower()
                    question_words = set(question_lower.split())
                    finding_words = set(finding_lower.split())

                    # If significant overlap, consider it addressed
                    common_words = question_words.intersection(finding_words)
                    if len(common_words) >= len(question_words) * 0.3:
                        question_addressed = True
                        break

                if not question_addressed:
                    gaps.append(f"Limited information on: {question}")

            # Check for low coverage areas based on confidence scores
            low_confidence_topics: list[str] = []
            for finding in findings:
                if finding.confidence < 0.5:
                    low_confidence_topics.append(finding.summary or finding.content[:100])

            if low_confidence_topics:
                gaps.append(f"Low confidence areas: {', '.join(low_confidence_topics[:3])}")

            return gaps

    async def compress_findings(
        self,
        findings: list[ResearchFinding],
        research_questions: list[str],
        deps: ResearchDependencies,
    ) -> CompressedFindings:
        """Compress and synthesize research findings.

        Args:
            findings: List of research findings to compress
            research_questions: Original research questions
            deps: Research dependencies

        Returns:
            Compressed and synthesized findings
        """
        # Prepare findings summary for compression
        findings_text = "\n\n".join(
            [
                (
                    f"Finding {i + 1} (Source: {f.source}, "
                    f"Relevance: {f.relevance_score:.2f}):\n{f.content}"
                )
                for i, f in enumerate(findings)
            ]
        )

        prompt = f"""Synthesize and compress the following research findings:

Research Questions:
{chr(10).join(f"- {q}" for q in research_questions)}

Findings to Compress:
{findings_text}

Instructions:
1. Create a comprehensive summary of all findings
2. Extract 5-7 key insights
3. Organize findings by theme
4. Identify any contradictions
5. Note research gaps
6. Find consensus points where sources agree
7. Extract important statistical data
8. Assess overall source quality

Provide a structured compression that makes the findings more valuable and actionable."""

        result = await self.run(prompt, deps, stream=True)

        # Store compressed findings in research state
        deps.research_state.compressed_findings = result.summary

        # Emit stage completed event
        await emit_stage_completed(
            request_id=deps.research_state.request_id,
            stage=ResearchStage.COMPRESSION,
            success=True,
            result=result,
        )

        return result


# Register the agent with the coordinator
compression_agent = CompressionAgent()
coordinator.register_agent(compression_agent)
