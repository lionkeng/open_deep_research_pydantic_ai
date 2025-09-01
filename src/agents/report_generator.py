"""Final Report Generator Agent for creating comprehensive research reports."""

from datetime import datetime

from pydantic_ai import RunContext

from agents.base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
    coordinator,
)
from agents.compression import CompressedFindings
from core.events import (
    ResearchCompletedEvent,
    emit_stage_completed,
    research_event_bus,
)
from models.research import (
    ResearchBrief,
    ResearchFinding,
    ResearchReport,
    ResearchSection,
    ResearchStage,
)


class ReportGeneratorAgent(BaseResearchAgent[ResearchDependencies, ResearchReport]):
    """Agent responsible for generating final research reports."""

    def __init__(self):
        """Initialize the reportgenerator agent."""
        config = AgentConfiguration(
            agent_name="reportgenerator_agent",
            agent_type="reportgenerator",
        )
        super().__init__(config=config)

    def _get_output_type(self) -> type[ResearchReport]:
        """Get the output type for this agent."""
        return ResearchReport

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for report generation."""
        return """You are a research report specialist. Your role is to create comprehensive,
well-structured, and professional research reports.

When generating reports:

**Report Structure:**
1. Title - Clear, descriptive, and engaging
2. Executive Summary - High-level overview of findings and recommendations
3. Introduction - Context, background, and research objectives
4. Methodology - How the research was conducted
5. Main Sections - Organized by theme or topic
6. Conclusion - Summary of key findings
7. Recommendations - Actionable insights and next steps
8. Citations - All sources properly referenced

**Writing Principles:**
- Clear and concise language
- Logical flow and organization
- Evidence-based arguments
- Balanced perspective
- Professional tone
- Accessible to target audience

**Quality Standards:**
- Accuracy: All facts are verified and correct
- Completeness: All research questions addressed
- Clarity: Ideas are clearly expressed
- Coherence: Sections flow logically
- Credibility: Proper source attribution
- Actionability: Clear recommendations

**Formatting Guidelines:**
- Use headings and subheadings for organization
- Include bullet points for lists
- Highlight key findings and insights
- Use data and statistics effectively
- Maintain consistent style throughout

Ensure the report is comprehensive yet readable, authoritative yet accessible."""

    def _register_tools(self) -> None:
        """Register report generation tools."""

        @self.agent.tool
        async def create_executive_summary(
            ctx: RunContext[ResearchDependencies],
            compressed_findings: CompressedFindings,
            brief: ResearchBrief,
        ) -> str:
            """Create an executive summary for the report.

            Args:
                ctx: Run context with dependencies
                compressed_findings: Compressed research findings
                brief: Original research brief

            Returns:
                Executive summary text
            """
            summary_parts: list[str] = []

            # Opening statement
            summary_parts.append(
                f"This research report addresses the topic of '{brief.topic}', "
                f"examining {len(brief.objectives)} key objectives through analysis."
            )

            # Key findings
            if compressed_findings.key_insights:
                summary_parts.append("\nKey Findings:")
                for insight in compressed_findings.key_insights[:3]:
                    summary_parts.append(f"• {insight}")

            # Consensus points
            if compressed_findings.consensus_points:
                summary_parts.append("\nAreas of Consensus:")
                for point in compressed_findings.consensus_points[:2]:
                    summary_parts.append(f"• {point}")

            # Challenges or contradictions
            if compressed_findings.contradictions:
                summary_parts.append(
                    f"\nThe research identified {len(compressed_findings.contradictions)} "
                    "areas requiring further investigation."
                )

            # Conclusion
            summary_parts.append(
                f"\n{compressed_findings.summary[:200]}..."
                if len(compressed_findings.summary) > 200
                else f"\n{compressed_findings.summary}"
            )

            return "\n".join(summary_parts)

        @self.agent.tool
        async def create_methodology_section(
            ctx: RunContext[ResearchDependencies],
            brief: ResearchBrief,
            findings_count: int,
        ) -> str:
            """Create the methodology section of the report.

            Args:
                ctx: Run context with dependencies
                brief: Research brief
                findings_count: Number of findings gathered

            Returns:
                Methodology section text
            """
            methodology = f"""Research Methodology

This research was conducted using a systematic approach to ensure comprehensive coverage
and reliable results.

Research Design:
• Objective-driven research focusing on {len(brief.objectives)} key objectives
• Multi-source information gathering from {findings_count} distinct findings
• Systematic synthesis and analysis of collected data

Data Collection:
• Comprehensive search across multiple authoritative sources
• Evaluation of source credibility and relevance
• Cross-verification of information across sources

Analysis Approach:
• Thematic analysis to identify patterns and relationships
• Comparative analysis to identify consensus and contradictions
• Gap analysis to identify areas requiring further research

Quality Assurance:
• Source credibility assessment
• Information verification across multiple sources
• Systematic documentation of all findings

Scope and Limitations:
• Research scope: {brief.scope}
• Constraints: {", ".join(brief.constraints) if brief.constraints else "None identified"}
• Time frame: Current analysis based on available information"""

            return methodology

        @self.agent.tool
        async def organize_sections_by_theme(
            ctx: RunContext[ResearchDependencies],
            compressed_findings: CompressedFindings,
            findings: list[ResearchFinding],
        ) -> list[ResearchSection]:
            """Organize report sections by theme.

            Args:
                ctx: Run context with dependencies
                compressed_findings: Compressed findings with themes
                findings: Original research findings

            Returns:
                List of organized report sections
            """
            sections: list[ResearchSection] = []

            for i, (theme, theme_content) in enumerate(compressed_findings.themes.items()):
                # Get relevant findings for this theme
                relevant_findings = [
                    f for f in findings if any(content in f.content for content in theme_content)
                ][:5]  # Limit to top 5 findings per theme

                section = ResearchSection(
                    title=theme,
                    content="\n\n".join(theme_content),
                    findings=relevant_findings,
                    order=i,
                )
                sections.append(section)

            return sections

        @self.agent.tool
        async def generate_recommendations(
            ctx: RunContext[ResearchDependencies],
            compressed_findings: CompressedFindings,
            brief: ResearchBrief,
        ) -> list[str]:
            """Generate actionable recommendations.

            Args:
                ctx: Run context with dependencies
                compressed_findings: Compressed research findings
                brief: Research brief

            Returns:
                List of recommendations
            """
            recommendations: list[str] = []

            # Based on key insights
            for insight in compressed_findings.key_insights[:3]:
                rec = (
                    f"Based on the finding that {insight}, "
                    "it is recommended to explore implementation strategies."
                )
                recommendations.append(rec)

            # Based on gaps
            for gap in compressed_findings.gaps[:2]:
                rec = f"Further research is recommended to address: {gap}"
                recommendations.append(rec)

            # Based on opportunities in themes
            if "Opportunities" in compressed_findings.themes:
                rec = "Leverage identified opportunities for strategic advantage"
                recommendations.append(rec)

            # Based on challenges
            if "Challenges" in compressed_findings.themes:
                rec = "Develop mitigation strategies for identified challenges"
                recommendations.append(rec)

            return recommendations[:5]  # Top 5 recommendations

        @self.agent.tool
        async def compile_citations(
            ctx: RunContext[ResearchDependencies], findings: list[ResearchFinding]
        ) -> list[str]:
            """Compile all citations from research findings.

            Args:
                ctx: Run context with dependencies
                findings: Research findings with sources

            Returns:
                List of formatted citations
            """
            # Get unique sources
            sources = list({f.source for f in findings})

            # Format citations (simplified - in production, use proper citation format)
            citations: list[str] = []
            for i, source in enumerate(sources, 1):
                citation = f"[{i}] {source} (Accessed: {datetime.now().strftime('%Y-%m-%d')})"
                citations.append(citation)

            return sorted(citations)

    async def generate_report(
        self,
        brief: ResearchBrief,
        findings: list[ResearchFinding],
        compressed_findings: CompressedFindings,
        deps: ResearchDependencies,
    ) -> ResearchReport:
        """Generate the final research report.

        Args:
            brief: Research brief
            findings: All research findings
            compressed_findings: Compressed and synthesized findings
            deps: Research dependencies

        Returns:
            Complete research report
        """
        # Prepare context for report generation
        context = f"""Generate a comprehensive research report based on:

Research Topic: {brief.topic}

Objectives:
{chr(10).join(f"- {obj}" for obj in brief.objectives)}

Key Questions:
{chr(10).join(f"- {q}" for q in brief.key_questions)}

Compressed Findings Summary:
{compressed_findings.summary}

Key Insights:
{chr(10).join(f"- {insight}" for insight in compressed_findings.key_insights)}

Instructions:
1. Create an engaging title
2. Write a comprehensive executive summary
3. Provide a clear introduction with context
4. Explain the research methodology
5. Organize findings into logical sections
6. Draw clear conclusions
7. Provide actionable recommendations
8. Include all source citations

Ensure the report is professional, comprehensive, and actionable."""

        result = await self.run(context, deps, stream=True)

        # Update research state
        deps.research_state.final_report = result
        deps.research_state.current_stage = ResearchStage.COMPLETED
        deps.research_state.completed_at = datetime.now()

        # Calculate duration
        duration = (
            deps.research_state.completed_at - deps.research_state.started_at
        ).total_seconds()

        # Emit research completed event
        await research_event_bus.emit(
            ResearchCompletedEvent(
                _request_id=deps.research_state.request_id,
                report=result,
                success=True,
                duration_seconds=duration,
            )
        )

        # Emit stage completed event
        await emit_stage_completed(
            request_id=deps.research_state.request_id,
            stage=ResearchStage.REPORT_GENERATION,
            success=True,
            result={"report_sections": len(result.sections)},
        )

        return result


# Register the agent with the coordinator
report_generator_agent = ReportGeneratorAgent()
coordinator.register_agent(report_generator_agent)
