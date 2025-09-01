"""Report generator agent for creating comprehensive research reports."""

from typing import Any

from pydantic_ai import RunContext

from agents.base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
    coordinator,
)
from models.report_generator import ResearchReport

# Global system prompt template for report generation
REPORT_GENERATOR_SYSTEM_PROMPT_TEMPLATE = """
## RESEARCH REPORT SPECIALIST:

You are an expert at synthesizing research findings into comprehensive, well-structured, 
professional reports.

### YOUR ROLE:
1. Organize research findings into logical sections
2. Create clear, informative executive summaries
3. Present findings with appropriate context and analysis
4. Draw meaningful conclusions from the research
5. Provide actionable recommendations
6. Ensure proper citations and references
7. Maintain professional tone and formatting

### REPORT STRUCTURE GUIDELINES:
- **Executive Summary**: Concise overview of key findings and recommendations
- **Introduction**: Context, objectives, and scope
- **Main Sections**: Organized by themes or topics
- **Analysis**: Critical evaluation of findings
- **Conclusions**: Synthesis of key insights
- **Recommendations**: Actionable next steps
- **References**: Proper attribution of sources
- **Appendices**: Supporting materials

### REPORT WRITING PRINCIPLES:
- Clear and well-organized structure
- Data-driven and evidence-based content
- Actionable and practical recommendations
- Professional and authoritative tone
- Accessible to the target audience
- Balanced and objective presentation
- Proper citations and references

### AUDIENCE ADAPTATION:
- **Executive**: Focus on strategic insights and business impact
- **Technical**: Include implementation details and specifications
- **General**: Use clear language and avoid jargon
- **Academic**: Follow scholarly conventions and citation styles

## CURRENT REPORT CONTEXT:
Research Topic: {research_topic}
Target Audience: {target_audience}
Report Format: {report_format}
Key Findings: {key_findings}
{conversation_context}

## REPORT REQUIREMENTS:
- Create a comprehensive research report
- Include executive summary and introduction
- Organize findings into logical sections
- Provide analysis and conclusions
- Offer actionable recommendations
- Include proper references
- Maintain professional quality
"""


class ReportGeneratorAgent(BaseResearchAgent[ResearchDependencies, ResearchReport]):
    """Agent responsible for generating comprehensive research reports.

    This agent synthesizes research findings into well-structured, professional
    reports with clear sections, analysis, and recommendations.
    """

    def __init__(self):
        """Initialize the report generator agent."""
        config = AgentConfiguration(
            agent_name="report_generator",
            agent_type="synthesis",
        )
        super().__init__(config=config)

        # Register dynamic instructions
        @self.agent.instructions
        async def add_report_context(ctx: RunContext[ResearchDependencies]) -> str:  # pyright: ignore
            """Inject report generation context as instructions."""
            query = ctx.deps.research_state.user_query
            metadata = ctx.deps.research_state.metadata or {}
            conversation = metadata.get("conversation_messages", [])
            research_topic = metadata.get("research_topic", query)
            target_audience = metadata.get("target_audience", "general")
            report_format = metadata.get("report_format", "standard")
            key_findings = metadata.get("key_findings", "")

            # Format conversation context
            conversation_context = self._format_conversation_context(conversation)

            # Use global template with variable substitution
            return REPORT_GENERATOR_SYSTEM_PROMPT_TEMPLATE.format(
                research_topic=research_topic,
                target_audience=target_audience,
                report_format=report_format,
                key_findings=key_findings,
                conversation_context=conversation_context,
            )

        # Register report generation tools
        @self.agent.tool
        async def structure_content(
            ctx: RunContext[ResearchDependencies], content: dict[str, Any]
        ) -> dict[str, list[Any]]:  # pyright: ignore
            """Structure content into report sections.

            Args:
                content: Dictionary of content to structure

            Returns:
                Structured content dictionary
            """
            structured = {
                "introduction": [],
                "background": [],
                "findings": [],
                "analysis": [],
                "conclusions": [],
                "recommendations": [],
            }

            # Categorize content based on keywords
            categorization_rules = {
                "introduction": ["overview", "purpose", "objective", "scope"],
                "background": ["history", "context", "previous", "existing"],
                "findings": ["found", "discovered", "identified", "observed", "results"],
                "analysis": ["analysis", "evaluation", "comparison", "assessment"],
                "conclusions": ["conclude", "summary", "overall", "final"],
                "recommendations": ["recommend", "suggest", "should", "propose", "advise"],
            }

            for key, value in content.items():
                value_str = str(value).lower()
                categorized = False

                for section, keywords in categorization_rules.items():
                    if any(keyword in value_str for keyword in keywords):
                        structured[section].append(value)
                        categorized = True
                        break

                if not categorized:
                    structured["findings"].append(value)

            return structured

        @self.agent.tool
        async def generate_executive_summary(
            ctx: RunContext[ResearchDependencies], findings: list[str], recommendations: list[str]
        ) -> str:  # pyright: ignore
            """Generate an executive summary from findings and recommendations.

            Args:
                findings: List of key findings
                recommendations: List of recommendations

            Returns:
                Executive summary text
            """
            summary_parts = []

            # Start with overview
            summary_parts.append(
                "This research report presents comprehensive findings and actionable recommendations."
            )

            # Add key findings
            if findings:
                top_findings = findings[:3]  # Top 3 findings
                findings_text = "Key findings include: " + "; ".join(str(f) for f in top_findings)
                summary_parts.append(findings_text)

            # Add primary recommendations
            if recommendations:
                top_recommendations = recommendations[:2]  # Top 2 recommendations
                rec_text = "Primary recommendations: " + " and ".join(
                    str(r) for r in top_recommendations
                )
                summary_parts.append(rec_text)

            # Add conclusion
            summary_parts.append(
                "The report provides detailed analysis and supporting evidence for all findings."
            )

            return " ".join(summary_parts)

        @self.agent.tool
        async def format_citations(
            ctx: RunContext[ResearchDependencies], sources: list[dict[str, Any]]
        ) -> list[str]:  # pyright: ignore
            """Format citations in a consistent style.

            Args:
                sources: List of source information

            Returns:
                List of formatted citations
            """
            formatted_citations = []

            for source in sources:
                if isinstance(source, dict):
                    # Build citation from components
                    parts = []

                    if source.get("author"):
                        parts.append(source["author"])

                    if source.get("date"):
                        parts.append(f"({source['date']})")

                    if source.get("title"):
                        parts.append(f'"{source["title"]}"')

                    if source.get("url"):
                        parts.append(f"Available at: {source['url']}")

                    if parts:
                        formatted_citations.append(". ".join(parts))
                else:
                    # Use as-is if not a dictionary
                    formatted_citations.append(str(source))

            return formatted_citations

        @self.agent.tool
        async def assess_report_completeness(
            ctx: RunContext[ResearchDependencies], report_sections: dict[str, Any]
        ) -> dict[str, Any]:  # pyright: ignore
            """Assess the completeness of a report.

            Args:
                report_sections: Dictionary of report sections

            Returns:
                Completeness assessment
            """
            required_sections = [
                "executive_summary",
                "introduction",
                "findings",
                "conclusions",
                "recommendations",
            ]

            assessment = {
                "complete_sections": [],
                "missing_sections": [],
                "weak_sections": [],
                "completeness_score": 0.0,
            }

            for section in required_sections:
                if section in report_sections:
                    content = report_sections[section]
                    if content and len(str(content)) > 50:
                        assessment["complete_sections"].append(section)
                    else:
                        assessment["weak_sections"].append(section)
                else:
                    assessment["missing_sections"].append(section)

            # Calculate completeness score
            total_sections = len(required_sections)
            complete_count = len(assessment["complete_sections"])
            weak_count = len(assessment["weak_sections"])

            assessment["completeness_score"] = (complete_count + 0.5 * weak_count) / total_sections

            return assessment

    def _format_conversation_context(self, conversation: list[Any]) -> str:
        """Format conversation history for the prompt."""
        if not conversation:
            return "No prior conversation context."

        formatted = []
        for msg in conversation[-3:]:  # Last 3 messages for context
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                formatted.append(f"{role.capitalize()}: {content}")
            else:
                formatted.append(str(msg))

        return "Recent Conversation:\n" + "\n".join(formatted)

    def _get_default_system_prompt(self) -> str:
        """Get the base system prompt for this agent."""
        return "You are a Research Report Specialist focused on creating comprehensive reports."

    def _get_output_type(self) -> type[ResearchReport]:
        """Get the output type for this agent."""
        return ResearchReport


# Register the agent with the coordinator
report_generator_agent = ReportGeneratorAgent()
coordinator.register_agent(report_generator_agent)
