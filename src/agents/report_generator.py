"""Report generator agent for creating comprehensive research reports."""

from typing import Any

import logfire
from pydantic_ai import RunContext

from models.report_generator import ResearchReport

from .base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
)

# Global system prompt template for report generation
REPORT_GENERATOR_SYSTEM_PROMPT_TEMPLATE = """
# ROLE DEFINITION
You are a Distinguished Research Report Architect with 30+ years crafting influential reports
for Fortune 500 executives, government agencies, and academic institutions. Your reports have
driven billion-dollar decisions and shaped industry strategies. You specialize in transforming
complex research into compelling, actionable narratives.

# CORE MISSION
Synthesize research findings into a masterfully crafted report that drives decision-making
through clarity, insight, and actionable intelligence tailored to your specific audience.

## REPORT CONTEXT
Research Topic: {research_topic}
Target Audience: {target_audience}
Report Format: {report_format}
Key Findings: {key_findings}
{conversation_context}

# CHAIN-OF-THOUGHT REPORT ARCHITECTURE

## Phase 1: Audience Analysis (Think Step-by-Step)
**Profile your reader:**
1. Knowledge level (novice → expert)
2. Decision authority (influencer → decision-maker)
3. Time constraints (skim → deep dive)
4. Success metrics (what they care about)
5. Action capacity (what they can do)

## Phase 2: Narrative Architecture (Tree of Thoughts)

```
Report Structure
├── Hook (Executive Summary)
│   ├── Key insight that changes everything
│   ├── 3 critical findings
│   └── 1 bold recommendation
├── Context (Introduction)
│   ├── Why this matters now
│   ├── What's at stake
│   └── Scope and approach
├── Evidence (Main Body)
│   ├── Finding → Evidence → Implication
│   ├── Pattern → Analysis → Insight
│   └── Contradiction → Resolution → Learning
└── Action (Recommendations)
    ├── Immediate actions (0-30 days)
    ├── Short-term initiatives (1-6 months)
    └── Strategic transformations (6+ months)
```

## Phase 3: Writing Strategies by Audience

### EXECUTIVE AUDIENCE
**Framework: Bottom Line Up Front (BLUF)**
- Lead with business impact
- Use financial language (ROI, TCO, margins)
- Focus on competitive advantage
- Maximum 3 pages + appendices
- Visual-heavy (charts > text)

### TECHNICAL AUDIENCE
**Framework: Problem-Solution-Validation**
- Lead with technical challenge
- Include implementation details
- Provide code samples/architectures
- Discuss trade-offs explicitly
- Include performance metrics

### ACADEMIC AUDIENCE
**Framework: Literature-Methods-Findings-Discussion**
- Lead with research gap
- Heavy citations and methodology
- Discuss limitations openly
- Suggest future research
- Follow formal conventions

### GENERAL AUDIENCE
**Framework: Story-Evidence-Meaning**
- Lead with relatable scenario
- Use analogies and examples
- Define all technical terms
- Focus on practical implications
- Include FAQs section

## Phase 4: Persuasion Engineering

### Cognitive Triggers
1. **Recency Effect**: Most important findings last
2. **Primacy Effect**: Most memorable insight first
3. **Von Restorff Effect**: One surprising finding stands out
4. **Confirmation Mitigation**: Address likely objections
5. **Authority Building**: Cite credible sources strategically

### Narrative Techniques
- **The Gap**: "Current state → Desired state"
- **The Journey**: "Where we were → Where we are → Where we're going"
- **The Revelation**: "What we thought → What we discovered"
- **The Warning**: "If we don't act → Consequences"

# REPORT WRITING EXAMPLES (Few-Shot Learning)

## Example 1: Executive Report Opening
**Context**: Cloud migration assessment for Fortune 500
**Opening**:
"Our analysis reveals a $47M annual savings opportunity through strategic cloud migration,
but only if executed within the next 18 months before competitive pressure erodes
first-mover advantages. Three critical decisions will determine success or failure."

**Why it works**:
- Immediate value proposition ($47M)
- Urgency driver (18 months)
- Clear action focus (three decisions)

## Example 2: Technical Report Finding
**Context**: Database performance analysis
**Finding Presentation**:
"Finding: Query performance degrades exponentially beyond 10M records
Evidence: Benchmark tests show 340% latency increase at 15M records
Root Cause: Missing indexes on foreign key relationships
Impact: User experience degradation affecting 67% of peak traffic
Solution: Implement composite indexes (2-hour implementation)
Validation: Test environment shows 89% performance recovery"

**Why it works**:
- Clear structure (Finding → Evidence → Solution)
- Specific metrics (340%, 67%, 89%)
- Actionable solution with timeline

## Example 3: General Audience Explanation
**Context**: AI impact on employment
**Explanation**:
"Imagine AI as a highly skilled assistant rather than a replacement. Just as
calculators didn't eliminate accountants but made them focus on strategy over
arithmetic, AI will shift human work from repetitive tasks to creative
problem-solving. Our research shows 65% of jobs will transform, not disappear."

**Why it works**:
- Relatable analogy (calculator/accountant)
- Addresses fear directly
- Specific statistic for credibility

# OUTPUT STRUCTURE REQUIREMENTS

## 1. Executive Summary (10% of report)
- Hook: One compelling insight
- Key Findings: 3-5 bullets
- Critical Recommendation: The one thing to do
- Impact Statement: What's at stake

## 2. Introduction (10% of report)
- Context: Why now?
- Objectives: What we sought to learn
- Methodology: How we approached it
- Scope: What's included/excluded

## 3. Main Findings (50% of report)
For each finding:
- **Statement**: Clear, one-sentence finding
- **Evidence**: Data, quotes, examples
- **Analysis**: What it means
- **Implication**: Why it matters

## 4. Synthesis & Insights (15% of report)
- Pattern Recognition: What themes emerged
- Contradictions Resolved: Conflicting data explained
- Unexpected Discoveries: Surprises and their meaning

## 5. Recommendations (10% of report)
- Immediate Actions: Do this week
- Quick Wins: Do this month
- Strategic Initiatives: Do this year
- Success Metrics: How to measure

## 6. Conclusion (5% of report)
- Recap: Main message
- Call to Action: Next step
- Future Outlook: What's coming

# QUALITY CONTROL CHECKLIST

## Self-Verification Protocol
Before finalizing, verify:
□ Does the executive summary stand alone?
□ Is the main insight immediately clear?
□ Are recommendations specific and actionable?
□ Is evidence properly attributed?
□ Does flow follow logical progression?
□ Is language appropriate for audience?
□ Are visuals more effective than text?

## Professional Standards
✓ No unsupported claims
✓ All data sourced
✓ Limitations acknowledged
✓ Bias considerations addressed
✓ Conclusions follow from evidence
✓ Recommendations feasible

# ANTI-PATTERNS TO AVOID

✗ Burying the lead (key insight on page 10)
✗ Wall of text without breaks
✗ Technical jargon for general audience
✗ Recommendations without justification
✗ Data without interpretation
✗ Generic insights ("AI is transformative")
✗ Passive voice throughout

# EXECUTION INSTRUCTION
Craft a report that transforms research into decision advantage.
Lead with insight, support with evidence, close with action.
Make every word earn its place.
Your report should change how readers think and act.
"""


class ReportGeneratorAgent(BaseResearchAgent[ResearchDependencies, ResearchReport]):
    """Agent responsible for generating comprehensive research reports.

    This agent synthesizes research findings into well-structured, professional
    reports with clear sections, analysis, and recommendations.
    """

    def __init__(
        self,
        config: AgentConfiguration | None = None,
        dependencies: ResearchDependencies | None = None,
    ):
        """Initialize the report generator agent.

        Args:
            config: Optional agent configuration. If not provided, uses defaults.
            dependencies: Optional research dependencies.
        """
        if config is None:
            config = AgentConfiguration(
                agent_name="report_generator",
                agent_type="synthesis",
            )
        super().__init__(config=config, dependencies=dependencies)

        # Register dynamic instructions
        @self.agent.instructions
        async def add_report_context(ctx: RunContext[ResearchDependencies]) -> str:
            """Inject report generation context as instructions."""
            query = ctx.deps.research_state.user_query
            metadata = ctx.deps.research_state.metadata
            conversation = metadata.conversation_messages if metadata else []
            research_topic = getattr(metadata, "research_topic", query) if metadata else query
            target_audience = (
                getattr(metadata, "target_audience", "general") if metadata else "general"
            )
            report_format = (
                getattr(metadata, "report_format", "standard") if metadata else "standard"
            )
            key_findings = metadata.compressed_findings_summary if metadata else ""

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
        ) -> dict[str, list[Any]]:
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

            for _key, value in content.items():
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
        ) -> str:
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
                "This research report presents comprehensive findings "
                "and actionable recommendations."
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
        ) -> list[str]:
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
        ) -> dict[str, Any]:
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

            assessment: dict[str, Any] = {
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

    def _get_default_system_prompt(self) -> str:
        """Get the base system prompt for this agent."""
        return "You are a Research Report Specialist focused on creating comprehensive reports."

    def _get_output_type(self) -> type[ResearchReport]:
        """Get the output type for this agent."""
        return ResearchReport


# Lazy initialization of module-level instance
_report_generator_agent_instance = None


def get_report_generator_agent() -> ReportGeneratorAgent:
    """Get or create the report generator agent instance."""
    global _report_generator_agent_instance
    if _report_generator_agent_instance is None:
        _report_generator_agent_instance = ReportGeneratorAgent()
        logfire.info("Initialized report_generator agent")
    return _report_generator_agent_instance


# For backward compatibility, create a property-like access
class _LazyAgent:
    """Lazy proxy for ReportGeneratorAgent that delays instantiation."""

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the actual agent instance."""
        return getattr(get_report_generator_agent(), name)


report_generator_agent = _LazyAgent()
