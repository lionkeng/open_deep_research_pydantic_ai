"""Report generator agent for creating comprehensive research reports."""

import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import logfire
from pydantic_ai import RunContext

from models.metadata import ReportSectionPlan
from models.report_generator import ResearchReport
from models.research_executor import ResearchResults, ResearchSource
from services.source_repository import summarize_sources_for_prompt

from .base import AgentConfiguration, BaseResearchAgent, ResearchDependencies
from .report_clean_merge import (
    record_clean_merge_metrics,
    run_clean_merge,
)

# Global system prompt template for report generation
MINIMUM_CITATIONS = 3
PREFERRED_MAX_CITATIONS = 5

CITATION_CONTRACT_TEMPLATE = """### Citation Contract
- Cite every substantive statement with the matching source marker using the form `[Sx]`.
- Reuse the same marker whenever you refer to the same source; never invent IDs.
- When at least {min_citations} sources are available, include at least {min_citations} distinct
  citations. Aim for {preferred_citations} when enough sources exist.
- Do not fabricate URLs or titles. Only cite sources provided in the context below.
- Rely on `[Sx]` markers throughout; the system converts them into numbered footnotes. Do not
  append a manual `## Sources` section.
"""

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
Sources Overview:
{source_overview}

{citation_contract}
Section Outline Guidance:
{section_outline_instructions}

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
"Query performance collapses once the dataset exceeds 10 million rows, "
"with benchmark tests showing a 340% latency surge at 15 million records. "
"The drag traces back to missing indexes on critical foreign keys, starving the "
"optimizer of fast lookup paths. Implementing the recommended composite indexes "
"takes roughly two hours, and validation in the staging environment already "
"recovers 89% of lost performance while restoring a smooth experience for the 67% "
"of traffic that hits the affected queries."

**Why it works**:
- Natural prose stitches together evidence, cause, and impact.
- Specific metrics (340%, 67%, 89%) ground the narrative.
- Actionable and time-bound solution builds confidence.

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

# STYLE & VOICE PRINCIPLES

- Write in polished, confident prose suitable for senior decision-makers.
- Use cohesive transitions that tie facts to implications; weave evidence into sentences
  instead of presenting isolated data.
- Favor active voice and varied sentence lengths to maintain rhythm and clarity.
- Rephrase the blueprint requirements naturally inside the report. Never repeat
  instruction labels verbatim (avoid phrases such as "Most important insight:" or
  "Statement:").
- Treat the Section Outline as scaffolding—elevate each bullet into polished prose
  instead of restating labels.
- Introduce bullet lists with a natural lead-in sentence; keep bullets concise and
  evidence-backed.
- Build paragraphs around clear topic sentences and close each with a
  forward-looking or interpretive statement to maintain flow.
- Present the report as a finished deliverable with no references to these
  instructions or to the writing process.

# OUTPUT STRUCTURE REQUIREMENTS

## 1. Executive Summary (10% of report)
- Open with a decisive insight in the first sentence; do not preface it with labels.
- Provide a bridging sentence that sets up a bullet list of 3-5 findings grounded in
  evidence.
- Deliver one critical recommendation as a complete sentence that begins with a strong
  verb rather than a label.
- Close with an impact-focused sentence that quantifies or qualifies what is at stake.

## 2. Introduction (10% of report)
- Explain why the topic is urgent now using 2-3 sentences with inline citations.
- Clarify what the research set out to learn in polished prose rather than colon-labeled fragments.
- Describe the methodology and data sources succinctly, emphasizing credibility.
- Define the scope and major exclusions within the narrative to calibrate expectations.

## 3. Main Findings (50% of report)
For each finding:
- Use the corresponding heading from the Section Outline; refine wording only for
  clarity while preserving the intended focus.
- Integrate every bullet from the Section Outline into complete sentences; avoid
  label prefixes such as "Implication:" or "Takeaway:".
- Weave evidence, analysis, and implications into flowing paragraphs that move
  logically from data to meaning.
- Use inline citations directly after the facts they support.
- Conclude each finding with a forward-looking takeaway that links to decisions the
  audience cares about.

## 4. Synthesis & Insights (15% of report)
- Highlight cross-cutting patterns in polished narrative form.
- Address how apparent contradictions resolve, explaining the nuance for the reader.
- Call out unexpected discoveries and unpack their significance for strategy or policy.

## 5. Recommendations (10% of report)
- Organize actions by time horizon (immediate, 1-6 months, 6+ months) with succinct
  bullet lists introduced by natural phrasing.
- Make every recommendation action-oriented, specific, and tied to cited evidence or findings.
- Include success metrics that make it clear how progress should be measured.

## 6. Conclusion (5% of report)
- Restate the overarching message in one cohesive sentence.
- Provide a decisive call to action that aligns with the recommendations.
- Offer a forward-looking statement that previews what to monitor next, keeping prose fluid.

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


_HEADING_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
_GENERIC_HEADING_PREFIXES = {
    "finding",
    "insight",
    "observation",
    "pattern",
    "analysis",
    "detail",
    "summary",
    "section",
    "topic",
    "theme",
    "point",
    "note",
}
_LABEL_PREFIXES = {
    "finding",
    "insight",
    "implication",
    "analysis",
    "observation",
    "statement",
    "takeaway",
    "summary",
    "key finding",
    "key insight",
    "key takeaway",
    "impact",
    "recommendation",
    "action",
    "risk",
    "opportunity",
}
_BAD_HEADING_RATIO_THRESHOLD = 0.25
_MIN_HEADING_OVERLAP = 0.3
_MAX_COLON_PREFIXES = 2


@dataclass(slots=True)
class SectionQuality:
    """Quality metrics for a single report section."""

    title: str
    overlap_ratio: float
    is_generic: bool
    colon_prefixes: int


@dataclass(slots=True)
class ReportQualityEvaluation:
    """Aggregated heuristics for the generated report."""

    sections: list[SectionQuality]
    total_colon_prefixes: int

    @property
    def bad_heading_ratio(self) -> float:
        if not self.sections:
            return 0.0
        bad = sum(
            1
            for section in self.sections
            if section.is_generic or section.overlap_ratio < _MIN_HEADING_OVERLAP
        )
        return bad / len(self.sections)

    @property
    def passed(self) -> bool:
        return (
            self.bad_heading_ratio <= _BAD_HEADING_RATIO_THRESHOLD
            and self.total_colon_prefixes <= _MAX_COLON_PREFIXES
        )


def _iter_report_sections(sections: Iterable[Any]) -> Iterable[Any]:
    for section in sections:
        yield section
        subsections = getattr(section, "subsections", None)
        if subsections:
            yield from _iter_report_sections(subsections)


def _tokenize(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in _HEADING_TOKEN_PATTERN.finditer(text or "")
        if len(match.group(0)) > 2
    }


def _heading_overlap_ratio(title: str, content: str) -> float:
    title_tokens = _tokenize(title)
    if not title_tokens:
        return 0.0
    content_tokens = set(list(_tokenize(content))[:60])
    if not content_tokens:
        return 0.0
    overlap = title_tokens & content_tokens
    return len(overlap) / max(len(title_tokens), 1)


def _is_generic_heading(title: str) -> bool:
    simplified = title.strip().lower()
    if not simplified:
        return True
    if len(simplified.split()) <= 1:
        return True
    if re.match(r"^(finding|insight|observation|pattern|section|theme)\s*\d+", simplified):
        return True
    for prefix in _GENERIC_HEADING_PREFIXES:
        if simplified.startswith(prefix + " ") or simplified == prefix:
            return True
    return False


def _count_colon_prefixed_labels(content: str) -> int:
    count = 0
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line.lstrip("-• ")
        if ":" not in line:
            continue
        prefix, _ = line.split(":", 1)
        prefix = prefix.lower().strip()
        if not prefix or len(prefix) > 40:
            continue
        if any(prefix.startswith(label) for label in _LABEL_PREFIXES):
            count += 1
    return count


def _evaluate_report_quality(report: ResearchReport) -> ReportQualityEvaluation:
    sections = []
    total_colon_prefixes = 0
    for section in _iter_report_sections(report.sections):
        title = getattr(section, "title", "") or ""
        content = getattr(section, "content", "") or ""
        overlap = _heading_overlap_ratio(title, content)
        generic = _is_generic_heading(title)
        colon_prefixes = _count_colon_prefixed_labels(content)
        sections.append(
            SectionQuality(
                title=title,
                overlap_ratio=overlap,
                is_generic=generic,
                colon_prefixes=colon_prefixes,
            )
        )
        total_colon_prefixes += colon_prefixes

    return ReportQualityEvaluation(sections=sections, total_colon_prefixes=total_colon_prefixes)


def _record_quality_metrics(
    evaluation: ReportQualityEvaluation,
    deps: ResearchDependencies | None,
) -> None:
    payload = {
        "bad_heading_ratio": round(evaluation.bad_heading_ratio, 3),
        "total_colon_prefixes": evaluation.total_colon_prefixes,
        "section_count": len(evaluation.sections),
        "passed": evaluation.passed,
    }
    logfire.info("Report quality evaluation", **payload)

    metrics = getattr(deps, "metrics_collector", None)
    if metrics is not None:
        metrics.record_quality_metric("report_bad_heading_ratio", evaluation.bad_heading_ratio)
        metrics.record_quality_metric("report_colon_prefixes", evaluation.total_colon_prefixes)
        metrics.record_quality_metric("report_validation_passed", evaluation.passed)


def _adjust_outline_for_retry(metadata: Any) -> bool:
    outline = getattr(getattr(metadata, "report", None), "section_outline", None)
    if not outline:
        return False

    changed = False
    for item in outline:
        if not isinstance(item, ReportSectionPlan):
            continue
        title = item.title or ""
        bullet = next((b for b in getattr(item, "bullets", []) if b.strip()), "")
        addition = _sanitize_outline_phrase(bullet)
        if addition and addition.lower() not in title.lower():
            item.title = f"{title} – {addition}" if title else addition
            changed = True
        if getattr(item, "bullets", None):
            expanded = [_expand_bullet_phrase(b) for b in item.bullets]
            item.bullets = [b for b in expanded if b]
    return changed


def _sanitize_outline_phrase(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\[S\d+\]", "", text).strip().rstrip(".;:")
    if len(cleaned) > 60:
        cleaned = cleaned[:60].rsplit(" ", 1)[0]
    return cleaned


def _expand_bullet_phrase(text: str) -> str:
    cleaned = _sanitize_outline_phrase(text)
    if not cleaned:
        return ""
    if cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


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
            research_results = ctx.deps.research_state.research_results
            if research_results and research_results.executive_summary:
                summary_findings = research_results.executive_summary.key_findings or []
            elif research_results:
                summary_findings = research_results.key_insights
            else:
                summary_findings = []
            key_findings = "; ".join(summary_findings[:3]) if summary_findings else ""

            report_metadata = getattr(metadata, "report", None) if metadata else None
            section_outline = list(getattr(report_metadata, "section_outline", []) or [])

            if section_outline:
                outline_lines: list[str] = []

                for index, item in enumerate(section_outline, start=1):
                    heading = getattr(item, "title", "").strip() or "Untitled Section"
                    outline_lines.append(f"{index}. {heading}")

                    bullets = [b.strip() for b in getattr(item, "bullets", []) if b.strip()]
                    for bullet in bullets[:3]:
                        outline_lines.append(f"    - {bullet}")

                    evidence_ids = [
                        evid.strip()
                        for evid in getattr(item, "salient_evidence_ids", [])
                        if evid and evid.strip()
                    ]
                    if evidence_ids:
                        outline_lines.append("    Evidence IDs: " + ", ".join(evidence_ids[:6]))

                outline_instructions = (
                    "Section Outline (use these as section headings; "
                    "refine wording only for clarity):\n" + "\n".join(outline_lines)
                )
            else:
                outline_instructions = (
                    "Section Outline (use these as section headings; "
                    "refine wording only for clarity):\n"
                    "(no outline provided)"
                )

            available_sources = (
                len(research_results.sources)
                if research_results and research_results.sources
                else 0
            )
            if available_sources:
                min_for_contract = (
                    MINIMUM_CITATIONS
                    if available_sources >= MINIMUM_CITATIONS
                    else available_sources
                )
                pref_for_contract = (
                    PREFERRED_MAX_CITATIONS
                    if available_sources >= PREFERRED_MAX_CITATIONS
                    else available_sources
                )
                if pref_for_contract < min_for_contract:
                    pref_for_contract = min_for_contract
                source_overview = summarize_sources_for_prompt(
                    research_results.sources[:10] if research_results else []
                )
            else:
                min_for_contract = MINIMUM_CITATIONS
                pref_for_contract = PREFERRED_MAX_CITATIONS
                source_overview = (
                    "- No sources registered yet. Cite evidence immediately once available."
                )
            citation_contract = CITATION_CONTRACT_TEMPLATE.format(
                min_citations=min_for_contract,
                preferred_citations=pref_for_contract,
            )

            # Format conversation context
            conversation_context = self._format_conversation_context(conversation)

            # Use global template with variable substitution
            return REPORT_GENERATOR_SYSTEM_PROMPT_TEMPLATE.format(
                research_topic=research_topic,
                target_audience=target_audience,
                report_format=report_format,
                key_findings=key_findings,
                source_overview=source_overview,
                citation_contract=citation_contract,
                conversation_context=conversation_context,
                section_outline_instructions=outline_instructions,
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

    async def run(
        self,
        deps: ResearchDependencies | None = None,
        message_history: list[Any] | None = None,
        stream: bool = False,
    ) -> ResearchReport:
        """Run the report generator with validation, retry, and clean-merge handling."""

        actual_deps = deps or self.dependencies
        metadata = actual_deps.research_state.metadata if actual_deps else None

        attempt = 0
        max_attempts = 2
        outline_adjusted = False
        report: ResearchReport | None = None
        evaluation: ReportQualityEvaluation | None = None

        while attempt < max_attempts:
            attempt += 1
            report = await super().run(deps=deps, message_history=message_history, stream=stream)

            evaluation = _evaluate_report_quality(report)
            _record_quality_metrics(evaluation, actual_deps)

            if evaluation.passed:
                break

            if attempt >= max_attempts:
                logfire.warning(
                    "Report quality validation failed after retries",
                    bad_heading_ratio=round(evaluation.bad_heading_ratio, 3),
                    colon_prefixes=evaluation.total_colon_prefixes,
                )
                break

            if not outline_adjusted and metadata and _adjust_outline_for_retry(metadata):
                outline_adjusted = True
                logfire.info(
                    "Retrying report generation with expanded outline",
                    attempt=attempt + 1,
                    bad_heading_ratio=round(evaluation.bad_heading_ratio, 3),
                    colon_prefixes=evaluation.total_colon_prefixes,
                )
                continue

            logfire.warning(
                "Report quality validation failing and no retry available",
                attempt=attempt,
                bad_heading_ratio=round(evaluation.bad_heading_ratio, 3),
                colon_prefixes=evaluation.total_colon_prefixes,
            )

            break

        if report is None:
            raise RuntimeError("Report generation returned no result")

        if actual_deps and getattr(actual_deps, "enable_llm_clean_merge", False):
            logfire.info("Clean-merge enabled; invoking guardrailed pass")
            try:
                report, metrics = await run_clean_merge(deps=actual_deps, report=report)
                record_clean_merge_metrics(metrics=metrics, deps=actual_deps)
            except Exception as exc:  # pragma: no cover - safety fallback
                logfire.warning("Clean-merge failed; keeping original", error=str(exc))

        if actual_deps:
            report = self._apply_citation_postprocessing(report, actual_deps)
        return report

    def _apply_citation_postprocessing(
        self, report: ResearchReport, deps: ResearchDependencies
    ) -> ResearchReport:
        """Convert source markers into footnotes and enforce minimum citation coverage."""

        research_results: ResearchResults | None = getattr(
            deps.research_state, "research_results", None
        )
        if not research_results or not research_results.sources:
            return report

        redundant_titles = {"sources", "sources and citations"}

        def _filter_sections(sections: list[Any]) -> list[Any]:
            filtered: list[Any] = []
            for section in sections:
                title = getattr(section, "title", "")
                if str(title).strip().lower() in redundant_titles:
                    continue
                if getattr(section, "subsections", None):
                    section.subsections = [
                        subsection
                        for subsection in section.subsections
                        if str(getattr(subsection, "title", "")).strip().lower()
                        not in redundant_titles
                    ]
                filtered.append(section)
            return filtered

        report.sections = _filter_sections(report.sections)

        source_map: dict[str, ResearchSource] = {
            source.source_id: source for source in research_results.sources if source.source_id
        }
        if not source_map:
            return report

        marker_pattern = re.compile(r"\[S(\d+)\]")

        def collect_markers(text: str | None) -> set[str]:
            if not text:
                return set()
            return {f"S{match.group(1)}" for match in marker_pattern.finditer(text)}

        text_targets: list[tuple[Any, str]] = [
            (report, "executive_summary"),
            (report, "introduction"),
            (report, "conclusions"),
        ]
        for section in report.sections:
            text_targets.append((section, "content"))
            for subsection in section.subsections:
                text_targets.append((subsection, "content"))

        marker_set: set[str] = set()
        for obj, attr in text_targets:
            marker_set.update(collect_markers(getattr(obj, attr, "")))
        for rec in report.recommendations:
            marker_set.update(collect_markers(rec))
        for appendix_value in report.appendices.values():
            marker_set.update(collect_markers(appendix_value))

        available_sources = len(source_map)
        if available_sources >= PREFERRED_MAX_CITATIONS:
            required_citations = PREFERRED_MAX_CITATIONS
        elif available_sources >= MINIMUM_CITATIONS:
            required_citations = MINIMUM_CITATIONS
        else:
            required_citations = max(available_sources, MINIMUM_CITATIONS)

        if len(marker_set) < required_citations:
            unused_sources = [sid for sid in source_map if sid not in marker_set]
            needed = min(required_citations - len(marker_set), len(unused_sources))
            if needed > 0 and unused_sources:
                selected = unused_sources[:needed]
                addition = (
                    "Additional supporting sources: "
                    + ", ".join(f"[{sid}]" for sid in selected)
                    + "."
                )
                report.conclusions = (
                    (report.conclusions.strip() + "\n\n") if report.conclusions else ""
                ) + addition
                marker_set.update(selected)
                text_targets.append((report, "conclusions"))

        ordered_markers: OrderedDict[str, int] = OrderedDict()

        def register_markers(text: str | None) -> None:
            if not text:
                return
            for match in marker_pattern.finditer(text):
                source_id = f"S{match.group(1)}"
                if source_id in source_map and source_id not in ordered_markers:
                    ordered_markers[source_id] = len(ordered_markers) + 1

        for obj, attr in text_targets:
            register_markers(getattr(obj, attr, ""))
        for rec in report.recommendations:
            register_markers(rec)
        for appendix_value in report.appendices.values():
            register_markers(appendix_value)

        if not ordered_markers:
            logfire.warning("No citations detected after enforcement", report_title=report.title)
            return report

        def replacement(match: re.Match[str]) -> str:
            source_id = f"S{match.group(1)}"
            number = ordered_markers.get(source_id)
            return match.group(0) if number is None else f"[{number}]"

        for obj, attr in text_targets:
            current = getattr(obj, attr, "")
            if current:
                updated = marker_pattern.sub(replacement, str(current))
                setattr(obj, attr, updated)

        report.recommendations = [
            marker_pattern.sub(replacement, str(rec)) for rec in report.recommendations
        ]
        for key, value in list(report.appendices.items()):
            report.appendices[key] = marker_pattern.sub(replacement, str(value))

        footnotes: list[str] = []
        source_summary: list[dict[str, Any]] = []
        for source_id, footnote_number in ordered_markers.items():
            source = source_map[source_id]
            descriptor = source.title
            if source.publisher:
                descriptor += f" — {source.publisher}"
            if source.date:
                descriptor += f" ({source.date.strftime('%Y-%m-%d')})"
            if source.url:
                descriptor += f" <{source.url}>"
            footnotes.append(f"[{footnote_number}]: {descriptor}")
            source_summary.append(
                {
                    "id": source_id,
                    "title": source.title,
                    "url": source.url or "",
                    "publisher": source.publisher or "",
                    "used": True,
                    "footnote_number": footnote_number,
                }
            )

        report.references = footnotes
        for source_id, source in source_map.items():
            if source_id in ordered_markers:
                continue
            source_summary.append(
                {
                    "id": source_id,
                    "title": source.title,
                    "url": source.url or "",
                    "publisher": source.publisher or "",
                    "used": False,
                }
            )
        report.metadata.source_summary = source_summary

        for source_id in ordered_markers:
            research_results.record_usage(source_id, report_section="final_report")

        audit_result = self._audit_citations(
            ordered_markers=ordered_markers,
            source_map=source_map,
            required_citations=required_citations,
        )
        report.metadata.citation_audit = audit_result

        quality_metrics = getattr(research_results.synthesis_metadata, "quality_metrics", None)
        if quality_metrics is not None:
            quality_metrics["citation_coverage"] = audit_result.get("coverage", 0.0)
            quality_metrics["citation_audit_status"] = audit_result.get("status")

        if audit_result.get("status") != "pass":
            logfire.warning(
                "Citation audit warnings detected",
                status=audit_result.get("status"),
                orphaned=audit_result.get("orphaned_sources"),
            )

        return report

    def _audit_citations(
        self,
        *,
        ordered_markers: OrderedDict[str, int],
        source_map: dict[str, ResearchSource],
        required_citations: int,
    ) -> dict[str, Any]:
        total_sources = len(source_map)
        cited_sources = len(ordered_markers)
        orphaned_sources = [sid for sid in source_map if sid not in ordered_markers]
        coverage = cited_sources / max(total_sources, 1)
        contiguous = list(ordered_markers.values()) == list(range(1, cited_sources + 1))
        required_met = cited_sources >= min(required_citations, total_sources)

        status = "pass"
        if not contiguous or not required_met:
            status = "warn"
        if total_sources and coverage < 0.4:
            status = "warn"

        return {
            "status": status,
            "coverage": coverage,
            "total_sources": total_sources,
            "cited_sources": cited_sources,
            "required_met": required_met,
            "contiguous": contiguous,
            "orphaned_sources": orphaned_sources,
        }

    @staticmethod
    def _extract_markers(text: str | None) -> set[str]:
        if not text:
            return set()
        pattern = re.compile(r"\[S(\d+)\]")
        return {f"S{m.group(1)}" for m in pattern.finditer(text)}

    @staticmethod
    def _markers_equal(a: str | None, b: str | None) -> bool:
        return ReportGeneratorAgent._extract_markers(a) == ReportGeneratorAgent._extract_markers(b)

    # --------------------
    # Style normalization
    # --------------------


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
