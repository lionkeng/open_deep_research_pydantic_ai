"""Research executor agent for conducting actual research."""

from datetime import datetime
from typing import Any

import logfire
from pydantic_ai import RunContext

from models.research_executor import ResearchResults

from .base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
)

# Global system prompt template for research execution
RESEARCH_EXECUTOR_SYSTEM_PROMPT_TEMPLATE = """
# ROLE DEFINITION
You are a Principal Research Scientist with 25+ years of expertise in systematic research,
data synthesis, and evidence-based analysis. You hold advanced degrees in Information Science
and have led research teams at top institutions. Your expertise spans academic research,
industry analysis, and strategic intelligence gathering.

# CORE MISSION
Execute comprehensive, systematic research that transforms queries into actionable intelligence
with clear confidence levels, source attribution, and identified patterns.

## RESEARCH CONTEXT
Query: {query}
Research Plan: {research_brief}
Methodology: {methodology}
{conversation_context}

# SYSTEMATIC RESEARCH PROTOCOL (Chain-of-Thought)

## Phase 1: Strategic Planning
**Think Step-by-Step:**
1. Decompose the research plan into executable tasks
2. Prioritize based on information value
3. Identify potential source categories
4. Anticipate likely challenges

## Phase 2: Source Discovery & Evaluation

### Source Credibility Matrix
**Tier 1 (Trust Score 0.9-1.0):**
- Peer-reviewed academic journals
- Government statistics/reports
- Standards organizations (ISO, IEEE, W3C)
- Original research papers

**Tier 2 (Trust Score 0.7-0.8):**
- Industry analyst reports (Gartner, Forrester)
- Established news organizations
- Technical documentation
- Conference proceedings

**Tier 3 (Trust Score 0.5-0.6):**
- Expert blogs with citations
- Company whitepapers
- Community documentation
- Preprints (arXiv, bioRxiv)

**Tier 4 (Trust Score <0.5):**
- Opinion pieces without data
- Outdated sources (>5 years unless historical)
- Unverified claims
- Marketing materials

## Phase 3: Information Extraction Protocol

For each source, extract:
1. **Core Finding**: Primary claim or data point
2. **Supporting Evidence**: Data, studies, or examples
3. **Confidence Level**: Based on source tier + evidence quality
4. **Relevance Score**: Direct (1.0) vs Tangential (0.3)
5. **Contradictions**: Conflicts with other sources
6. **Limitations**: What the source doesn't address

## Phase 4: Pattern Recognition (Tree of Thoughts)

Analyze findings across multiple dimensions:
```
Pattern Analysis
├── Convergence: Multiple sources agree
│   ├── Strong consensus (>80% agreement)
│   └── Weak consensus (50-80% agreement)
├── Divergence: Sources disagree
│   ├── Methodological differences
│   └── Temporal changes
└── Emergence: New trends appearing
    ├── Early signals
    └── Accelerating adoption
```

## Phase 5: Synthesis Framework (ReAct Pattern)

**Thought**: What patterns am I seeing?
**Action**: Synthesize related findings
**Observation**: What insights emerge?
**Thought**: Are there gaps or contradictions?
**Action**: Investigate discrepancies
**Observation**: Root causes identified

# RESEARCH EXECUTION EXAMPLES (Few-Shot Learning)

## Example 1: Technical Comparison Research
**Query**: "PostgreSQL vs MySQL for e-commerce at scale"
**Execution**:
1. Found 12 benchmark studies (Tier 1-2 sources)
2. Extracted performance metrics:
   - PostgreSQL: Better complex queries (confidence: 0.85)
   - MySQL: Better simple reads (confidence: 0.90)
3. Pattern identified: PostgreSQL preferred >10M products
4. Gap identified: Limited data on specific e-commerce workloads
5. Synthesis: PostgreSQL recommended with caveats

## Example 2: Emerging Technology Research
**Query**: "Quantum computing readiness for enterprise"
**Execution**:
1. Found 8 academic papers, 5 industry reports
2. Key finding: 5-10 years from broad enterprise use (confidence: 0.75)
3. Pattern: Optimism decreasing over time
4. Contradiction: Vendor claims vs academic assessments
5. Insight: Useful for specific optimization problems only

## Example 3: Best Practices Research
**Query**: "API rate limiting strategies for SaaS"
**Execution**:
1. Analyzed 15 implementations from major providers
2. Common patterns:
   - Token bucket (60% adoption, confidence: 0.95)
   - Sliding window (25% adoption, confidence: 0.90)
3. Emerging trend: Adaptive rate limiting with ML
4. Gap: Limited data on user experience impact

# OUTPUT STRUCTURE REQUIREMENTS

## 1. Executive Summary
- 3-5 bullet points of KEY findings
- Overall confidence in research completeness
- Critical gaps or limitations

## 2. Detailed Findings
For each finding:
```
Finding: [Clear statement]
Source: [Attribution with tier]
Confidence: [0.0-1.0 with justification]
Evidence: [Supporting data/quotes]
Relevance: [How it addresses the query]
```

## 3. Pattern Analysis
- Consensus areas (where sources agree)
- Controversy areas (where sources conflict)
- Emerging trends (new developments)
- Temporal patterns (changes over time)

## 4. Insights & Recommendations
- Actionable insights derived from patterns
- Recommendations with confidence levels
- Risk factors and caveats
- Areas requiring further investigation

## 5. Research Quality Assessment
- Coverage: What % of the query was addressed
- Confidence: Overall confidence in findings
- Limitations: What couldn't be determined
- Bias assessment: Potential biases identified

# SELF-VERIFICATION PROTOCOL

Before outputting, verify:
□ All aspects of research plan addressed?
□ Sources properly evaluated for credibility?
□ Patterns and contradictions identified?
□ Confidence levels justified by evidence?
□ Gaps and limitations acknowledged?
□ Findings traceable to sources?
□ Actionable insights provided?

# QUALITY ANTI-PATTERNS TO AVOID

✗ Accepting sources without credibility assessment
✗ Ignoring contradictory evidence
✗ Overgeneralizing from limited data
✗ Mixing facts with speculation without labels
✗ Providing findings without confidence levels
✗ Missing obvious follow-up questions

# EXECUTION INSTRUCTION
Systematically execute the research protocol.
Maintain rigorous standards for evidence quality.
Provide transparent confidence assessments.
Deliver actionable intelligence, not just information.
"""


class ResearchExecutorAgent(BaseResearchAgent[ResearchDependencies, ResearchResults]):
    """Agent responsible for executing research and gathering findings.

    This agent conducts systematic research, evaluates sources, extracts findings,
    and synthesizes insights according to the research brief.
    """

    def __init__(
        self,
        config: AgentConfiguration | None = None,
        dependencies: ResearchDependencies | None = None,
    ):
        """Initialize the research executor agent.

        Args:
            config: Optional agent configuration. If not provided, uses defaults.
            dependencies: Optional research dependencies.
        """
        if config is None:
            config = AgentConfiguration(
                agent_name="research_executor",
                agent_type="execution",
            )
        super().__init__(config=config, dependencies=dependencies)

        # Register dynamic instructions
        @self.agent.instructions
        async def add_execution_context(ctx: RunContext[ResearchDependencies]) -> str:
            """Inject research execution context as instructions."""
            query = ctx.deps.research_state.user_query
            metadata = ctx.deps.research_state.metadata
            conversation = metadata.conversation_messages if metadata else []

            # Extract research plan from transformed query metadata
            research_brief = ""
            methodology = ""
            if metadata and metadata.query.transformed_query:
                transformed_data = metadata.query.transformed_query
                research_plan = transformed_data.get("research_plan", {})
                research_brief = str(research_plan)
                # Extract methodology if available
                methodology = research_plan.get("methodology", "")

            # Include search queries if available
            if ctx.deps.search_queries:
                search_context = f"\n\nSearch Queries to Execute: {ctx.deps.search_queries.queries}"
                research_brief += search_context

            # Format conversation context
            conversation_context = self._format_conversation_context(conversation)

            # Use global template with variable substitution
            return RESEARCH_EXECUTOR_SYSTEM_PROMPT_TEMPLATE.format(
                query=query,
                research_brief=research_brief,
                methodology=methodology,
                conversation_context=conversation_context,
            )

        # Register research execution tools
        @self.agent.tool
        async def evaluate_source_credibility(
            ctx: RunContext[ResearchDependencies], source_info: dict[str, Any]
        ) -> float:
            """Evaluate the credibility of a source.

            Args:
                source_info: Dictionary with source information

            Returns:
                Credibility score between 0 and 1
            """
            score = 0.5  # Base score

            # Check for academic sources
            if any(
                domain in source_info.get("url", "").lower()
                for domain in [".edu", ".gov", "arxiv", "pubmed", "scholar"]
            ):
                score += 0.3

            # Check for peer review mention
            if "peer" in source_info.get("description", "").lower():
                score += 0.2

            # Check for author credentials
            if source_info.get("author"):
                score += 0.1

            # Check for publication date (prefer recent)
            if source_info.get("date"):
                try:
                    pub_date = datetime.fromisoformat(source_info["date"])
                    years_old = (datetime.now() - pub_date).days / 365
                    if years_old < 2:
                        score += 0.1
                    elif years_old < 5:
                        score += 0.05
                except Exception:
                    pass

            return min(1.0, score)

        @self.agent.tool
        async def categorize_findings(
            ctx: RunContext[ResearchDependencies], findings: list[str]
        ) -> dict[str, list[str]]:
            """Categorize research findings by topic.

            Args:
                findings: List of finding descriptions

            Returns:
                Dictionary of categorized findings
            """
            categories = {
                "technical": [],
                "business": [],
                "social": [],
                "environmental": [],
                "regulatory": [],
                "economic": [],
                "other": [],
            }

            category_keywords = {
                "technical": [
                    "technology",
                    "technical",
                    "software",
                    "hardware",
                    "algorithm",
                    "system",
                ],
                "business": ["business", "market", "company", "revenue", "profit", "strategy"],
                "social": ["social", "people", "community", "culture", "society", "human"],
                "environmental": ["environment", "climate", "sustainable", "green", "carbon"],
                "regulatory": ["regulation", "law", "policy", "compliance", "government"],
                "economic": ["economic", "economy", "financial", "cost", "investment", "gdp"],
            }

            for finding in findings:
                finding_lower = finding.lower()
                categorized = False

                for category, keywords in category_keywords.items():
                    if any(keyword in finding_lower for keyword in keywords):
                        categories[category].append(finding)
                        categorized = True
                        break

                if not categorized:
                    categories["other"].append(finding)

            # Remove empty categories
            return {k: v for k, v in categories.items() if v}

        @self.agent.tool
        async def identify_patterns(
            ctx: RunContext[ResearchDependencies], findings: list[str]
        ) -> list[str]:
            """Identify patterns across research findings.

            Args:
                findings: List of findings to analyze

            Returns:
                List of identified patterns
            """
            patterns = []

            # Look for repeated themes
            word_count = {}
            for finding in findings:
                words = finding.lower().split()
                for word in words:
                    if len(word) > 5:  # Focus on meaningful words
                        word_count[word] = word_count.get(word, 0) + 1

            # Identify frequently mentioned terms
            frequent_terms = [
                word for word, count in word_count.items() if count > len(findings) * 0.3
            ]
            if frequent_terms:
                patterns.append(f"Recurring themes: {', '.join(frequent_terms[:5])}")

            # Look for consensus
            if len(findings) > 3:
                # Simple sentiment check
                positive_words = ["improve", "increase", "benefit", "success", "effective"]
                negative_words = ["decrease", "risk", "challenge", "problem", "issue"]

                positive_count = sum(
                    1 for f in findings if any(w in f.lower() for w in positive_words)
                )
                negative_count = sum(
                    1 for f in findings if any(w in f.lower() for w in negative_words)
                )

                if positive_count > len(findings) * 0.6:
                    patterns.append("Generally positive outlook across findings")
                elif negative_count > len(findings) * 0.6:
                    patterns.append("Significant challenges or concerns identified")

            return patterns

    def _get_default_system_prompt(self) -> str:
        """Get the base system prompt for this agent."""
        return "You are a Research Executor focused on conducting systematic research."

    def _get_output_type(self) -> type[ResearchResults]:
        """Get the output type for this agent."""
        return ResearchResults


# Lazy initialization of module-level instance
_research_executor_agent_instance = None


def get_research_executor_agent() -> ResearchExecutorAgent:
    """Get or create the research executor agent instance."""
    global _research_executor_agent_instance
    if _research_executor_agent_instance is None:
        _research_executor_agent_instance = ResearchExecutorAgent()
        logfire.info("Initialized research_executor agent")
    return _research_executor_agent_instance


# For backward compatibility, create a property-like access
class _LazyAgent:
    """Lazy proxy for ResearchExecutorAgent that delays instantiation."""

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the actual agent instance."""
        return getattr(get_research_executor_agent(), name)


research_executor_agent = _LazyAgent()
