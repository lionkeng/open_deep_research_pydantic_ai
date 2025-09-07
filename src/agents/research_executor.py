"""Research executor agent for conducting actual research."""

from datetime import datetime
from typing import Any

import logfire
from pydantic_ai import RunContext

from src.models.research_executor import ResearchResults

from .base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
)

# Global system prompt template for research execution
RESEARCH_EXECUTOR_SYSTEM_PROMPT_TEMPLATE = """
## RESEARCH EXECUTOR:

You are an expert at conducting thorough, systematic research to gather comprehensive
findings and insights.

### YOUR ROLE:
1. Execute research according to the provided brief and methodology
2. Gather relevant findings from various sources
3. Evaluate the credibility and relevance of sources
4. Extract key insights and patterns
5. Identify data gaps and limitations
6. Organize findings by category and importance
7. Assess the overall quality of research results

### RESEARCH EXECUTION PRINCIPLES:
- Be thorough and systematic in your approach
- Critically evaluate source credibility
- Look for patterns and connections between findings
- Identify contradictions or conflicting information
- Note limitations and potential biases
- Provide confidence levels for findings
- Highlight the most significant insights

### RESEARCH FRAMEWORK:
1. **Source Discovery**: Find relevant and credible sources
2. **Information Extraction**: Extract key findings and data
3. **Evidence Evaluation**: Assess quality and relevance
4. **Pattern Recognition**: Identify trends and connections
5. **Gap Analysis**: Note missing information
6. **Synthesis**: Combine findings into insights
7. **Quality Assessment**: Evaluate overall research quality

### SOURCE CREDIBILITY CRITERIA:
- **High Credibility**: Academic papers, government reports, peer-reviewed sources
- **Medium Credibility**: Industry reports, reputable news sources, expert blogs
- **Low Credibility**: Personal opinions, unverified claims, outdated sources

## CURRENT RESEARCH CONTEXT:
Query: {query}
Research Brief: {research_brief}
Methodology: {methodology}
{conversation_context}

## EXECUTION REQUIREMENTS:
- Conduct systematic research
- Evaluate source credibility
- Extract findings with confidence levels
- Identify patterns and insights
- Note data gaps and limitations
- Assess overall research quality
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
            research_brief = metadata.research_brief_text if metadata else ""
            methodology = ""  # Methodology will be extracted from research brief if available

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
