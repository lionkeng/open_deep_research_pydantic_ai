"""Query transformation agent for optimizing research queries."""

from typing import Any

from pydantic_ai import RunContext

from ..models.query_transformation import TransformedQuery
from .base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
)

# Global system prompt template for query transformation
QUERY_TRANSFORMATION_SYSTEM_PROMPT_TEMPLATE = """
## QUERY TRANSFORMATION SPECIALIST:

You are an expert at analyzing and optimizing research queries to make them more effective
for comprehensive research.

### YOUR ROLE:
1. Analyze the user's original query to understand intent and requirements
2. Transform vague or broad queries into specific, actionable research questions
3. Extract key search terms and concepts
4. Define clear research scope and boundaries
5. Identify the expected output type and format
6. Optimize the query for better research results

### TRANSFORMATION PRINCIPLES:
- Break down complex queries into manageable components
- Identify implicit assumptions and make them explicit
- Suggest related areas that might be relevant
- Clarify ambiguous terms or concepts
- Ensure the query is specific enough to yield focused results
- Maintain the original intent while improving clarity

### ANALYSIS FRAMEWORK:
1. **Intent Analysis**: What is the user really trying to learn or accomplish?
2. **Scope Definition**: What are the boundaries of this research?
3. **Key Concepts**: What are the core terms and concepts to investigate?
4. **Output Requirements**: What type of deliverable would best serve the user?
5. **Search Strategy**: What keywords and phrases will yield the best results?

### EXAMPLES OF GOOD TRANSFORMATIONS:
- Vague: "Tell me about AI" → Specific: "What are the current applications, benefits, and
  limitations of artificial intelligence in healthcare diagnostics?"
- Broad: "Research climate change" → Focused: "What are the primary causes and measurable
  impacts of climate change on coastal ecosystems in the past decade?"
- Ambiguous: "How does it work?" → Clear: "Explain the technical architecture and data flow
  of transformer neural networks"

## CURRENT QUERY CONTEXT:
Original Query: {original_query}
{conversation_context}

## TRANSFORMATION REQUIREMENTS:
- Maintain the user's original intent
- Make the query more specific and actionable
- Extract 3-10 relevant search keywords
- Define a clear research scope
- Identify the expected output type
- Provide confidence in your transformation
"""


class QueryTransformationAgent(BaseResearchAgent[ResearchDependencies, TransformedQuery]):
    """Agent responsible for transforming and optimizing user queries for research.

    This agent takes vague or broad queries and transforms them into specific,
    actionable research questions with clear scope and search strategies.
    """

    def __init__(self):
        """Initialize the query transformation agent."""
        config = AgentConfiguration(
            agent_name="query_transformation",
            agent_type="transformation",
        )
        super().__init__(config=config)

        # Register dynamic instructions
        @self.agent.instructions
        async def add_transformation_context(ctx: RunContext[ResearchDependencies]) -> str:
            """Inject query transformation context as instructions."""
            query = ctx.deps.research_state.user_query
            metadata = ctx.deps.research_state.metadata
            conversation = metadata.conversation_messages if metadata else []

            # Format conversation context
            conversation_context = self._format_conversation_context(conversation)

            # Use global template with variable substitution
            return QUERY_TRANSFORMATION_SYSTEM_PROMPT_TEMPLATE.format(
                original_query=query, conversation_context=conversation_context
            )

        # Register transformation tools
        @self.agent.tool
        async def analyze_query_complexity(
            ctx: RunContext[ResearchDependencies], query: str
        ) -> dict[str, Any]:
            """Analyze the complexity of a query.

            Args:
                query: The query to analyze

            Returns:
                Dictionary with complexity metrics
            """
            words = query.split()
            return {
                "word_count": len(words),
                "estimated_complexity": "high"
                if len(words) > 50
                else "medium"
                if len(words) > 20
                else "low",
                "has_multiple_questions": "?" in query and query.count("?") > 1,
                "has_technical_terms": any(len(word) > 10 for word in words),
            }

        @self.agent.tool
        async def extract_key_concepts(
            ctx: RunContext[ResearchDependencies], text: str
        ) -> list[str]:
            """Extract key concepts from text.

            Args:
                text: The text to analyze

            Returns:
                List of key concepts
            """
            import re

            words = text.split()
            concepts = []

            # Look for capitalized words (potential proper nouns)
            for word in words:
                if word and word[0].isupper() and word.lower() not in ["the", "a", "an"]:
                    concepts.append(word)

            # Look for quoted terms
            quoted = re.findall(r'"([^"]*)"', text)
            concepts.extend(quoted)

            return list(set(concepts))[:10]

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
        return "You are a Query Transformation Specialist focused on optimizing research queries."

    def _get_output_type(self) -> type[TransformedQuery]:
        """Get the output type for this agent."""
        return TransformedQuery


# Create module-level instance
query_transformation_agent = QueryTransformationAgent()
