"""Compression agent for condensing research content."""

from typing import Any

from pydantic_ai import RunContext

from ..models.compression import CompressedContent
from .base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
)

# Global system prompt template for compression
COMPRESSION_SYSTEM_PROMPT_TEMPLATE = """
## CONTENT COMPRESSION SPECIALIST:

You are an expert at condensing lengthy content while preserving essential information
and maintaining clarity.

### YOUR ROLE:
1. Analyze content structure and identify key information
2. Remove redundancies and verbose expressions
3. Preserve critical facts, insights, and conclusions
4. Maintain logical flow and readability
5. Apply appropriate compression strategies
6. Ensure no critical information is lost
7. Measure and report compression effectiveness

### COMPRESSION STRATEGIES:
- Eliminate redundant phrases and repetitions
- Convert verbose expressions to concise alternatives
- Combine related points into unified statements
- Use bullet points for lists instead of paragraphs
- Remove filler words and unnecessary qualifiers
- Preserve technical terms and specific data points
- Maintain context and relationships between ideas

### COMPRESSION FRAMEWORK:
1. **Content Analysis**: Identify structure and key elements
2. **Redundancy Detection**: Find repetitive content
3. **Fact Extraction**: Identify critical information
4. **Synthesis**: Combine related information
5. **Optimization**: Apply compression techniques
6. **Quality Check**: Ensure information retention
7. **Metrics**: Measure compression effectiveness

### PRESERVATION PRIORITIES:
- **Must Preserve**: Facts, numbers, conclusions, recommendations
- **Should Preserve**: Key examples, important context, methodologies
- **Can Compress**: Verbose explanations, repetitions, filler content
- **Can Remove**: Redundancies, unnecessary qualifiers, obvious statements

## CURRENT COMPRESSION CONTEXT:
Content Type: {content_type}
Target Ratio: {target_ratio}
Preservation Requirements: {preservation_requirements}
{conversation_context}

## COMPRESSION REQUIREMENTS:
- Achieve effective compression while maintaining quality
- Preserve all critical information
- Maintain readability and coherence
- Extract key facts and insights
- Report compression metrics
- Apply appropriate strategy for content type
"""


class CompressionAgent(BaseResearchAgent[ResearchDependencies, CompressedContent]):
    """Agent responsible for compressing and summarizing research content.

    This agent condenses lengthy content while preserving essential information,
    maintaining clarity, and providing compression metrics.
    """

    def __init__(self):
        """Initialize the compression agent."""
        config = AgentConfiguration(
            agent_name="compression",
            agent_type="processing",
        )
        super().__init__(config=config)

        # Register dynamic instructions
        @self.agent.instructions
        async def add_compression_context(ctx: RunContext[ResearchDependencies]) -> str:  # pyright: ignore
            """Inject compression context as instructions."""
            metadata = ctx.deps.research_state.metadata or {}
            conversation = metadata.get("conversation_messages", [])
            content_type = metadata.get("content_type", "general")
            target_ratio = metadata.get("target_ratio", "0.5")
            preservation_requirements = metadata.get("preservation_requirements", "standard")

            # Format conversation context
            conversation_context = self._format_conversation_context(conversation)

            # Use global template with variable substitution
            return COMPRESSION_SYSTEM_PROMPT_TEMPLATE.format(
                content_type=content_type,
                target_ratio=target_ratio,
                preservation_requirements=preservation_requirements,
                conversation_context=conversation_context,
            )

        # Register compression tools
        @self.agent.tool
        async def calculate_compression_metrics(
            ctx: RunContext[ResearchDependencies], original: str, compressed: str
        ) -> dict[str, Any]:  # pyright: ignore
            """Calculate compression metrics.

            Args:
                original: Original text
                compressed: Compressed text

            Returns:
                Dictionary of compression metrics
            """
            original_len = len(original)
            compressed_len = len(compressed)
            original_words = len(original.split())
            compressed_words = len(compressed.split())

            return {
                "character_ratio": 1 - (compressed_len / original_len) if original_len > 0 else 0,
                "word_ratio": 1 - (compressed_words / original_words) if original_words > 0 else 0,
                "original_characters": original_len,
                "compressed_characters": compressed_len,
                "original_words": original_words,
                "compressed_words": compressed_words,
                "percentage_retained": (compressed_len / original_len * 100)
                if original_len > 0
                else 100,
            }

        @self.agent.tool
        async def identify_redundancies(
            ctx: RunContext[ResearchDependencies], text: str
        ) -> list[str]:  # pyright: ignore
            """Identify redundancies in text.

            Args:
                text: Text to analyze

            Returns:
                List of identified redundancies
            """
            redundancies = []
            sentences = text.split(". ")

            # Check for repeated phrases
            phrase_count = {}
            for sentence in sentences:
                # Extract 3-5 word phrases
                words = sentence.split()
                for i in range(len(words) - 2):
                    phrase = " ".join(words[i : i + 3])
                    if len(phrase) > 10:
                        phrase_count[phrase] = phrase_count.get(phrase, 0) + 1

            # Identify repeated phrases
            for phrase, count in phrase_count.items():
                if count > 1:
                    redundancies.append(f"Repeated phrase ({count}x): '{phrase}'")

            # Check for filler phrases
            filler_phrases = [
                "it is important to note that",
                "it should be mentioned that",
                "as a matter of fact",
                "in order to",
                "due to the fact that",
                "in the event that",
                "at this point in time",
            ]

            for filler in filler_phrases:
                if filler in text.lower():
                    redundancies.append(f"Filler phrase: '{filler}'")

            return redundancies[:10]  # Limit to top 10

        @self.agent.tool
        async def extract_key_information(
            ctx: RunContext[ResearchDependencies], text: str
        ) -> dict[str, list[str]]:  # pyright: ignore
            """Extract key information from text.

            Args:
                text: Text to analyze

            Returns:
                Dictionary of key information types
            """
            import re

            key_info = {
                "numbers": [],
                "dates": [],
                "names": [],
                "conclusions": [],
                "recommendations": [],
            }

            # Extract numbers and percentages
            numbers = re.findall(r"\b\d+\.?\d*%?\b", text)
            key_info["numbers"] = list(set(numbers))[:10]

            # Extract potential dates
            date_patterns = [
                r"\b\d{4}\b",  # Years
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            ]
            for pattern in date_patterns:
                dates = re.findall(pattern, text, re.IGNORECASE)
                key_info["dates"].extend(dates)
            key_info["dates"] = list(set(key_info["dates"]))[:5]

            # Extract capitalized words (potential names/entities)
            names = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", text)
            # Filter out common words
            common_words = {"The", "This", "That", "These", "Those", "However", "Therefore"}
            key_info["names"] = [n for n in set(names) if n not in common_words][:10]

            # Look for conclusion indicators
            conclusion_indicators = ["conclude", "summary", "therefore", "thus", "finally"]
            sentences = text.split(". ")
            for sent in sentences:
                if any(indicator in sent.lower() for indicator in conclusion_indicators):
                    key_info["conclusions"].append(sent.strip())

            # Look for recommendation indicators
            recommendation_indicators = ["recommend", "suggest", "should", "must", "advise"]
            for sent in sentences:
                if any(indicator in sent.lower() for indicator in recommendation_indicators):
                    key_info["recommendations"].append(sent.strip())

            return key_info

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
        return "You are a Content Compression Specialist focused on condensing content effectively."

    def _get_output_type(self) -> type[CompressedContent]:
        """Get the output type for this agent."""
        return CompressedContent


# Create module-level instance
compression_agent = CompressionAgent()
