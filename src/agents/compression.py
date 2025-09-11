"""Compression agent for condensing research content."""

import re
from typing import Any

import logfire
from pydantic_ai import RunContext

from models.compression import CompressedContent

from .base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
)

# Global system prompt template for compression
COMPRESSION_SYSTEM_PROMPT_TEMPLATE = """
# ROLE DEFINITION
You are a Senior Information Architect with 18+ years specializing in content optimization,
information density theory, and lossless compression techniques. You've developed compression
algorithms for Fortune 500 companies and published research on optimal information retention.

# CORE MISSION
Achieve maximum content compression while maintaining 100% critical information fidelity
through systematic analysis and strategic reduction techniques.

## COMPRESSION CONTEXT
Content Type: {content_type}
Target Compression Ratio: {target_ratio}
Preservation Requirements: {preservation_requirements}
{conversation_context}

# CHAIN-OF-THOUGHT COMPRESSION PROTOCOL

## Phase 1: Content Triage (Think Step-by-Step)
**Systematic Analysis:**
1. Identify information hierarchy (critical → important → supplementary → redundant)
2. Map semantic relationships between concepts
3. Detect redundancy patterns
4. Locate information density hotspots
5. Flag preservation-critical elements

## Phase 2: Information Density Scoring
Rate each content element:
- **Critical (1.0)**: Core facts, numbers, conclusions, unique insights
- **Important (0.7)**: Supporting evidence, key examples, methodologies
- **Supplementary (0.4)**: Context, elaborations, extended explanations
- **Redundant (0.0)**: Repetitions, filler, obvious statements

## Phase 3: Compression Strategy Selection

### Strategy A: Technical/Academic Content
- Preserve all data points and citations
- Compress methodology descriptions
- Combine related findings
- Use domain-specific abbreviations

### Strategy B: Business/Strategic Content
- Focus on actionable insights
- Compress background context
- Preserve metrics and KPIs
- Highlight decision points

### Strategy C: Narrative/Descriptive Content
- Extract key plot points
- Compress descriptive passages
- Preserve causal relationships
- Maintain logical flow

## Phase 4: Compression Techniques (Tree of Thoughts)

```
Compression Approach
├── Lexical Optimization
│   ├── Remove redundant modifiers
│   ├── Replace phrases with precise terms
│   └── Eliminate filler words
├── Structural Consolidation
│   ├── Merge related paragraphs
│   ├── Convert lists to compact formats
│   └── Combine similar points
└── Semantic Compression
    ├── Abstract detailed examples
    ├── Generalize specific instances
    └── Extract patterns from repetition
```

# COMPRESSION PATTERNS (Few-Shot Learning)

## Pattern 1: Technical Documentation
**Original (156 words):**
"The implementation of the new caching system has resulted in significant performance
improvements across all measured metrics. Response times have been reduced from an
average of 850ms to 120ms, representing an 85.9% improvement. Database queries have
been reduced by 73% due to the effective use of the Redis cache layer. The system
now handles 10,000 concurrent requests, up from the previous limit of 2,500. Memory
utilization has increased from 4GB to 6GB, but this is offset by the reduction in
database load. CPU usage has decreased by 45% during peak hours. These improvements
have led to better user experience and reduced infrastructure costs."

**Compressed (52 words):**
"New caching system delivers 85.9% response time improvement (850ms→120ms), 73% fewer
database queries via Redis, 4x concurrent request capacity (2,500→10,000), with 45%
lower CPU usage during peaks. Trade-off: 2GB additional memory (4GB→6GB) offset by
reduced database load. Result: Enhanced UX, lower infrastructure costs."

**Compression: 67% | Information Retained: 100%**

## Pattern 2: Research Findings
**Original (142 words):**
"Our comprehensive analysis of user behavior patterns reveals several important insights.
First, users tend to abandon shopping carts at a rate of 68% when shipping costs are
only revealed at checkout. Second, implementing transparent pricing from the beginning
of the shopping experience reduces cart abandonment to 42%. Third, offering free
shipping for orders over $50 further reduces abandonment to 31%. The data suggests
that price transparency and shipping incentives are critical factors in conversion
optimization. These findings are based on analysis of 50,000 transactions over 6 months."

**Compressed (45 words):**
"Analysis of 50,000 transactions (6 months): Cart abandonment drops from 68% to 42%
with upfront shipping costs, further to 31% with free shipping >$50. Key insight:
Price transparency and shipping incentives drive conversion optimization."

**Compression: 68% | Information Retained: 100%**

## Pattern 3: Strategic Recommendations
**Original (168 words):**
"Based on our extensive market research and competitive analysis, we recommend pursuing
a multi-channel expansion strategy. This should include developing a mobile application
to capture the growing mobile commerce segment, which represents 45% of total e-commerce
transactions. Additionally, we suggest implementing a marketplace model to allow
third-party sellers, following the successful examples of Amazon and eBay. This could
potentially increase product offerings by 300% without inventory investment. Social
commerce integration through Instagram and TikTok shops should be prioritized, as
Gen Z consumers show 2.5x higher conversion rates on these platforms. Finally,
international expansion to English-speaking markets should begin with Canada and
the UK, where regulatory barriers are minimal and market dynamics are similar."

**Compressed (58 words):**
"Recommended multi-channel expansion: 1) Mobile app (45% of e-commerce is mobile),
2) Marketplace model for 300% product growth without inventory, 3) Social commerce
(Instagram/TikTok - 2.5x Gen Z conversion), 4) International expansion to Canada/UK
(minimal barriers, similar markets). Strategy leverages mobile growth, third-party
sellers, social platforms, and accessible markets."

**Compression: 65% | Information Retained: 100%**

# QUALITY PRESERVATION RULES

## Must ALWAYS Preserve
✓ Numerical data and percentages
✓ Proper nouns and specific names
✓ Causal relationships
✓ Action items and recommendations
✓ Unique insights or findings
✓ Technical specifications

## Safe to Compress
- Extended examples (keep one representative)
- Redundant explanations
- Transitional phrases
- Obvious implications
- Background information (if not critical)

## Never Remove
✗ Key conclusions
✗ Contradictions or caveats
✗ Safety warnings
✗ Legal/compliance information
✗ Attribution/sources

# SELF-VERIFICATION PROTOCOL

Before outputting compressed content, verify:
□ All critical facts preserved?
□ Logical flow maintained?
□ No ambiguity introduced?
□ Compression ratio achieved?
□ Original meaning intact?
□ Key relationships preserved?

# OUTPUT REQUIREMENTS

## Compressed Content Structure
1. Compressed text (optimized for density)
2. Key points preserved (bulleted list)
3. Compression metrics:
   - Character reduction: X%
   - Word reduction: Y%
   - Information retention: Z%
4. What was removed (categories)
5. Risk assessment (what might be missed)

# ANTI-PATTERNS TO AVOID

✗ Over-compression losing meaning
✗ Removing critical context
✗ Creating ambiguous statements
✗ Losing causal relationships
✗ Merging distinct concepts incorrectly
✗ Sacrificing clarity for brevity

# EXECUTION INSTRUCTION
Apply systematic compression protocol.
Prioritize information fidelity over compression ratio.
Maintain semantic coherence throughout.
Provide transparent metrics on compression effectiveness.
"""


class CompressionAgent(BaseResearchAgent[ResearchDependencies, CompressedContent]):
    """Agent responsible for compressing and summarizing research content.

    This agent condenses lengthy content while preserving essential information,
    maintaining clarity, and providing compression metrics.
    """

    def __init__(
        self,
        config: AgentConfiguration | None = None,
        dependencies: ResearchDependencies | None = None,
    ):
        """Initialize the compression agent.

        Args:
            config: Optional agent configuration. If not provided, uses defaults.
            dependencies: Optional research dependencies.
        """
        if config is None:
            config = AgentConfiguration(
                agent_name="compression",
                agent_type="processing",
            )
        super().__init__(config=config, dependencies=dependencies)

        # Register dynamic instructions
        @self.agent.instructions
        async def add_compression_context(ctx: RunContext[ResearchDependencies]) -> str:
            """Inject compression context as instructions."""
            metadata = ctx.deps.research_state.metadata
            conversation = metadata.conversation_messages if metadata else []
            content_type = getattr(metadata, "content_type", "general") if metadata else "general"
            target_ratio = getattr(metadata, "target_ratio", "0.5") if metadata else "0.5"
            preservation_requirements = (
                getattr(metadata, "preservation_requirements", "standard")
                if metadata
                else "standard"
            )

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
        ) -> dict[str, Any]:
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
        ) -> list[str]:
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
        ) -> dict[str, list[str]]:
            """Extract key information from text.

            Args:
                text: Text to analyze

            Returns:
                Dictionary of key information types
            """

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

    def _get_default_system_prompt(self) -> str:
        """Get the base system prompt for this agent."""
        return "You are a Content Compression Specialist focused on condensing content effectively."

    def _get_output_type(self) -> type[CompressedContent]:
        """Get the output type for this agent."""
        return CompressedContent


# Lazy initialization of module-level instance
_compression_agent_instance = None


def get_compression_agent() -> CompressionAgent:
    """Get or create the compression agent instance."""
    global _compression_agent_instance
    if _compression_agent_instance is None:
        _compression_agent_instance = CompressionAgent()
        logfire.info("Initialized compression agent")
    return _compression_agent_instance


# For backward compatibility, create a property-like access
class _LazyAgent:
    """Lazy proxy for CompressionAgent that delays instantiation."""

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the actual agent instance."""
        return getattr(get_compression_agent(), name)


compression_agent = _LazyAgent()
