"""Clarification agent for identifying when user queries need additional information."""

import re
from datetime import datetime
from typing import Any

import logfire
from pydantic import BaseModel, Field

from open_deep_research_with_pydantic_ai.agents.base import (
    BaseResearchAgent,
    ResearchDependencies,
    coordinator,
)
from open_deep_research_with_pydantic_ai.core.events import (
    emit_stage_completed,
)
from open_deep_research_with_pydantic_ai.models.research import ResearchStage, TransformedQuery


class ClarifyWithUser(BaseModel):
    """Model for user clarification requests - matches Langgraph's approach."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question."
    )
    question: str = Field(
        default="", description="A question to ask the user to clarify the report scope"
    )
    verification: str = Field(
        default="", description="Verification message that we will start research"
    )


class ClarificationAgent(BaseResearchAgent[ResearchDependencies, ClarifyWithUser]):
    """Agent responsible for determining if clarification is needed from the user.

    Enhanced Phase 1 implementation with:
    - Quantitative breadth assessment
    - Context dimension detection
    - Explicit conditional logic
    - Comprehensive scope analysis
    """

    # Broad query indicators that suggest clarification is needed
    BROAD_INDICATORS = [
        r"^what is\b",
        r"^how does\b",
        r"^explain\b",
        r"^tell me about\b",
        r"^overview of\b",
        r"^research\b",
        r"^analyze\b",
        r"^compare\b",
        r"^study\b",
    ]

    # Context dimension patterns
    CONTEXT_FLAGS = {
        "audience_level": ["beginner", "expert", "academic", "business", "professional", "student"],
        "purpose": ["for", "because", "need", "purpose", "use", "goal", "objective"],
        "scope": ["overview", "detailed", "comprehensive", "brief", "summary", "in-depth"],
        "specificity": ["which", "what kind", "type of", "specific", "particular", "exact"],
    }

    def __init__(self):
        """Initialize the clarification agent."""
        super().__init__(
            name="clarification_agent",
            output_type=ClarifyWithUser,
        )

    def _get_default_system_prompt(self) -> str:
        """Get the enhanced system prompt with explicit conditional logic."""
        return """You are a research clarification specialist. Today's date is {date}.

Your role is to assess whether you need to ask a clarifying question, or if the user
has already provided enough information for comprehensive research.

IMPORTANT: If you can see in the messages history that you have already asked a
clarifying question, you almost always do not need to ask another one. Only ask
another question if ABSOLUTELY NECESSARY.

## CRITICAL ASSESSMENT CRITERIA:
Ask for clarification if ANY of these essential dimensions are missing:

1. **Audience Level & Purpose**:
   - Who is this research for? (academic, business, personal)
   - What background knowledge level? (beginner, expert, professional)
   - What's the intended use? (presentation, analysis, decision-making)

2. **Scope & Focus Areas**:
   - Is the topic too broad without specific focus?
   - Are there multiple aspects that need prioritization?
   - What depth of coverage is needed?

3. **Source & Quality Requirements**:
   - Academic papers vs. general sources?
   - Specific time periods or geographic regions?
   - Industry vs. theoretical focus?

4. **Deliverable Specifications**:
   - What format of information is most useful?
   - Are there specific questions to be answered?
   - Any particular frameworks or methodologies to apply?

## EXAMPLES OF QUERIES REQUIRING CLARIFICATION:
- "What is [broad topic]?" → Ask about audience, purpose, specific focus
- "Compare [general category]" → Ask which specific items, criteria, timeframe
- "Research [technology]" → Ask about technical depth, applications, audience
- "Analyze [industry]" → Ask about specific aspects, timeframe, geographic scope

## INFORMATION GATHERING GUIDELINES:
- Be concise while gathering all necessary information
- Use bullet points or numbered lists for clarity in questions
- Don't ask for unnecessary information already provided
- Focus on gathering information needed for comprehensive research

## OUTPUT REQUIREMENTS (Explicit Conditional Instructions):

If you NEED to ask a clarifying question:
- need_clarification: true
- question: Your detailed clarifying question using bullet points for multiple aspects
- verification: "" (empty string - do not provide verification when asking questions)

If you DO NOT need clarification:
- need_clarification: false
- question: "" (empty string - do not provide a question when not needed)
- verification: Acknowledgment message that:
  * Confirms you have sufficient information
  * Briefly summarizes key aspects understood
  * States research will begin
  * Keeps message concise and professional

IMPORTANT: Only populate the field relevant to your decision. Leave the other field as an
empty string.
"""

    def _assess_query_breadth(
        self, query: str, conversation: list[str]
    ) -> tuple[float, dict[str, Any]]:
        """Assess if query is too broad or missing essential context.

        Args:
            query: The user's research query
            conversation: Previous conversation messages

        Returns:
            Tuple of (breadth_score, metadata) where breadth_score is 0.0-1.0
        """
        query_lower = query.lower()

        # Check for broad indicators
        broad_indicators_found: list[str] = []
        for pattern in self.BROAD_INDICATORS:
            if re.search(pattern, query_lower):
                broad_indicators_found.append(pattern)

        # Check for missing context flags
        missing_context_flags: dict[str, bool] = {}
        conversation_text = " ".join(conversation).lower() if conversation else ""

        for context_type, keywords in self.CONTEXT_FLAGS.items():
            has_context = any(
                keyword in conversation_text or keyword in query_lower for keyword in keywords
            )
            missing_context_flags[context_type] = not has_context

        # Calculate breadth score
        broad_score = len(broad_indicators_found) * 0.2
        missing_context_count = sum(missing_context_flags.values())
        context_penalty = missing_context_count * 0.15

        # Length factor - very short queries are often broad
        length_factor = max(0, (8 - len(query.split())) * 0.05)

        breadth_score = min(broad_score + context_penalty + length_factor, 1.0)

        metadata: dict[str, Any] = {
            "broad_indicators_found": broad_indicators_found,
            "missing_context_flags": missing_context_flags,
            "word_count": len(query.split()),
            "has_specific_terms": self._has_specific_terms(query),
            "has_constraints": self._has_constraints(query),
        }

        return breadth_score, metadata

    def _has_specific_terms(self, query: str) -> bool:
        """Check if query contains specific technical terms."""
        specific_patterns = [
            r"\d+\.\d+",  # Version numbers
            r"\b(api|sdk|framework|library|package)\b",  # Technical terms
            r"\b(python|javascript|java|c\+\+|react|angular|vue)\b",  # Languages/frameworks
            r"\b(aws|azure|gcp|docker|kubernetes)\b",  # Platforms
            r"\b(mysql|postgresql|mongodb|redis)\b",  # Databases
        ]

        query_lower = query.lower()
        return any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in specific_patterns)

    def _has_constraints(self, query: str) -> bool:
        """Check if query has explicit constraints or qualifiers."""
        constraint_patterns = [
            r"\b(for|using|with|in|specifically|only|just|exactly)\b",
            r"\b(comparing|versus|vs|between|difference)\b",
            r"\b(step.by.step|tutorial|guide|implementation)\b",
        ]

        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in constraint_patterns)

    def _register_tools(self) -> None:
        """Register clarification-specific tools.

        For this agent, we don't need complex tools - the LLM will analyze
        the conversation history directly to determine if clarification is needed.
        """
        # No tools needed for simple clarification assessment
        pass

    async def assess_query(
        self,
        query: str,
        deps: ResearchDependencies,
    ) -> ClarifyWithUser:
        """Assess if a query needs clarification with enhanced breadth detection.

        Args:
            query: The user's research query
            deps: Research dependencies with state and context

        Returns:
            ClarifyWithUser model with assessment result
        """
        try:
            # Get conversation history from research state metadata
            metadata = deps.research_state.metadata or {}
            conversation_history = metadata.get("conversation_messages", [])

            # Perform breadth assessment
            breadth_score, breadth_metadata = self._assess_query_breadth(
                query, conversation_history
            )

            # Build context from conversation history
            if conversation_history:
                messages_context = "\n".join(
                    [
                        f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}"
                        for i, msg in enumerate(conversation_history)
                    ]
                )
                messages_context = f"{messages_context}\nUser: {query}"
            else:
                messages_context = f"User: {query}"

            # Format the prompt with current date and messages
            current_date = datetime.now().strftime("%B %d, %Y")

            prompt = f"""These are the messages that have been exchanged so far from the user
asking for the report:
<Messages>
{messages_context}
</Messages>

BREADTH ANALYSIS:
- Breadth Score: {breadth_score:.2f} (0=specific, 1=broad)
- Broad Indicators Found: {
    ', '.join(breadth_metadata.get('broad_indicators_found', []))
}
- Missing Context Flags: {
    [k for k, v in breadth_metadata.get('missing_context_flags', {}).items() if v]
}
- Query Word Count: {breadth_metadata['word_count']}
- Has Specific Terms: {breadth_metadata['has_specific_terms']}
- Has Constraints: {breadth_metadata['has_constraints']}

Assess whether you need to ask a clarifying question, or if the user has already
provided enough information for you to start research.

Remember: Only ask a clarifying question if absolutely necessary. If you have enough
information to understand what the user wants to research, proceed without asking."""

            # Replace date placeholder in system prompt
            system_prompt = self._get_default_system_prompt().replace("{date}", current_date)

            # Create agent with formatted system prompt
            from pydantic_ai import Agent

            temp_agent = Agent(
                model=self.model,
                deps_type=ResearchDependencies,
                output_type=ClarifyWithUser,
                system_prompt=system_prompt,
            )

            # Run the agent with breadth context
            result = await temp_agent.run(
                prompt,
                deps=deps,
            )

            # Store breadth assessment in metadata for monitoring
            metadata["breadth_assessment"] = {"score": breadth_score, "metadata": breadth_metadata}

            # Update metadata with clarification status
            if result.data.need_clarification:  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
                metadata["clarification_count"] = metadata.get("clarification_count", 0) + 1
                metadata["last_clarification_question"] = result.data.question  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
            else:
                metadata["clarification_complete"] = True
                metadata["verification_message"] = result.data.verification  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]

            deps.research_state.metadata = metadata

            # Log the assessment for monitoring
            logfire.info(
                f"Query breadth assessment - Score: {breadth_score:.2f}, "
                f"Clarification needed: {result.data.need_clarification}",  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
                request_id=deps.research_state.request_id,
                breadth_score=breadth_score,
                breadth_metadata=breadth_metadata,
            )

            # Emit stage completed event
            await emit_stage_completed(
                request_id=deps.research_state.request_id,
                stage=ResearchStage.CLARIFICATION,
                success=not result.data.need_clarification,  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
                result=result.data,  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
            )

            return result.data  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]

        except Exception as e:
            logfire.error(
                f"Error in clarification assessment: {str(e)}",
                request_id=deps.research_state.request_id,
                exc_info=True,
            )

            # Return conservative fallback result
            return ClarifyWithUser(
                need_clarification=True,
                question="Could you provide more specific details about your research needs?",
                verification="",
            )

    async def refine_query(
        self, original_query: str, user_responses: dict[str, str], deps: ResearchDependencies
    ) -> str:
        """Refine a query based on user responses to clarification questions.

        Args:
            original_query: The original research query
            user_responses: User's responses to clarification questions
            deps: Research dependencies

        Returns:
            Refined query incorporating the clarifications
        """
        try:
            # Format user responses
            responses_text = "\n".join(
                [f"Q: {question}\nA: {answer}" for question, answer in user_responses.items()]
            )

            prompt = f"""Original query: {original_query}

User provided the following clarifications:
{responses_text}

Create a refined, more specific research query that:
1. Incorporates all the user's clarifications
2. Maintains the original intent
3. Is clear and unambiguous
4. Includes relevant constraints and context
5. Specifies the scope and depth needed

The refined query should be a single, well-structured question or request."""

            # Use a simple string output type for refinement
            from pydantic_ai import Agent

            refinement_agent = Agent(
                model=self.model,
                deps_type=ResearchDependencies,
                output_type=str,
                system_prompt=(
                    """You are a query refinement specialist. Create clear, specific research
                    queries
                    based on user input.

Your refined queries should:
- Be specific and actionable
- Include all relevant context from user responses
- Maintain professional, clear language
- Be structured as complete questions or requests
- Include constraints, scope, and purpose when provided"""
                ),
            )

            result = await refinement_agent.run(prompt, deps=deps)

            refined_query = result.data
            logfire.info(
                "Query refined successfully",
                original_query=original_query,
                refined_query=refined_query,
                request_id=deps.research_state.request_id,
            )

            return refined_query

        except Exception as e:
            logfire.error(
                f"Error refining query: {str(e)}",
                original_query=original_query,
                request_id=deps.research_state.request_id,
                exc_info=True,
            )
            return original_query

    async def should_ask_another_question(
        self, deps: ResearchDependencies, max_questions: int = 2
    ) -> bool:
        """Determine if another clarifying question should be asked.

        Args:
            deps: Research dependencies with state
            max_questions: Maximum number of questions to ask

        Returns:
            True if another question should be asked
        """
        metadata = deps.research_state.metadata or {}

        # Check if we've reached the question limit
        clarification_count = metadata.get("clarification_count", 0)
        if clarification_count >= max_questions:
            return False

        # Check if clarification is already complete
        if metadata.get("clarification_complete", False):
            return False

        # Check if we already have a research brief with high confidence
        brief_confidence = metadata.get("research_brief_confidence", 0.0)
        if brief_confidence >= 0.7:  # Configurable threshold
            return False

        return True

    async def process_clarification_responses_with_transformation(
        self,
        original_query: str,
        clarification_responses: dict[str, str],
        deps: ResearchDependencies,
    ) -> "TransformedQuery":
        """Process clarification responses and transform the query.

        This method integrates with the QueryTransformationAgent to convert
        the original query and clarification responses into a specific research question.

        Args:
            original_query: The original user query
            clarification_responses: Dict mapping questions to user responses
            deps: Research dependencies

        Returns:
            TransformedQuery with transformed research questions
        """
        try:
            # Import here to avoid circular imports

            # Get the transformation agent from coordinator
            transformation_agent = coordinator.agents.get("query_transformation_agent")
            if not transformation_agent:
                logfire.warning("QueryTransformationAgent not found in coordinator")
                return self._create_basic_transformation(original_query, clarification_responses)

            # Use the transformation agent to process the clarifications
            transformed_query = await transformation_agent.transform_query(
                original_query=original_query,
                clarification_responses=clarification_responses,
                conversation_context=deps.research_state.metadata.get("conversation_messages", []),
                deps=deps,
            )

            # Store the transformation in research state metadata
            deps.research_state.metadata["transformed_query"] = {
                "original_query": transformed_query.original_query,
                "transformed_query": transformed_query.transformed_query,
                "supporting_questions": transformed_query.supporting_questions,
                "specificity_score": transformed_query.specificity_score,
                "transformation_timestamp": transformed_query.created_at.isoformat(),
            }

            # Update clarified_query field with the transformed query
            deps.research_state.clarified_query = transformed_query.transformed_query

            # Emit completion event for the clarification stage
            await emit_stage_completed(
                request_id=deps.research_state.request_id,
                stage=ResearchStage.CLARIFICATION,
                success=True,
                result=transformed_query,
            )

            logfire.info(
                "Clarification and transformation completed",
                original_query=original_query,
                transformed_query=transformed_query.transformed_query[:100] + "...",
                specificity_score=transformed_query.specificity_score,
                request_id=deps.research_state.request_id,
            )

            return transformed_query

        except Exception as e:
            logfire.error(
                f"Error in clarification response processing: {str(e)}",
                original_query=original_query,
                request_id=deps.research_state.request_id,
                exc_info=True,
            )

            # Return basic transformation as fallback
            return self._create_basic_transformation(original_query, clarification_responses)

    def _create_basic_transformation(
        self,
        original_query: str,
        clarification_responses: dict[str, str],
    ) -> "TransformedQuery":
        """Create a basic transformation when the transformation agent is unavailable."""
        from open_deep_research_with_pydantic_ai.models.research import TransformedQuery

        # Simple enhancement of query with clarification responses
        enhanced_query = original_query
        context_parts = []

        for question, response in clarification_responses.items():
            if response.strip() and response.lower() not in ["no", "none", "n/a"]:
                if any(term in question.lower() for term in ["time", "when", "period"]):
                    context_parts.append(f"during {response}")
                elif any(term in question.lower() for term in ["where", "location", "region"]):
                    context_parts.append(f"in {response}")
                elif len(response) < 50:
                    context_parts.append(response)

        if context_parts:
            enhanced_query = f"{original_query} ({', '.join(context_parts[:3])})"

        return TransformedQuery(
            original_query=original_query,
            transformed_query=enhanced_query,
            supporting_questions=[],
            transformation_rationale=(
                "Basic transformation applied - full transformation agent unavailable"
            ),
            specificity_score=0.5,
            missing_dimensions=["comprehensive transformation"],
            clarification_responses=clarification_responses,
            transformation_metadata={
                "method": "basic_fallback",
                "reason": "transformation_agent_unavailable",
            },
        )


# Register the agent with the coordinator
clarification_agent = ClarificationAgent()
coordinator.register_agent(clarification_agent)
