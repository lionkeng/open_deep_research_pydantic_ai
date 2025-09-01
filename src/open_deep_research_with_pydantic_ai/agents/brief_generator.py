"""Research Brief Generator Agent - transforms conversation into research brief."""

from datetime import datetime

from pydantic import BaseModel, Field

from open_deep_research_with_pydantic_ai.agents.base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
    coordinator,
)
from open_deep_research_with_pydantic_ai.core.events import (
    emit_stage_completed,
)
from open_deep_research_with_pydantic_ai.models.research import ResearchStage


class ResearchBrief(BaseModel):
    """Research brief with confidence scoring - matches our implementation plan."""

    brief: str = Field(description="Detailed research brief from conversation")
    confidence_score: float = Field(
        description="Confidence in brief completeness (0.0-1.0)", ge=0.0, le=1.0
    )
    missing_aspects: list[str] = Field(
        default_factory=list, description="What aspects are still unclear or missing"
    )


class BriefGeneratorAgent(BaseResearchAgent[ResearchDependencies, ResearchBrief]):
    """Agent responsible for transforming conversation into research brief.

    This agent follows Langgraph's approach:
    - Takes full conversation history
    - Transforms messages into detailed research question
    - Provides confidence scoring for completeness
    - Identifies missing aspects that may need clarification
    """

    def __init__(self):
        """Initialize the brief generator agent."""
        config = AgentConfiguration(
            agent_name="brief_generator_agent",
            agent_type="brief_generator",
        )
        super().__init__(config=config)

    def _get_output_type(self) -> type[ResearchBrief]:
        """Get the output type for this agent."""
        return ResearchBrief

    def _get_default_system_prompt(self) -> str:
        """Get system prompt from Langgraph's transform_messages_into_research_topic_prompt."""
        return """You will be given a set of messages that have been exchanged between a user
and an assistant.
Your job is to translate these messages into a more detailed and concrete research question
that will be used to guide the research.

Today's date is {date}.

You will return a research brief with a confidence score indicating how complete the information is.

Guidelines:
1. **Maximize Specificity and Detail**
   - Include all known user preferences and explicitly list key attributes or dimensions
     to consider.
   - All details from the user must be included in the research brief.

2. **Fill in Unstated But Necessary Dimensions as Open-Ended**
   - If certain attributes are essential for meaningful output but the user hasn't provided them,
     explicitly state that they are open-ended or default to no specific constraint.

3. **Avoid Unwarranted Assumptions**
   - If the user hasn't provided a particular detail, don't invent one.
   - Instead, state the lack of specification and guide the researcher to treat it as flexible.

4. **Use the First Person**
   - Phrase the request from the perspective of the user.

5. **Sources**
   - If specific sources should be prioritized, specify them in the research question.
   - For product/travel research, prefer official sites over aggregators.
   - For academic queries, prefer original papers over summaries.
   - If query is in a specific language, prioritize sources in that language.

6. **Confidence Scoring**
   - Score 0.8-1.0: Very clear, specific request with all necessary context
   - Score 0.6-0.7: Generally clear but missing some important details
   - Score 0.3-0.5: Ambiguous or missing significant context
   - Score 0.0-0.2: Very unclear, needs substantial clarification

7. **Missing Aspects**
   - Identify specific information gaps that would improve the research
   - Consider: timeframe, location, scope, specific criteria, target audience, etc."""

    def _register_tools(self) -> None:
        """Register brief generation-specific tools.

        For this agent, we don't need complex tools - the LLM will analyze
        the full conversation history to generate a research brief.
        """
        # No tools needed for conversation-to-brief transformation
        pass

    async def generate_from_conversation(
        self,
        deps: ResearchDependencies,
    ) -> ResearchBrief:
        """Generate research brief from conversation history.

        Args:
            deps: Research dependencies with conversation history in state

        Returns:
            Research brief with confidence score
        """
        # Get conversation history from research state metadata
        metadata = deps.research_state.metadata or {}
        conversation_history = metadata.get("conversation_messages", [])

        # Include the original query if no conversation history
        if not conversation_history and deps.research_state.user_query:
            conversation_history = [deps.research_state.user_query]

        # Build messages context
        if conversation_history:
            messages_context = "\n".join(
                [
                    f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}"
                    for i, msg in enumerate(conversation_history)
                ]
            )
        else:
            messages_context = f"User: {deps.research_state.user_query or 'No query provided'}"

        # Format the prompt with conversation and date
        prompt = f"""The messages that have been exchanged between yourself and the user are:
<Messages>
{messages_context}
</Messages>

Transform these messages into a detailed research brief that will guide the research process.

Your response should include:
- A comprehensive research brief written from the user's perspective
- A confidence score (0.0-1.0) indicating how complete the information is
- A list of missing aspects that would improve the research if clarified

Remember to:
- Include all user-provided details in the brief
- Be specific about requirements and constraints
- Identify information gaps that might need clarification
- Write the brief from the user's perspective ("I want to research...")"""

        # Format the prompt with current date
        formatted_prompt = prompt.replace("{date}", datetime.now().strftime("%Y-%m-%d"))

        # Run the agent
        result = await self.run(formatted_prompt, deps, stream=False)

        # Update research state metadata with brief and confidence
        # Note: We store the brief as a string in a separate field since
        # ResearchState.research_brief
        # expects a different ResearchBrief model structure
        if deps.research_state.metadata:
            deps.research_state.metadata.update(
                {
                    "research_brief_confidence": result.confidence_score,
                    "research_brief_text": result.brief,
                }
            )
        else:
            deps.research_state.metadata = {
                "research_brief_confidence": result.confidence_score,
                "research_brief_text": result.brief,
            }

        # Emit stage completed event
        await emit_stage_completed(
            request_id=deps.research_state.request_id,
            stage=ResearchStage.BRIEF_GENERATION,
            success=result.confidence_score >= 0.3,  # Success if confidence > 0.3
            result=result,
        )

        return result


# Register the agent with the coordinator
brief_generator_agent = BriefGeneratorAgent()
coordinator.register_agent(brief_generator_agent)
