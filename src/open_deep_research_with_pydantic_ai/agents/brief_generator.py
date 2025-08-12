"""Research Brief Generator Agent for creating structured research plans."""

from pydantic_ai import RunContext

from open_deep_research_with_pydantic_ai.agents.base import (
    BaseResearchAgent,
    ResearchDependencies,
    coordinator,
)
from open_deep_research_with_pydantic_ai.core.events import (
    emit_stage_completed,
)
from open_deep_research_with_pydantic_ai.models.research import ResearchBrief, ResearchStage


class BriefGeneratorAgent(BaseResearchAgent[ResearchDependencies, ResearchBrief]):
    """Agent responsible for generating structured research briefs."""

    def __init__(self, model: str = "openai:gpt-4o"):
        """Initialize the brief generator agent."""
        super().__init__(
            name="brief_generator_agent",
            model=model,
            output_type=ResearchBrief,
        )

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for brief generation."""
        return """You are a research planning specialist. Your role is to create comprehensive,
structured research briefs that guide the research process effectively.

When creating a research brief:

1. **Topic Definition**: Clearly state the main research topic in specific terms
2. **Objectives**: List 3-5 specific, measurable research objectives
3. **Key Questions**: Formulate 5-8 key questions that the research should answer
4. **Scope**: Define clear boundaries - what will and won't be covered
5. **Priority Areas**: Identify the most important aspects to focus on
6. **Constraints**: Note any limitations (time, resources, access to information)
7. **Deliverables**: Specify expected outputs and their format

Guidelines:
- Be specific and actionable in all sections
- Ensure objectives are achievable within a reasonable timeframe
- Questions should be answerable through available research methods
- Prioritize depth over breadth when scope is broad
- Consider multiple perspectives and potential biases
- Include both qualitative and quantitative aspects where relevant

Structure the brief to be:
- Comprehensive yet focused
- Logical and well-organized
- Clear and unambiguous
- Actionable for research execution"""

    def _register_tools(self) -> None:
        """Register brief generation-specific tools."""

        @self.agent.tool
        async def decompose_topic(  # pyright: ignore[reportUnusedFunction]
            _ctx: RunContext[ResearchDependencies], topic: str
        ) -> dict[str, list[str]]:
            """Decompose a topic into subtopics and aspects.

            Args:
                ctx: Run context with dependencies
                topic: Main topic to decompose

            Returns:
                Dictionary with subtopics and aspects
            """
            # Generate subtopics based on common research dimensions
            subtopics: list[str] = []
            aspects: list[str] = []

            # Common research dimensions
            dimensions = [
                "historical context",
                "current state",
                "future trends",
                "key stakeholders",
                "challenges and opportunities",
                "best practices",
                "case studies",
                "comparative analysis",
                "impact assessment",
                "technical details",
            ]

            # Generate relevant subtopics
            for dimension in dimensions[:5]:  # Limit to top 5 most relevant
                subtopics.append(f"{topic} - {dimension}")

            # Generate research aspects
            aspect_categories = [
                "Economic implications",
                "Social impact",
                "Technical considerations",
                "Policy and regulation",
                "Environmental factors",
            ]

            for category in aspect_categories[:3]:  # Top 3 aspects
                aspects.append(f"{category} of {topic}")

            return {
                "subtopics": subtopics,
                "aspects": aspects,
            }

        @self.agent.tool
        async def generate_research_questions(  # pyright: ignore[reportUnusedFunction]
            _ctx: RunContext[ResearchDependencies], topic: str, objectives: list[str]
        ) -> list[str]:
            """Generate research questions based on topic and objectives.

            Args:
                ctx: Run context with dependencies
                topic: Research topic
                objectives: Research objectives

            Returns:
                List of research questions
            """
            questions: list[str] = []

            # Question templates for different types of research
            templates = [
                "What are the main factors influencing {topic}?",
                "How has {topic} evolved over time?",
                "What are the current best practices in {topic}?",
                "What challenges and opportunities exist in {topic}?",
                "How does {topic} compare across different contexts?",
                "What is the impact of {topic} on stakeholders?",
                "What future developments are expected in {topic}?",
                "What are the key success factors for {topic}?",
            ]

            # Generate questions from templates
            for template in templates[:5]:
                questions.append(template.format(topic=topic))

            # Add objective-specific questions
            for obj in objectives[:3]:
                questions.append(f"How can we achieve: {obj}?")

            return questions

        @self.agent.tool
        async def identify_priority_areas(  # pyright: ignore[reportUnusedFunction]
            _ctx: RunContext[ResearchDependencies], topic: str, complexity: str
        ) -> list[str]:
            """Identify priority areas for research focus.

            Args:
                ctx: Run context with dependencies
                topic: Research topic
                complexity: Complexity level (simple, medium, complex)

            Returns:
                List of priority areas
            """
            priority_areas: list[str] = []

            if complexity == "complex":
                # For complex topics, focus on foundational understanding first
                priority_areas.extend(
                    [
                        f"Fundamental concepts and definitions in {topic}",
                        f"Key theoretical frameworks for understanding {topic}",
                        f"Major debates and controversies in {topic}",
                        f"Interdisciplinary connections of {topic}",
                        f"Methodological approaches to studying {topic}",
                    ]
                )
            elif complexity == "medium":
                # For medium complexity, balance theory and practice
                priority_areas.extend(
                    [
                        f"Core principles of {topic}",
                        f"Practical applications of {topic}",
                        f"Recent developments in {topic}",
                        f"Common challenges in {topic}",
                    ]
                )
            else:
                # For simple topics, focus on practical information
                priority_areas.extend(
                    [
                        f"Basic overview of {topic}",
                        f"Key facts and figures about {topic}",
                        f"Common use cases for {topic}",
                    ]
                )

            return priority_areas[:4]  # Return top 4 priorities

    async def generate_brief(
        self,
        clarified_query: str,
        complexity: str,
        deps: ResearchDependencies,
    ) -> ResearchBrief:
        """Generate a structured research brief.

        Args:
            clarified_query: Clarified research query
            complexity: Complexity assessment
            deps: Research dependencies

        Returns:
            Structured research brief
        """
        prompt = f"""Create a comprehensive research brief for the following query:

Query: {clarified_query}
Complexity Level: {complexity}

Generate a structured research brief that includes:
1. Clear topic definition
2. Specific, measurable objectives (3-5)
3. Key research questions (5-8)
4. Well-defined scope and boundaries
5. Priority areas for focus
6. Any constraints or limitations
7. Expected deliverables

Ensure the brief is actionable and provides clear guidance for research execution."""

        result = await self.run(prompt, deps, stream=True)

        # Update research state
        deps.research_state.research_brief = result

        # Emit stage completed event
        await emit_stage_completed(
            request_id=deps.research_state.request_id,
            stage=ResearchStage.BRIEF_GENERATION,
            success=True,
            result=result,
        )

        return result


# Register the agent with the coordinator
brief_generator_agent = BriefGeneratorAgent()
coordinator.register_agent(brief_generator_agent)
