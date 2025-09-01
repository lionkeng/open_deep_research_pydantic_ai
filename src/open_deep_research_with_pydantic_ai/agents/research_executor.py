"""Research Execution Agents for parallel information gathering."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from open_deep_research_with_pydantic_ai.agents.base import (
    AgentConfiguration,
    BaseResearchAgent,
    ResearchDependencies,
    coordinator,
)
from open_deep_research_with_pydantic_ai.core.events import (
    FindingDiscoveredEvent,
    emit_stage_completed,
    research_event_bus,
)
from open_deep_research_with_pydantic_ai.models.research import (
    ResearchBrief,
    ResearchFinding,
    ResearchResults,
    ResearchStage,
)
from open_deep_research_with_pydantic_ai.services.search import search_service

# Type alias to help PyRight with type inference
FindingsList = list[ResearchFinding]


class SearchResult(BaseModel):
    """Search result data model."""

    query: str = Field(description="The search query")
    results: list[dict[str, Any]] = Field(description="Search results")
    total_results: int = Field(default=0, description="Total number of results found")


class ResearchTask(BaseModel):
    """Individual research task model."""

    task_id: str = Field(description="Unique task identifier")
    description: str = Field(description="Task description")
    query: str = Field(description="Search query for this task")
    findings: FindingsList = Field(default_factory=list, description="Research findings")


class SpecializedResearchAgent(BaseResearchAgent[ResearchDependencies, ResearchResults]):
    """Specialized research agent for focused domain research."""

    def __init__(self):
        """Initialize the researchexecutor agent."""
        config = AgentConfiguration(
            agent_name="researchexecutor_agent",
            agent_type="researchexecutor",
        )
        super().__init__(config=config)
        self.sub_agents: dict[str, BaseResearchAgent[ResearchDependencies, BaseModel]] = {}

    def _get_output_type(self) -> type[ResearchResults]:
        """Get the output type for this agent."""
        return ResearchResults

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for research execution."""
        return """You are a research execution specialist. Your role is to:

1. Execute research tasks efficiently and thoroughly
2. Gather information from multiple sources
3. Evaluate source credibility and relevance
4. Extract key findings and insights
5. Maintain accuracy and objectivity

When conducting research:
- Use multiple search queries to cover different angles
- Verify information across multiple sources
- Prioritize authoritative and recent sources
- Extract specific facts, data, and insights
- Note any conflicting information
- Maintain proper source attribution

Quality criteria:
- Relevance: Information directly addresses the research question
- Credibility: Sources are authoritative and trustworthy
- Accuracy: Facts are verifiable and consistent
- Completeness: Coverage of all important aspects
- Recency: Information is up-to-date where relevant

Always provide structured findings with clear source attribution."""

    def _register_tools(self) -> None:
        """Register research execution tools."""

        @self.agent.tool
        async def web_search(
            ctx: RunContext[ResearchDependencies], query: str, num_results: int = 5
        ) -> SearchResult:
            """Perform a web search for information.

            Args:
                ctx: Run context with dependencies
                query: Search query
                num_results: Number of results to return

            Returns:
                Search results
            """
            # Use the search service with Tavily or fallback to mock
            search_response = await search_service.search(
                query=query,
                num_results=num_results,
                search_depth="advanced",
            )

            # Convert search response to SearchResult format
            results: list[dict[str, Any]] = []
            for result in search_response.results:
                results.append(
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "score": result.score,
                    }
                )

            return SearchResult(
                query=query,
                results=results,
                total_results=search_response.total_results,
                source=search_response.source,
            )

        @self.agent.tool
        async def extract_finding(
            ctx: RunContext[ResearchDependencies],
            content: str,
            source: str,
            relevance_score: float = 0.5,
        ) -> ResearchFinding:
            """Extract and structure a research finding.

            Args:
                ctx: Run context with dependencies
                content: Finding content
                source: Source URL or reference
                relevance_score: Relevance score (0-1)

            Returns:
                Structured research finding
            """
            # Create summary if content is long
            summary = None
            if len(content) > 500:
                summary = content[:200] + "..."

            finding = ResearchFinding(
                content=content,
                source=source,
                relevance_score=relevance_score,
                confidence=0.8,  # Default confidence
                summary=summary,
            )

            # Emit finding discovered event
            await research_event_bus.emit(
                FindingDiscoveredEvent(
                    _request_id=ctx.deps.research_state.request_id,
                    finding=finding,
                    agent=self.name,
                )
            )

            # Add to research state
            ctx.deps.research_state.add_finding(finding)

            return finding

        @self.agent.tool
        async def parallel_search(
            ctx: RunContext[ResearchDependencies], queries: list[str]
        ) -> list[SearchResult]:
            """Execute multiple searches in parallel.

            Args:
                ctx: Run context with dependencies
                queries: List of search queries

            Returns:
                List of search results
            """
            # Use search service for parallel search
            search_responses = await search_service.parallel_search(
                queries=queries,
                num_results=5,
            )

            # Convert to SearchResult format
            results: list[SearchResult] = []
            for response in search_responses:
                result_data: list[dict[str, Any]] = []
                for r in response.results:
                    result_data.append(
                        {
                            "title": r.title,
                            "url": r.url,
                            "snippet": r.snippet,
                            "score": r.score,
                        }
                    )

                results.append(
                    SearchResult(
                        query=response.query,
                        results=result_data,
                        total_results=response.total_results,
                        source=response.source,
                    )
                )

            return results

        @self.agent.tool
        async def evaluate_source_credibility(
            ctx: RunContext[ResearchDependencies], source: str
        ) -> float:
            """Evaluate the credibility of a source.

            Args:
                ctx: Run context with dependencies
                source: Source URL or reference

            Returns:
                Credibility score (0-1)
            """
            # Simple heuristic for source credibility
            credibility_indicators = {
                ".gov": 0.95,
                ".edu": 0.90,
                ".org": 0.80,
                "wikipedia": 0.75,
                "arxiv": 0.85,
                "pubmed": 0.90,
                "nature.com": 0.95,
                "science.org": 0.95,
                ".com": 0.60,
                ".blog": 0.50,
            }

            score = 0.5  # Default score
            for indicator, cred_score in credibility_indicators.items():
                if indicator in source.lower():
                    score = max(score, cred_score)
                    break

            return score

    def create_sub_agent(
        self, name: str, specialization: str
    ) -> BaseResearchAgent[ResearchDependencies, BaseModel]:
        """Create a specialized sub-agent for delegation.

        Args:
            name: Sub-agent name
            specialization: Area of specialization

        Returns:
            Specialized research agent
        """
        # Create proper specialized agent instead of recursive ResearchExecutorAgent
        sub_agent = SpecializedResearchAgent(
            name=name, specialization=specialization
        )

        self.sub_agents[name] = sub_agent
        return sub_agent

    async def execute_research(
        self,
        brief: ResearchBrief,
        deps: ResearchDependencies,
        max_parallel_tasks: int = 3,
    ) -> list[ResearchFinding]:
        """Execute research based on the research brief.

        Args:
            brief: Research brief with objectives and questions
            deps: Research dependencies
            max_parallel_tasks: Maximum parallel research tasks

        Returns:
            List of research findings
        """
        # Generate research tasks from brief
        tasks: list[ResearchTask] = []
        for i, question in enumerate(brief.key_questions[:max_parallel_tasks]):
            task = ResearchTask(
                task_id=f"task_{i}",
                description=f"Research: {question}",
                query=question,
                priority=i,
            )
            tasks.append(task)

        # Execute tasks in parallel
        all_findings: list[ResearchFinding] = []

        # Create prompts for each task
        task_prompts: list[str] = []
        for task in tasks:
            prompt = f"""Execute the following research task:

Task: {task.description}
Query: {task.query}

Research Context:
- Topic: {brief.topic}
- Objectives: {", ".join(brief.objectives[:3])}
- Scope: {brief.scope}

Instructions:
1. Search for relevant information
2. Evaluate source credibility
3. Extract key findings
4. Ensure accuracy and completeness

Provide structured findings with clear source attribution."""
            task_prompts.append(prompt)

        # Execute all tasks in parallel
        research_tasks = [self.run(prompt, deps, stream=True) for prompt in task_prompts]

        results = await asyncio.gather(*research_tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, list):
                all_findings.extend(result)
            elif isinstance(result, Exception):
                # Log error but continue with other results
                import logfire

                logfire.error(f"Research task failed: {str(result)}")

        # Update research state
        deps.research_state.findings = all_findings

        # Emit stage completed event
        await emit_stage_completed(
            request_id=deps.research_state.request_id,
            stage=ResearchStage.RESEARCH_EXECUTION,
            success=True,
            result={"findings_count": len(all_findings)},
        )

        return all_findings

    async def delegate_specialized_research(
        self,
        ctx: RunContext[ResearchDependencies],
        topic: str,
        specialization: str,
    ) -> list[ResearchFinding]:
        """Delegate specialized research to a sub-agent.

        Args:
            ctx: Run context
            topic: Research topic
            specialization: Area of specialization

        Returns:
            Specialized research findings
        """
        # Create or get specialized sub-agent
        if specialization not in self.sub_agents:
            sub_agent = self.create_sub_agent(specialization, specialization)
        else:
            sub_agent = self.sub_agents[specialization]

        # Delegate research
        prompt = f"Conduct specialized research on: {topic}"
        result = await self.delegate_to_agent(
            ctx,
            sub_agent,
            prompt,
            context={"specialization": specialization},
        )

        return result


# Register the agent with the coordinator
research_executor_agent = SpecializedResearchAgent()
coordinator.register_agent(research_executor_agent)
