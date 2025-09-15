# Implementation Design & Development Guide

> **🏗️ Architectural Overview**: For high-level system design and principles, see [System Architecture](./system_architecture.md)

## Table of Contents

- [Two-phase Clarification Implementation](#Two-phase-clarification-implementation)
- [Pydantic-AI Agent Patterns](#pydantic-ai-agent-patterns)
- [Memory-Safe Event System](#memory-safe-event-system)
- [Circuit Breaker Implementation](#circuit-breaker-implementation)
- [Data Models & Validation](#data-models--validation)
- [API Integration Patterns](#api-integration-patterns)
- [Performance Optimization](#performance-optimization)
- [Production Deployment](#production-deployment)
- [Development Patterns](#development-patterns)
- [Testing Strategies](#testing-strategies)

> **📋 Quick Reference**: This document provides detailed implementation patterns and code examples. For system architecture and design overview, see [System Architecture](./system_architecture.md)

---

## Two-phase Clarification Implementation

### Phase 1: Enhanced Clarification Assessment

The clarification agent analyzes query breadth and missing dimensions:

```python
# Implementation: agents/clarification.py
class ClarificationAgent(BaseResearchAgent[ResearchDependencies, ClarificationResult]):
    def __init__(self):
        super().__init__(
            name="clarification_agent",
            output_type=ClarificationResult,
        )

    def _get_default_system_prompt(self) -> str:
        return """You are a research query analysis specialist. Assess whether queries need
        clarification by identifying missing dimensions and ambiguity indicators.

        Key assessment criteria:
        - Audience level (beginner, expert, academic)
        - Temporal context (current, historical, future)
        - Scope specificity (narrow, broad, comprehensive)
        - Domain focus (technical, business, academic)
        - Purpose clarity (learning, analysis, decision-making)

        Generate targeted clarification questions when breadth score > 0.6."""

@clarification_agent.tool
async def assess_query_breadth(
    ctx: RunContext[ResearchDependencies],
    query: str
) -> dict[str, Any]:
    """Algorithmic analysis of query specificity with missing dimension detection."""
    broad_indicators = ["what is", "how does", "explain", "overview", "about"]
    missing_context = {
        "audience_level": not any(word in query.lower()
                                for word in ["beginner", "expert", "academic"]),
        "temporal_context": not any(word in query.lower()
                                  for word in ["recent", "2024", "current", "latest"]),
        "scope_specificity": not any(word in query.lower()
                                   for word in ["specific", "detailed", "comprehensive"]),
        "domain_focus": len([word for word in query.split() if len(word) > 10]) < 2,
        "purpose_clarity": not any(word in query.lower()
                                 for word in ["analyze", "compare", "evaluate"])
    }

    breadth_score = sum(missing_context.values()) / len(missing_context)
    broad_indicator_count = sum(1 for indicator in broad_indicators
                              if indicator in query.lower())

    # Adjust score based on broad indicators
    if broad_indicator_count > 0:
        breadth_score = min(1.0, breadth_score + (broad_indicator_count * 0.1))

    return {
        "breadth_score": breadth_score,
        "needs_clarification": breadth_score > 0.6,
        "missing_dimensions": [dim for dim, missing in missing_context.items() if missing],
        "broad_indicators_found": broad_indicator_count
    }

@clarification_agent.output_validator
async def validate_clarification_result(
    ctx: RunContext[ResearchDependencies],
    result: ClarificationResult
) -> ClarificationResult:
    """Quality assurance for clarification outputs."""
    if result.needs_clarification:
        if not result.question.strip():
            raise ModelRetry("Question required when clarification needed")
        if len(result.question.split()) < 5:
            raise ModelRetry("Clarification question must be substantial")
    else:
        if not result.verification.strip():
            raise ModelRetry("Verification required when no clarification needed")

    return result
```

### Phase 2: Query Transformation

Context-aware query enhancement with specificity scoring:

```python
# Implementation: agents/query_transformation.py
class QueryTransformationAgent(BaseResearchAgent[ResearchDependencies, TransformedQuery]):
    def __init__(self):
        super().__init__(
            name="query_transformation_agent",
            output_type=TransformedQuery,
        )

    def _get_default_system_prompt(self) -> str:
        return """You are a research query transformation specialist. Transform broad, vague
        queries into specific, focused, actionable research questions.

        Transformation guidelines:
        - Preserve original intent while adding specificity
        - Include temporal constraints when provided
        - Add geographical focus when specified
        - Incorporate domain-specific context
        - Generate supporting questions (max 2)
        - Ensure questions are researchable with available sources

        Specificity scoring:
        - 0.0-0.3: Very broad, needs significant refinement
        - 0.4-0.6: Moderately specific, some refinement possible
        - 0.7-0.9: Highly specific, well-defined scope
        - 1.0: Perfectly specific, no further refinement needed"""

    async def transform_query(
        self,
        original_query: str,
        clarification_responses: dict[str, str],
        conversation_context: list[str] | None = None,
        deps: ResearchDependencies | None = None,
    ) -> TransformedQuery:
        """Transform query based on clarification responses."""
        try:
            logfire.info(
                "Starting query transformation",
                original_query=original_query,
                num_clarifications=len(clarification_responses),
            )

            prompt = self._build_transformation_prompt(
                original_query, clarification_responses, conversation_context
            )

            if deps:
                result = await self.run(prompt, deps)
            else:
                # Fallback transformation
                result = self._create_fallback_transformation(
                    original_query, clarification_responses
                )

            # Store responses and enhance metadata
            result.clarification_responses = clarification_responses
            result = self._enhance_transformation_metadata(result, original_query)

            logfire.info(
                "Query transformation completed",
                specificity_score=result.specificity_score,
                transformed_length=len(result.transformed_query),
            )

            return result

        except Exception as e:
            logfire.error(f"Query transformation failed: {str(e)}")
            return self._create_fallback_transformation(original_query, clarification_responses)

    def _build_transformation_prompt(
        self,
        original_query: str,
        clarification_responses: dict[str, str],
        conversation_context: list[str] | None = None,
    ) -> str:
        """Build comprehensive transformation prompt."""
        prompt_parts = [
            f"Original Query: {original_query}",
            "",
            "Clarification Responses:"
        ]

        for question, response in clarification_responses.items():
            prompt_parts.extend([
                f"Q: {question}",
                f"A: {response}",
                ""
            ])

        if conversation_context:
            prompt_parts.extend([
                "Conversation Context:",
                "\n".join(conversation_context),
                ""
            ])

        prompt_parts.extend([
            "Transform this query following the system prompt guidelines.",
            "Focus on creating a specific, actionable research question that:",
            "1. Preserves the original user intent",
            "2. Incorporates clarification responses",
            "3. Adds temporal and domain specificity where appropriate",
            "4. Is researchable with available sources"
        ])

        return "\n".join(prompt_parts)

    def _create_fallback_transformation(
        self,
        original_query: str,
        clarification_responses: dict[str, str],
    ) -> TransformedQuery:
        """Fallback transformation when AI processing fails."""
        logfire.warning("Using fallback transformation due to processing error")

        # Basic enhancement: combine query with key responses
        enhanced_query = original_query
        context_additions = []

        for question, response in clarification_responses.items():
            if response.strip() and response.lower() not in ["no", "none", "n/a"]:
                if any(word in question.lower() for word in ["time", "when"]):
                    context_additions.append(f"during {response}")
                elif any(word in question.lower() for word in ["where", "region"]):
                    context_additions.append(f"in {response}")
                elif len(response) < 50:  # Short, specific responses
                    context_additions.append(response)

        if context_additions:
            enhanced_query = f"{original_query} ({', '.join(context_additions[:3])})"

        return TransformedQuery(
            original_query=original_query,
            transformed_query=enhanced_query,
            transformation_rationale="Fallback transformation - combined original query with clarification responses",
            specificity_score=0.4,  # Conservative score
            missing_dimensions=["detailed scope analysis", "comprehensive transformation"],
            clarification_responses=clarification_responses,
            transformation_metadata={"method": "fallback", "reason": "AI transformation failed"}
        )

    def _enhance_transformation_metadata(
        self, result: TransformedQuery, original_query: str
    ) -> TransformedQuery:
        """Add comprehensive metadata for analysis."""
        from datetime import datetime

        metadata = result.transformation_metadata or {}

        # Calculate transformation metrics
        original_words = set(original_query.lower().split())
        transformed_words = set(result.transformed_query.lower().split())

        metadata.update({
            "original_word_count": len(original_query.split()),
            "transformed_word_count": len(result.transformed_query.split()),
            "word_overlap_ratio": len(original_words & transformed_words) / max(len(original_words), 1),
            "transformation_timestamp": datetime.now().isoformat(),
            "agent": "query_transformation_agent",
            "enhancement_ratio": len(result.transformed_query) / max(len(original_query), 1)
        })

        result.transformation_metadata = metadata
        return result
```

### Phase 3: Enhanced Brief Generation

Comprehensive research planning with methodology suggestions:

```python
# Implementation: agents/brief_generator.py
class BriefGeneratorAgent(BaseResearchAgent[ResearchDependencies, BriefGenerationResult]):
    def __init__(self):
        super().__init__(
            name="brief_generation_agent",
            output_type=BriefGenerationResult,
        )

    def _get_default_system_prompt(self) -> str:
        return """You are a research planning specialist. Generate comprehensive research briefs
        that transform clarified queries into actionable research plans.

        Brief components:
        - Clear research objectives and scope
        - Key research areas and methodological approaches
        - Complexity assessment and resource requirements
        - Potential challenges and success criteria
        - Suggested sources and evaluation criteria

        Quality standards:
        - Minimum 100 words for comprehensive coverage
        - At least 3 key research areas
        - Methodology appropriate for query complexity
        - Realistic timeline and resource estimates
        - Specific success criteria and deliverables"""

    async def generate_from_conversation(
        self, deps: ResearchDependencies
    ) -> BriefGenerationResult:
        """Generate brief from conversation context in research state."""
        research_state = deps.research_state

        # Extract conversation elements
        conversation_messages = research_state.metadata.get("conversation_messages", [])
        clarification_assessment = research_state.metadata.get("clarification_assessment", {})
        transformed_query = research_state.metadata.get("transformed_query", {})

        # Build comprehensive prompt
        prompt_parts = [
            f"Original Query: {research_state.user_query}",
            ""
        ]

        if clarification_assessment:
            prompt_parts.extend([
                "Clarification Assessment:",
                f"- Needs clarification: {clarification_assessment.get('needs_clarification', False)}",
                f"- Confidence: {clarification_assessment.get('confidence_score', 0.5)}",
                f"- Missing dimensions: {clarification_assessment.get('missing_dimensions', [])}",
                ""
            ])

        if transformed_query:
            prompt_parts.extend([
                f"Transformed Query: {transformed_query.get('transformed_query', research_state.user_query)}",
                f"Transformation Rationale: {transformed_query.get('transformation_rationale', 'N/A')}",
                f"Specificity Score: {transformed_query.get('specificity_score', 0.5)}",
                ""
            ])

        if conversation_messages:
            prompt_parts.extend([
                "Conversation Context:",
                "\n".join(conversation_messages[-10:]),  # Last 10 messages
                ""
            ])

        prompt_parts.extend([
            "Generate a comprehensive research brief that:",
            "1. Synthesizes all available context",
            "2. Defines clear research objectives",
            "3. Identifies key research areas and methodologies",
            "4. Assesses complexity and resource requirements",
            "5. Provides actionable next steps"
        ])

        prompt = "\n".join(prompt_parts)
        result = await self.run(prompt, deps)

        # Store result in research state metadata
        research_state.metadata.update({
            "research_brief_text": result.brief_text,
            "research_brief_confidence": result.confidence_score,
            "research_brief_areas": result.key_research_areas,
            "research_brief_complexity": result.estimated_complexity
        })

        logfire.info(
            "Brief generation completed",
            confidence=result.confidence_score,
            areas_count=len(result.key_research_areas),
            complexity=result.estimated_complexity
        )

        return result

@brief_generation_agent.tool
async def analyze_research_requirements(
    ctx: RunContext[ResearchDependencies],
    query: str,
    context: dict[str, Any]
) -> dict[str, Any]:
    """Systematic analysis of research requirements and methodology selection."""
    # Analyze query complexity
    complexity_indicators = {
        "multi_domain": len([word for word in query.split() if len(word) > 8]) > 3,
        "technical_depth": any(term in query.lower() for term in
                              ["analysis", "evaluation", "comparison", "assessment"]),
        "temporal_scope": any(term in query.lower() for term in
                             ["trend", "change", "evolution", "development"]),
        "quantitative": any(term in query.lower() for term in
                           ["statistics", "data", "metrics", "numbers"])
    }

    complexity_score = sum(complexity_indicators.values()) / len(complexity_indicators)

    # Suggest methodologies based on complexity
    methodologies = []
    if complexity_score > 0.7:
        methodologies.extend(["systematic literature review", "meta-analysis", "expert interviews"])
    elif complexity_score > 0.4:
        methodologies.extend(["literature review", "case study analysis", "comparative analysis"])
    else:
        methodologies.extend(["web research", "basic literature review", "fact-finding"])

    # Estimate timeline
    if complexity_score > 0.7:
        timeline = "3-5 days"
        resource_level = "high"
    elif complexity_score > 0.4:
        timeline = "1-2 days"
        resource_level = "medium"
    else:
        timeline = "2-4 hours"
        resource_level = "low"

    return {
        "complexity_score": complexity_score,
        "complexity_indicators": complexity_indicators,
        "suggested_methodologies": methodologies,
        "estimated_timeline": timeline,
        "resource_level": resource_level,
        "recommended_sources": ["academic papers", "industry reports", "expert analysis"]
    }

@brief_generation_agent.output_validator
async def validate_brief_quality(
    ctx: RunContext[ResearchDependencies],
    result: BriefGenerationResult
) -> BriefGenerationResult:
    """Comprehensive brief quality validation."""
    # Length validation
    if len(result.brief_text.split()) < 100:
        raise ModelRetry("Brief must be at least 100 words for comprehensive coverage")

    # Key research areas validation
    if len(result.key_research_areas) < 1:
        raise ModelRetry("At least one key research area must be identified")

    # Consistency validation
    if result.confidence_score > 0.9 and result.estimated_complexity == "high":
        raise ModelRetry("High confidence inconsistent with high complexity assessment")

    if result.estimated_complexity == "low" and len(result.key_research_areas) > 5:
        raise ModelRetry("Low complexity should not require more than 5 research areas")

    # Content quality validation
    brief_lower = result.brief_text.lower()
    required_elements = ["objective", "research", "scope"]
    missing_elements = [elem for elem in required_elements if elem not in brief_lower]

    if missing_elements:
        raise ModelRetry(f"Brief missing key elements: {', '.join(missing_elements)}")

    return result

# Register agents
clarification_agent = ClarificationAgent()
query_transformation_agent = QueryTransformationAgent()
brief_generation_agent = BriefGeneratorAgent()

coordinator.register_agent(clarification_agent)
coordinator.register_agent(query_transformation_agent)
coordinator.register_agent(brief_generation_agent)
```

---

## Pydantic-AI Agent Patterns

### Base Agent Implementation Pattern

All research agents implement a standardized Pydantic-AI pattern with comprehensive error handling:

```python
# Base agent structure following Pydantic-AI best practices
class BaseResearchAgent[DepsT: ResearchDependencies, OutputT: BaseModel](ABC):
    """Base class for all research agents with Pydantic-AI integration."""

    def __init__(
        self,
        name: str,
        model: str | None = None,
        system_prompt: str | None = None,
        output_type: type[OutputT] | None = None,
    ):
        self.name = name

        # Get model configuration
        model_config = config.get_model_config()
        self.model = model_config["model"]  # claude-3-5-sonnet-20241022
        self._output_type = output_type

        # Create Pydantic AI agent with proper configuration
        self.agent: Agent[ResearchDependencies, Any] = Agent(
            model=self.model,
            retries=model_config.get("retries", 3),
            deps_type=ResearchDependencies,
            output_type=output_type or dict,
            system_prompt=system_prompt or self._get_default_system_prompt(),
        )

        # Register agent-specific tools
        self._register_tools()

        # Configure logging
        configure_logging()
        logfire.info(f"Initialized {self.name} agent", model=self.model)

    @abstractmethod
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for this agent."""
        pass

    def _register_tools(self) -> None:
        """Register agent-specific tools. Override in subclasses."""
        pass

    async def run(
        self,
        prompt: str,
        deps: DepsT,
        message_history: list[ModelMessage] | None = None,
        stream: bool = False,
    ) -> OutputT:
        """Run the agent with comprehensive error handling."""
        try:
            # Emit progress update
            if deps.stream_callback and stream:
                await research_event_bus.emit(
                    StreamingUpdateEvent(
                        _request_id=deps.research_state.request_id,
                        content=f"{self.name} processing...",
                        stage=deps.research_state.current_stage,
                    )
                )

            # Execute with retry logic
            result = await self.agent.run(
                prompt,
                deps=deps,
                message_history=message_history,
                usage=deps.usage,
            )

            logfire.info(
                f"{self.name} completed",
                request_id=deps.research_state.request_id,
                usage=result.usage() if result.usage() else None,
            )

            return result.output

        except ModelRetry:
            # Let Pydantic-AI handle retries
            raise
        except Exception as e:
            logfire.error(
                f"{self.name} failed",
                request_id=deps.research_state.request_id,
                error=str(e),
                exc_info=True,
            )
            raise
```

### Tool Registration Pattern

Pydantic-AI tools provide agents with specific capabilities:

```python
# Example: Research Executor Agent with comprehensive tools
class ResearchExecutorAgent(BaseResearchAgent[ResearchDependencies, ResearchResult]):
    def _register_tools(self) -> None:
        """Register research execution tools."""

        @self.agent.tool
        async def web_search(
            ctx: RunContext[ResearchDependencies],
            query: str,
            num_results: int = 10,
            search_type: Literal["general", "academic", "news"] = "general"
        ) -> SearchResult:
            """Perform web search using Exa API with semantic understanding."""
            try:
                # Use Exa search API via dependencies
                search_client = ctx.deps.http_client
                api_keys = ctx.deps.api_keys

                # Configure search parameters based on type
                search_params = {
                    "query": query,
                    "num_results": num_results,
                    "type": search_type,
                    "include_domains": self._get_trusted_domains(search_type),
                    "start_crawl_date": "2023-01-01"  # Recent content
                }

                response = await search_client.post(
                    "https://api.exa.ai/search",
                    headers={"x-api-key": api_keys.exa_api_key.get_secret_value()},
                    json=search_params
                )

                search_data = response.json()

                return SearchResult(
                    query=query,
                    results=[
                        SearchResultItem(
                            title=item["title"],
                            url=item["url"],
                            content=item.get("text", "")[:1000],
                            relevance_score=item.get("score", 0.5)
                        )
                        for item in search_data.get("results", [])
                    ],
                    total_results=search_data.get("total", 0),
                    search_metadata={
                        "search_type": search_type,
                        "api_provider": "exa",
                        "timestamp": datetime.now().isoformat()
                    }
                )

            except Exception as e:
                logfire.error(f"Web search failed: {e}", query=query)
                # Return empty result rather than failing
                return SearchResult(
                    query=query,
                    results=[],
                    total_results=0,
                    error_message=str(e)
                )

        @self.agent.tool
        async def extract_finding(
            ctx: RunContext[ResearchDependencies],
            content: str,
            source: str,
            relevance_score: float = 0.5
        ) -> ResearchFinding:
            """Extract structured finding from source content."""
            # Validate inputs
            if not content.strip():
                raise ValueError("Content cannot be empty")

            if not source.strip():
                raise ValueError("Source URL required")

            if not 0.0 <= relevance_score <= 1.0:
                raise ValueError("Relevance score must be between 0.0 and 1.0")

            # Create structured finding
            finding = ResearchFinding(
                content=content.strip()[:2000],  # Limit content length
                source=source,
                relevance_score=relevance_score,
                confidence=min(relevance_score + 0.1, 1.0),
                summary=content.strip()[:200] + "..." if len(content) > 200 else content.strip(),
                metadata={
                    "extraction_agent": "research_executor",
                    "content_length": len(content),
                    "extraction_timestamp": datetime.now().isoformat()
                }
            )

            # Store in research state
            ctx.deps.research_state.add_finding(finding)

            return finding

        @self.agent.tool
        async def evaluate_source_credibility(
            ctx: RunContext[ResearchDependencies],
            url: str,
            title: str,
            content: str = ""
        ) -> SourceCredibilityResult:
            """Evaluate source credibility with scoring and reasoning."""
            credibility_factors = {
                "domain_authority": self._assess_domain_authority(url),
                "content_quality": self._assess_content_quality(title, content),
                "recency": self._assess_content_recency(url, content),
                "author_expertise": self._assess_author_indicators(content),
                "citation_quality": self._assess_citation_indicators(content)
            }

            # Calculate overall credibility score
            credibility_score = sum(credibility_factors.values()) / len(credibility_factors)

            # Determine credibility level
            if credibility_score >= 0.8:
                credibility_level = "high"
            elif credibility_score >= 0.6:
                credibility_level = "medium"
            else:
                credibility_level = "low"

            return SourceCredibilityResult(
                url=url,
                credibility_score=credibility_score,
                credibility_level=credibility_level,
                factors=credibility_factors,
                recommendations=[
                    f"Domain authority: {credibility_factors['domain_authority']:.2f}",
                    f"Content quality: {credibility_factors['content_quality']:.2f}",
                    f"Recency assessment: {credibility_factors['recency']:.2f}"
                ],
                evaluation_reasoning=f"Overall credibility: {credibility_level} based on domain authority, content quality, and recency indicators."
            )

    def _get_trusted_domains(self, search_type: str) -> list[str]:
        """Get trusted domains based on search type."""
        domain_sets = {
            "academic": ["scholar.google.com", "arxiv.org", "pubmed.ncbi.nlm.nih.gov"],
            "news": ["reuters.com", "bbc.com", "apnews.com", "npr.org"],
            "general": ["wikipedia.org", "britannica.com", "gov", "edu"]
        }
        return domain_sets.get(search_type, domain_sets["general"])

    def _assess_domain_authority(self, url: str) -> float:
        """Assess domain authority based on URL patterns."""
        high_authority_domains = [".gov", ".edu", ".org"]
        medium_authority_domains = [".com", ".net"]

        url_lower = url.lower()

        if any(domain in url_lower for domain in high_authority_domains):
            return 0.9
        elif any(domain in url_lower for domain in medium_authority_domains):
            return 0.6
        else:
            return 0.3

    def _assess_content_quality(self, title: str, content: str) -> float:
        """Assess content quality based on structure and completeness."""
        quality_score = 0.5  # Base score

        # Title quality indicators
        if len(title.split()) >= 5:
            quality_score += 0.1

        # Content length indicators
        if len(content) > 1000:
            quality_score += 0.2
        elif len(content) > 500:
            quality_score += 0.1

        # Structure indicators
        if any(indicator in content.lower() for indicator in ["conclusion", "summary", "findings"]):
            quality_score += 0.1

        # Citation indicators
        if any(indicator in content for indicator in ["http://", "https://", "doi:"]):
            quality_score += 0.1

        return min(quality_score, 1.0)

    def _assess_content_recency(self, url: str, content: str) -> float:
        """Assess content recency based on URL and content indicators."""
        current_year = datetime.now().year

        # Look for year indicators in URL or content
        years_found = []
        for year in range(current_year - 5, current_year + 1):
            if str(year) in url or str(year) in content:
                years_found.append(year)

        if not years_found:
            return 0.3  # Unknown recency

        most_recent_year = max(years_found)
        years_old = current_year - most_recent_year

        # Score based on recency
        if years_old <= 1:
            return 1.0
        elif years_old <= 3:
            return 0.8
        elif years_old <= 5:
            return 0.5
        else:
            return 0.2

    def _assess_author_indicators(self, content: str) -> float:
        """Assess author expertise indicators in content."""
        expertise_indicators = [
            "dr.", "prof.", "phd", "professor", "researcher",
            "expert", "analyst", "specialist", "author"
        ]

        content_lower = content.lower()
        found_indicators = sum(1 for indicator in expertise_indicators
                             if indicator in content_lower)

        return min(found_indicators * 0.2, 1.0)

    def _assess_citation_indicators(self, content: str) -> float:
        """Assess citation quality indicators in content."""
        citation_indicators = [
            "references", "bibliography", "sources", "cited",
            "doi:", "arxiv:", "pubmed", "isbn"
        ]

        content_lower = content.lower()
        found_indicators = sum(1 for indicator in citation_indicators
                             if indicator in content_lower)

        return min(found_indicators * 0.25, 1.0)
```

### Output Validation Patterns

Comprehensive validation ensures high-quality agent outputs:

```python
# Example: Advanced output validation for research results
@research_executor_agent.output_validator
async def validate_research_result(
    ctx: RunContext[ResearchDependencies],
    result: ResearchResult
) -> ResearchResult:
    """Comprehensive research result validation."""

    # Minimum findings requirement
    if len(result.findings) < 3:
        raise ModelRetry("Research must produce at least 3 findings for comprehensive coverage")

    # Source diversity validation
    unique_domains = set()
    for finding in result.findings:
        try:
            from urllib.parse import urlparse
            domain = urlparse(finding.source).netloc
            unique_domains.add(domain)
        except:
            pass  # Skip invalid URLs

    if len(unique_domains) < 2:
        raise ModelRetry("Research findings must come from at least 2 different domains")

    # Quality threshold validation
    high_quality_findings = [f for f in result.findings if f.relevance_score >= 0.7]
    if len(high_quality_findings) < 2:
        raise ModelRetry("At least 2 findings must have high relevance scores (≥0.7)")

    # Content completeness validation
    empty_findings = [f for f in result.findings if len(f.content.strip()) < 50]
    if empty_findings:
        raise ModelRetry(f"Found {len(empty_findings)} findings with insufficient content")

    # Confidence consistency validation
    if result.overall_confidence > 0.8:
        low_confidence_findings = [f for f in result.findings if f.confidence < 0.5]
        if low_confidence_findings:
            raise ModelRetry("High overall confidence inconsistent with low-confidence findings")

    return result
```

---

## Memory-Safe Event System

### WeakRef-Based Event Bus

The event system uses advanced Python patterns to prevent memory leaks:

```python
# Implementation: core/events.py
import asyncio
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING
from weakref import WeakSet, WeakMethod

if TYPE_CHECKING:
    pass

@dataclass(frozen=True)
class ResearchEvent:
    """Base class for all research events."""
    _request_id: str

@dataclass(frozen=True)
class ClarificationRequestedEvent(ResearchEvent):
    """Event emitted when clarification is needed."""
    question: str
    query_context: str
    missing_dimensions: list[str]
    confidence_score: float

@dataclass(frozen=True)
class TransformationCompletedEvent(ResearchEvent):
    """Event emitted when query transformation completes."""
    original_query: str
    transformed_query: str
    specificity_improvement: float
    supporting_questions: list[str]

@dataclass(frozen=True)
class StreamingUpdateEvent(ResearchEvent):
    """Event for real-time progress updates."""
    content: str
    stage: ResearchStage
    is_partial: bool = True
    metadata: dict[str, Any] | None = None

class ResearchEventBus:
    """Memory-safe event bus using WeakRef patterns."""

    def __init__(self):
        # WeakSet automatically removes dead handler references
        self._handlers: dict[type, WeakSet] = defaultdict(WeakSet)
        self._history_lock = asyncio.Lock()

        # Bounded event history per user
        self._event_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._max_events_per_user = 1000

        # User isolation tracking
        self._event_count_by_user: dict[str, int] = defaultdict(int)
        self._active_users: set[str] = set()

        # Performance monitoring
        self._total_events_processed = 0
        self._cleanup_counter = 0

    def subscribe(self, event_type: type[ResearchEvent], handler: Callable) -> None:
        """Subscribe to events with automatic memory management."""
        if hasattr(handler, '__self__'):
            # Method handler - use WeakMethod to prevent circular references
            weak_handler = WeakMethod(handler)
        else:
            # Function handler - use weak reference
            weak_handler = weakref.ref(handler)

        self._handlers[event_type].add(weak_handler)

        logfire.info(
            "Event handler subscribed",
            event_type=event_type.__name__,
            handler_count=len(self._handlers[event_type])
        )

    async def emit(self, event: ResearchEvent) -> None:
        """Emit event to all subscribers with memory-safe delivery."""
        try:
            # Extract user ID from request ID for isolation
            user_id = self._extract_user_id(event._request_id)

            # Update tracking
            self._event_count_by_user[user_id] += 1
            self._active_users.add(user_id)
            self._total_events_processed += 1

            # Store in bounded history
            async with self._history_lock:
                self._event_history[user_id].append(event)

            # Deliver to handlers asynchronously
            await self._deliver_event(event)

            # Periodic cleanup trigger
            if self._total_events_processed % 1000 == 0:
                await self._periodic_cleanup()

        except Exception as e:
            logfire.error(f"Event emission failed: {e}", event_type=type(event).__name__)

    async def _deliver_event(self, event: ResearchEvent) -> None:
        """Deliver event to appropriate handlers."""
        event_type = type(event)
        handlers = self._handlers.get(event_type, WeakSet())

        # Create delivery tasks for all live handlers
        delivery_tasks = []

        for weak_handler in list(handlers):  # Copy to avoid modification during iteration
            handler = weak_handler() if hasattr(weak_handler, '__call__') else weak_handler()

            if handler is None:
                # Handler was garbage collected - will be automatically removed from WeakSet
                continue

            # Create async delivery task
            delivery_tasks.append(self._safe_handler_call(handler, event))

        # Execute all deliveries concurrently
        if delivery_tasks:
            await asyncio.gather(*delivery_tasks, return_exceptions=True)

    async def _safe_handler_call(self, handler: Callable, event: ResearchEvent) -> None:
        """Safely call event handler with error isolation."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logfire.error(
                f"Event handler failed: {e}",
                handler_name=getattr(handler, '__name__', 'unknown'),
                event_type=type(event).__name__,
                exc_info=True
            )

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of memory and stale references."""
        self._cleanup_counter += 1

        # Clean up event history for inactive users
        inactive_threshold = 100  # Events threshold for inactive users
        inactive_users = []

        for user_id, count in self._event_count_by_user.items():
            if count < inactive_threshold:
                inactive_users.append(user_id)

        # Remove inactive user data
        for user_id in inactive_users:
            if user_id in self._event_history:
                del self._event_history[user_id]
            if user_id in self._event_count_by_user:
                del self._event_count_by_user[user_id]
            self._active_users.discard(user_id)

        # Force garbage collection periodically
        if self._cleanup_counter % 10 == 0:
            import gc
            collected = gc.collect()
            logfire.info(
                "Periodic cleanup completed",
                objects_collected=collected,
                active_users=len(self._active_users),
                total_events=self._total_events_processed
            )

    async def cleanup_history(self, keep_recent: int = 500) -> None:
        """Clean up event history keeping only recent events."""
        async with self._history_lock:
            for user_id, history in self._event_history.items():
                if len(history) > keep_recent:
                    # Keep only recent events
                    recent_events = list(history)[-keep_recent:]
                    history.clear()
                    history.extend(recent_events)

    async def subscribe_to_request(self, request_id: str) -> AsyncIterator[ResearchEvent]:
        """Subscribe to events for a specific request with async iteration."""
        user_id = self._extract_user_id(request_id)

        # Create a queue for this subscription
        event_queue = asyncio.Queue()

        async def handler(event: ResearchEvent):
            if event._request_id == request_id:
                await event_queue.put(event)

        # Subscribe to all event types for this request
        event_types = [
            ClarificationRequestedEvent,
            TransformationCompletedEvent,
            StreamingUpdateEvent,
            # Add other event types as needed
        ]

        for event_type in event_types:
            self.subscribe(event_type, handler)

        try:
            while True:
                # Wait for next event with timeout
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=300.0)  # 5 min timeout
                    yield event

                    # Check if this is a completion event
                    if isinstance(event, ResearchCompletedEvent):
                        break

                except asyncio.TimeoutError:
                    # Subscription timed out
                    break
        finally:
            # Cleanup happens automatically via WeakRef
            pass

    def _extract_user_id(self, request_id: str) -> str:
        """Extract user ID from scoped request ID."""
        # Request ID format: user_id:session_id:uuid or user_id:uuid
        parts = request_id.split(":")
        return parts[0] if parts else "unknown"

    async def health_check(self) -> dict[str, Any]:
        """Health check for monitoring."""
        return {
            "total_handler_types": len(self._handlers),
            "total_handlers": sum(len(handlers) for handlers in self._handlers.values()),
            "active_users": len(self._active_users),
            "total_events_processed": self._total_events_processed,
            "cleanup_cycles": self._cleanup_counter,
            "memory_safe": True
        }

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive event bus statistics."""
        return {
            "total_handlers": sum(len(handlers) for handlers in self._handlers.values()),
            "handler_types": len(self._handlers),
            "active_users": len(self._active_users),
            "events_processed": self._total_events_processed,
            "cleanup_cycles": self._cleanup_counter,
            "event_history_users": len(self._event_history)
        }

# Global event bus instance
research_event_bus = ResearchEventBus()
```

### Context Management for User Isolation

```python
# Implementation: core/context.py
class ResearchContextManager:
    """Context manager for user-scoped research operations."""

    def __init__(self, user_id: str, session_id: str | None = None, request_id: str | None = None):
        self.user_id = user_id
        self.session_id = session_id
        self.request_id = request_id
        self._context_data: dict[str, Any] = {}

    async def __aenter__(self):
        """Enter user research context."""
        # Set up user isolation
        self._context_data = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "start_time": datetime.now(),
        }

        # Register with event bus for cleanup
        research_event_bus._active_users.add(self.user_id)

        logfire.info(
            "Research context established",
            user_id=self.user_id,
            session_id=self.session_id,
            request_id=self.request_id
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit user research context with cleanup."""
        duration = datetime.now() - self._context_data["start_time"]

        # Trigger cleanup for this user
        await research_event_bus.cleanup_history()

        logfire.info(
            "Research context completed",
            user_id=self.user_id,
            duration_seconds=duration.total_seconds(),
            had_exception=exc_type is not None
        )
```

---

## Circuit Breaker Implementation

### Circuit Breaker Pattern

Automatic failure detection and recovery for system stability:

```python
# Implementation: core/workflow.py (circuit breaker section)
import time
from enum import Enum
from typing import Dict, Optional

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit open, failing fast
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass

class ResearchWorkflow:
    """Main workflow orchestrator with circuit breaker protection."""

    def __init__(self):
        # Circuit breaker configuration
        self._circuit_breaker_threshold = 3  # Fail after 3 consecutive errors
        self._circuit_breaker_timeout = 60.0  # Reset after 1 minute
        self._half_open_max_calls = 3  # Max calls to test in half-open state

        # Circuit state tracking
        self._circuit_states: Dict[str, CircuitState] = defaultdict(lambda: CircuitState.CLOSED)
        self._consecutive_errors: Dict[str, int] = defaultdict(int)
        self._last_error_time: Dict[str, float] = {}
        self._last_success_time: Dict[str, float] = {}
        self._half_open_calls: Dict[str, int] = defaultdict(int)

        # Performance metrics
        self._total_calls: Dict[str, int] = defaultdict(int)
        self._total_errors: Dict[str, int] = defaultdict(int)
        self._total_successes: Dict[str, int] = defaultdict(int)

        # Concurrency control
        self._max_concurrent_tasks = 5
        self._semaphore = asyncio.Semaphore(self._max_concurrent_tasks)
        self._task_timeout = 300.0  # 5 minutes per task

        # Initialize coordinator
        self._ensure_initialized()

    def _ensure_initialized(self):
        """Ensure coordinator is properly initialized."""
        if not hasattr(self, 'coordinator'):
            from open_deep_research_pydantic_ai.agents.base import coordinator
            self.coordinator = coordinator

    async def _run_agent_with_circuit_breaker(
        self,
        agent_type: str,
        prompt: str,
        deps: ResearchDependencies,
        **kwargs
    ) -> Any:
        """Execute agent with circuit breaker protection and comprehensive error handling."""

        # Check circuit state
        if not self._check_circuit_breaker(agent_type):
            error_msg = f"Circuit breaker open for {agent_type}"
            logfire.warning(error_msg, agent_type=agent_type)

            # Try fallback behavior
            fallback_result = await self._get_fallback_result(agent_type, prompt, deps)
            if fallback_result is not None:
                return fallback_result

            raise CircuitBreakerError(error_msg)

        # Execute with comprehensive protection
        async with self._semaphore:  # Concurrency control
            try:
                # Track call
                self._total_calls[agent_type] += 1

                # Execute with timeout
                result = await asyncio.wait_for(
                    self.coordinator.run_agent(agent_type, prompt, deps, **kwargs),
                    timeout=self._task_timeout
                )

                # Record success
                self._record_success(agent_type)
                return result

            except asyncio.TimeoutError:
                error_msg = f"Agent {agent_type} timed out after {self._task_timeout}s"
                self._record_error(agent_type, error_msg)
                logfire.error(error_msg, agent_type=agent_type)
                raise

            except Exception as e:
                error_msg = str(e)
                self._record_error(agent_type, error_msg)

                # Check for specific error types
                if "rate limit" in error_msg.lower():
                    logfire.warning(f"Rate limit hit for {agent_type}: {error_msg}")
                    # Wait and retry once
                    await asyncio.sleep(5)
                    try:
                        result = await asyncio.wait_for(
                            self.coordinator.run_agent(agent_type, prompt, deps, **kwargs),
                            timeout=self._task_timeout
                        )
                        self._record_success(agent_type)
                        return result
                    except Exception as retry_error:
                        self._record_error(agent_type, str(retry_error))
                        raise retry_error

                logfire.error(
                    f"Agent {agent_type} failed",
                    error=error_msg,
                    consecutive_errors=self._consecutive_errors[agent_type],
                    exc_info=True
                )
                raise

    def _check_circuit_breaker(self, agent_type: str) -> bool:
        """Check circuit breaker state and handle transitions."""
        current_time = time.time()
        current_state = self._circuit_states[agent_type]

        if current_state == CircuitState.CLOSED:
            # Normal operation
            return True

        elif current_state == CircuitState.OPEN:
            # Check if timeout has expired
            last_error_time = self._last_error_time.get(agent_type, 0)
            if current_time - last_error_time >= self._circuit_breaker_timeout:
                # Transition to half-open
                self._transition_to_half_open(agent_type)
                return True
            else:
                # Still in timeout period
                return False

        elif current_state == CircuitState.HALF_OPEN:
            # Check if we've exceeded half-open call limit
            if self._half_open_calls[agent_type] < self._half_open_max_calls:
                self._half_open_calls[agent_type] += 1
                return True
            else:
                # Too many half-open calls, transition back to open
                self._transition_to_open(agent_type, "Half-open call limit exceeded")
                return False

        return False

    def _record_success(self, agent_type: str) -> None:
        """Record successful agent execution."""
        current_time = time.time()
        current_state = self._circuit_states[agent_type]

        # Update metrics
        self._total_successes[agent_type] += 1
        self._last_success_time[agent_type] = current_time
        self._consecutive_errors[agent_type] = 0  # Reset error count

        # Handle state transitions on success
        if current_state == CircuitState.HALF_OPEN:
            # Success in half-open - transition to closed
            self._transition_to_closed(agent_type)
        elif current_state == CircuitState.OPEN:
            # Shouldn't happen, but handle gracefully
            self._transition_to_closed(agent_type)

        logfire.info(
            f"Agent {agent_type} succeeded",
            agent_type=agent_type,
            consecutive_errors=0,
            circuit_state=self._circuit_states[agent_type].value
        )

    def _record_error(self, agent_type: str, error_message: str) -> None:
        """Record agent execution error and handle circuit state transitions."""
        current_time = time.time()
        current_state = self._circuit_states[agent_type]

        # Update metrics
        self._total_errors[agent_type] += 1
        self._consecutive_errors[agent_type] += 1
        self._last_error_time[agent_type] = current_time

        error_count = self._consecutive_errors[agent_type]

        # Handle state transitions based on error count
        if current_state == CircuitState.CLOSED:
            if error_count >= self._circuit_breaker_threshold:
                self._transition_to_open(agent_type, f"Error threshold reached: {error_count}")

        elif current_state == CircuitState.HALF_OPEN:
            # Any error in half-open transitions back to open
            self._transition_to_open(agent_type, "Error in half-open state")

        logfire.error(
            f"Agent {agent_type} error recorded",
            agent_type=agent_type,
            consecutive_errors=error_count,
            circuit_state=self._circuit_states[agent_type].value,
            error_message=error_message[:200]  # Truncate long messages
        )

    def _transition_to_open(self, agent_type: str, reason: str) -> None:
        """Transition circuit to open state."""
        old_state = self._circuit_states[agent_type]
        self._circuit_states[agent_type] = CircuitState.OPEN
        self._half_open_calls[agent_type] = 0  # Reset half-open counter

        logfire.warning(
            f"Circuit breaker OPENED for {agent_type}",
            agent_type=agent_type,
            old_state=old_state.value,
            reason=reason,
            consecutive_errors=self._consecutive_errors[agent_type],
            timeout_seconds=self._circuit_breaker_timeout
        )

    def _transition_to_half_open(self, agent_type: str) -> None:
        """Transition circuit to half-open state for testing."""
        self._circuit_states[agent_type] = CircuitState.HALF_OPEN
        self._half_open_calls[agent_type] = 0

        logfire.info(
            f"Circuit breaker HALF-OPEN for {agent_type}",
            agent_type=agent_type,
            max_test_calls=self._half_open_max_calls
        )

    def _transition_to_closed(self, agent_type: str) -> None:
        """Transition circuit to closed state (normal operation)."""
        old_state = self._circuit_states[agent_type]
        self._circuit_states[agent_type] = CircuitState.CLOSED
        self._consecutive_errors[agent_type] = 0
        self._half_open_calls[agent_type] = 0

        logfire.info(
            f"Circuit breaker CLOSED for {agent_type}",
            agent_type=agent_type,
            old_state=old_state.value,
            recovery_successful=True
        )

    async def _get_fallback_result(
        self,
        agent_type: str,
        prompt: str,
        deps: ResearchDependencies
    ) -> Any | None:
        """Provide fallback behavior when circuit is open."""

        if agent_type == "clarification":
            # Heuristic-based fallback for clarification
            query = deps.research_state.user_query
            word_count = len(query.split())

            return ClarificationResult(
                needs_clarification=word_count < 6,
                question="Could you provide more specific details about your research topic?" if word_count < 6 else "",
                verification="Query appears sufficiently specific for research." if word_count >= 6 else "",
                confidence_score=0.5,
                breadth_score=max(0.8 - (word_count * 0.1), 0.3),
                assessment_reasoning=f"Fallback assessment - {agent_type} agent unavailable. Basic word count analysis applied.",
                suggested_clarifications=["scope", "timeframe", "specific aspects"] if word_count < 6 else []
            )

        elif agent_type == "transformation":
            # Basic transformation fallback
            original_query = deps.research_state.user_query
            enhanced_query = f"{original_query} (comprehensive analysis needed)"

            return TransformedQuery(
                original_query=original_query,
                transformed_query=enhanced_query,
                transformation_rationale="Fallback transformation - added scope clarification",
                specificity_score=0.4,
                supporting_questions=["What are the key aspects?", "What scope is appropriate?"],
                clarification_responses={},
                domain_indicators=[],
                transformation_metadata={"method": "fallback", "agent_unavailable": agent_type}
            )

        elif agent_type == "brief_generation":
            # Basic brief generation fallback
            query = deps.research_state.user_query

            return BriefGenerationResult(
                brief_text=f"Research brief for: {query}. This is a fallback brief generated when the primary agent was unavailable. A comprehensive research plan would include detailed methodology, scope definition, and resource requirements.",
                confidence_score=0.3,
                key_research_areas=["primary research", "secondary analysis"],
                research_objectives=[f"Investigate {query}"],
                methodology_suggestions=["literature review", "web research"],
                estimated_complexity="medium",
                estimated_duration="1-2 hours",
                potential_challenges=["limited agent availability"],
                success_criteria=["basic information gathered"]
            )

        # No fallback available
        return None

    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker statistics."""
        stats = {
            "agent_stats": {},
            "system_stats": {
                "total_agent_types": len(self._circuit_states),
                "max_concurrent_tasks": self._max_concurrent_tasks,
                "task_timeout_seconds": self._task_timeout,
                "circuit_threshold": self._circuit_breaker_threshold,
                "circuit_timeout_seconds": self._circuit_breaker_timeout
            }
        }

        for agent_type in self._circuit_states.keys():
            stats["agent_stats"][agent_type] = {
                "circuit_state": self._circuit_states[agent_type].value,
                "consecutive_errors": self._consecutive_errors[agent_type],
                "total_calls": self._total_calls[agent_type],
                "total_successes": self._total_successes[agent_type],
                "total_errors": self._total_errors[agent_type],
                "success_rate": (
                    self._total_successes[agent_type] / max(self._total_calls[agent_type], 1)
                ),
                "last_error_time": self._last_error_time.get(agent_type),
                "last_success_time": self._last_success_time.get(agent_type),
                "half_open_calls": self._half_open_calls[agent_type]
            }

        return stats

# Create global workflow instance
workflow = ResearchWorkflow()
```

---

## Data Models & Validation

The system uses comprehensive Pydantic models with advanced validation patterns:

```python
# Core research models with cross-field validation
class ClarificationResult(BaseModel):
    """Structured output for clarification assessment."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    needs_clarification: bool
    confidence_score: Annotated[float, Field(ge=0.0, le=1.0)]
    breadth_score: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def validate_consistency(self) -> "ClarificationResult":
        if self.needs_clarification and self.confidence_score > 0.8:
            raise ValueError("High confidence inconsistent with need for clarification")
        return self
```

_For complete model definitions, see `src/models/research.py`_

---

## API Integration Patterns

### FastAPI Server-Sent Events

Real-time progress streaming to web clients:

```python
# SSE streaming for research progress
@app.get("/research/{request_id}/stream")
async def stream_research_updates(request_id: str, request: Request):
    """Stream research updates via Server-Sent Events."""
    return create_sse_response(request_id, request, active_sessions)

async def create_sse_response(request_id: str, request: Request, sessions: dict):
    """Create SSE response with event bus integration."""
    async def event_generator():
        async for event in research_event_bus.subscribe_to_request(request_id):
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps(event.model_dump())}\n\n"

    return StreamingResponse(event_generator(), media_type="text/plain")
```

### CLI Integration

Rich terminal interface with progress tracking:

```python
# CLI with Rich formatting
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

async def run_research_cli(query: str, mode: str = "direct"):
    """Execute research with beautiful terminal output."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Starting research...", total=None)

        if mode == "direct":
            result = await workflow.execute_research(query)
        else:
            result = await http_client_mode(query)

        progress.remove_task(task)
        console.print("[green]Research completed![/green]")
        return result
```

---

## Performance Optimization

### Concurrent Processing Patterns

```python
# Optimized concurrent search execution
class ResearchExecutorAgent:
    def __init__(self):
        self._search_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent searches
        self._extraction_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent extractions

    async def execute_parallel_research(self, queries: list[str]) -> list[ResearchFinding]:
        """Execute multiple research queries concurrently."""
        search_tasks = []

        for query in queries:
            task = asyncio.create_task(self._search_with_semaphore(query))
            search_tasks.append(task)

        # Wait for all searches with timeout
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Filter successful results and flatten
        findings = []
        for result in results:
            if isinstance(result, list):
                findings.extend(result)
            elif isinstance(result, Exception):
                logfire.error(f"Search failed: {result}")

        return findings

    async def _search_with_semaphore(self, query: str) -> list[ResearchFinding]:
        """Execute search with semaphore control."""
        async with self._search_semaphore:
            return await self.web_search(query)
```

### Caching and Optimization

```python
# Redis-based caching for production
class CachedResearchExecutor:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour

    async def cached_search(self, query: str) -> SearchResult:
        """Search with Redis caching."""
        cache_key = f"search:{hashlib.md5(query.encode()).hexdigest()}"

        # Try cache first
        cached_result = await self.redis.get(cache_key)
        if cached_result:
            return SearchResult.model_validate_json(cached_result)

        # Execute search
        result = await self.web_search(query)

        # Cache result
        await self.redis.setex(
            cache_key,
            self.cache_ttl,
            result.model_dump_json()
        )

        return result
```

---

## Production Deployment

### Kubernetes Configuration

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deep-research-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deep-research-api
  template:
    metadata:
      labels:
        app: deep-research-api
    spec:
      containers:
        - name: api
          image: deep-research:latest
          ports:
            - containerPort: 8000
          env:
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: api-secrets
                  key: anthropic-key
            - name: EXA_API_KEY
              valueFrom:
                secretKeyRef:
                  name: api-secrets
                  key: exa-key
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          resources:
            requests:
              memory: '512Mi'
              cpu: '250m'
            limits:
              memory: '1Gi'
              cpu: '500m'
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy application code
COPY src/ src/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uv", "run", "uvicorn", "src.open_deep_research_pydantic_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Configuration

```python
# Production settings
class ProductionSettings(Settings):
    """Production-specific configuration."""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    workers: int = 4

    # Redis Configuration
    redis_url: str = "redis://redis:6379/0"
    redis_max_connections: int = 20

    # Monitoring
    logfire_token: str = Field(..., description="Logfire monitoring token")
    prometheus_port: int = 9090

    # Security
    allowed_origins: list[str] = ["https://your-frontend.com"]
    rate_limit_per_minute: int = 60

    # Circuit Breaker
    circuit_failure_threshold: int = 5
    circuit_timeout_seconds: int = 120
```

---

## Development Patterns

### Testing Strategies

```python
# Comprehensive test patterns
import pytest
from unittest.mock import AsyncMock, Mock

@pytest.fixture
async def mock_dependencies():
    """Provide mock research dependencies."""
    return ResearchDependencies(
        http_client=AsyncMock(),
        api_keys=APIKeys(),
        research_state=ResearchState(
            request_id="test-123",
            user_id="test-user",
            session_id="test-session",
            user_query="Test query"
        ),
        metadata=ResearchMetadata(),
        usage=RunUsage()
    )

@pytest.mark.asyncio
async def test_clarification_agent_integration(mock_dependencies):
    """Test complete clarification workflow."""
    agent = ClarificationAgent()

    # Test with broad query
    result = await agent.assess_query(
        "What is machine learning?",
        mock_dependencies
    )

    assert result.needs_clarification is True
    assert result.breadth_score > 0.6
    assert len(result.missing_dimensions) > 0
    assert result.confidence_score < 0.8

# Performance testing
@pytest.mark.asyncio
async def test_concurrent_processing_performance():
    """Test system performance under concurrent load."""
    import time

    start_time = time.time()

    # Simulate 10 concurrent research requests
    tasks = [
        workflow.execute_planning_only(
            f"Test query {i}",
            APIKeys()
        )
        for i in range(10)
    ]

    results = await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time

    # Should complete within reasonable time
    assert duration < 30.0
    assert len(results) == 10
    assert all(r.current_stage == ResearchStage.BRIEF_GENERATION for r in results)
```

### Code Quality Patterns

```python
# Quality assurance patterns
from typing import TypeVar, Generic

T = TypeVar('T', bound=BaseModel)

class QualityValidator(Generic[T]):
    """Generic quality validator for research outputs."""

    def __init__(self, model_class: type[T]):
        self.model_class = model_class
        self._quality_thresholds = self._get_quality_thresholds()

    def validate_quality(self, instance: T) -> list[str]:
        """Validate instance against quality criteria."""
        issues = []

        # Check confidence thresholds
        if hasattr(instance, 'confidence_score'):
            if instance.confidence_score < self._quality_thresholds.get('min_confidence', 0.5):
                issues.append(f"Low confidence: {instance.confidence_score}")

        # Check content completeness
        if hasattr(instance, 'content') and len(instance.content.strip()) < 50:
            issues.append("Content too brief for quality assessment")

        return issues

    def _get_quality_thresholds(self) -> dict[str, float]:
        """Get quality thresholds for model type."""
        thresholds = {
            ClarificationResult: {'min_confidence': 0.6},
            BriefGenerationResult: {'min_confidence': 0.7},
            ResearchResult: {'min_confidence': 0.5}
        }
        return thresholds.get(self.model_class, {})
```

---

_This implementation design document provides comprehensive code examples and patterns for building production-ready research systems with Pydantic-AI. For architectural overview and system design principles, see the companion [System Architecture](./system_architecture.md) document._
