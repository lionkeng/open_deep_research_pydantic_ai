# Research Executor Agent: Consolidated Implementation Plan

## Executive Summary

This document provides a comprehensive implementation plan for the Research Executor Agent using an **Enhanced Hybrid Architecture** that combines deterministic search execution with advanced intelligent synthesis. This approach guarantees 100% query execution fidelity while leveraging GPT-4o/GPT-5's advanced reasoning capabilities with proven prompt engineering patterns.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Architecture Principles](#core-architecture-principles)
3. [Type System Design](#type-system-design)
4. [Core Components](#core-components)
5. [Enhanced Synthesis Agent](#enhanced-synthesis-agent)
6. [System Prompts](#system-prompts)
7. [Implementation Details](#implementation-details)
8. [Testing Strategy](#testing-strategy)
9. [Configuration](#configuration)
10. [Summary](#summary)

## Architecture Overview

### Current State

The Research Executor Agent receives:

- `SearchQueryBatch` from Query Transformation Agent (10-15 specific queries)
- Each query has priority, type, and execution parameters
- Need to execute ALL queries deterministically

### Implementation Plan for an Enhanced Hybrid Architecture

Combine the best of both worlds:

1. **Deterministic Search Layer**: Use Tavily/external services for guaranteed execution
2. **Enhanced Intelligent Synthesis Layer**: Use GPT-4o/GPT-5 with advanced prompt patterns
3. **Quality Assurance Layer**: Track metrics and ensure completeness

## Core Architecture Principles

### 1. Hybrid Execution Model

- **Deterministic Search**: Every query in SearchQueryBatch executes exactly as specified
- **Intelligent Synthesis**: Advanced LLM analyzes and synthesizes all results
- **Quality Verification**: Self-verification protocols ensure synthesis quality

### 2. Unified Type System

- **Rich Data Models**: Comprehensive `ResearchResults` model with full synthesis capabilities
- **Direct Integration**: All features directly exposed in the primary models
- **Clean Architecture**: No legacy constraints or workarounds

### 3. Phased Implementation Strategy

- **Phase 1 (MVP)**: Core synthesis with essential features
  - Streamlined models (defer PatternAnalysis for Phase 2)
  - Research Executor Agent with 4 core tools
  - SynthesisEngine with full ML clustering
  - Simplified ContradictionDetector (2 types)
  - Core integration tests
- **Phase 2**: Enhanced synthesis and analysis capabilities
  - PatternAnalysis model and PatternRecognizer service
  - ConfidenceAnalyzer service
  - Enhanced contradiction detection (4 types)
  - Comprehensive test coverage
- **Phase 3**: Performance optimizations and advanced features
  - Caching and parallel execution
  - Advanced metrics and monitoring
  - Production optimizations

## Core Components

### 1. SearchOrchestrator

Manages deterministic execution of all search queries.

```python
class SearchOrchestrator:
    """
    Responsibilities:
    - Execute every query in SearchQueryBatch
    - Respect execution strategy (parallel/sequential/hierarchical)
    - Handle retries and fallbacks
    - Track execution trace
    """
```

### 2. Enhanced Internal Synthesis Agent

An advanced Pydantic AI agent within Research Executor for intelligent analysis.

```python
class ResearchExecutorAgent:
    def __init__(self):
        self.synthesis_agent = self._create_enhanced_synthesis_agent()
        self.synthesis_tools = SynthesisTools()

    def _create_enhanced_synthesis_agent(self):
        return Agent(
            model='openai:gpt-4o',  # or gpt-5 when available
            system_prompt=ENHANCED_SYNTHESIS_PROMPT,
            result_type=ResearchResults
        )
```

**Important**: This is NOT a separate agent in the workflow. It's an internal component of the Research Executor Agent.

### 3. QualityMonitor

Enhanced metrics tracking with synthesis quality assessment.

```python
class QualityMonitor:
    """
    Enhanced metrics tracked:
    - Query execution rate (must be 100%)
    - Source diversity score
    - Pattern recognition accuracy
    - Synthesis coherence score
    - Information hierarchy distribution
    - Contradiction detection rate
    - Confidence calibration accuracy
    """
```

### 4. SynthesisTools

Advanced synthesis operations toolset.

```python
class SynthesisTools:
    """
    Tool suite for synthesis enhancement:
    - Information hierarchy scoring
    - Contradiction detection
    - Convergence analysis
    - Quality verification
    - Metrics generation
    """
```

## Enhanced Synthesis Agent

### Information Hierarchy Framework

The synthesis agent uses a sophisticated information hierarchy scoring system:

```python
class InformationHierarchy:
    CRITICAL = 1.0      # Core facts, primary conclusions, unique insights
    IMPORTANT = 0.7     # Supporting evidence, methodologies, key patterns
    SUPPLEMENTARY = 0.4 # Context, elaborations, secondary examples
    CONTEXTUAL = 0.0    # Background, tangential information
```

### Tree of Thoughts Structure

```
Research Synthesis Process
├── Pattern Recognition
│   ├── Convergence Analysis (agreement across sources)
│   ├── Divergence Mapping (contradictions/conflicts)
│   └── Emergence Detection (new trends/signals)
├── Insight Extraction
│   ├── Primary Insights (directly address query)
│   ├── Secondary Insights (related discoveries)
│   └── Meta-Insights (patterns about patterns)
└── Quality Verification
    ├── Completeness Check
    ├── Coherence Validation
    └── Confidence Calibration
```

## Type System Design

### Primary Research Models

```python
# src/models/research_executor.py
from pydantic import BaseModel, Field
from typing import List, Dict, Set, Optional, Literal
from datetime import datetime
from enum import Enum

class ConfidenceLevel(str, Enum):
    """Confidence levels for findings"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"

class ImportanceLevel(str, Enum):
    """Importance hierarchy for findings"""
    CRITICAL = "critical"
    IMPORTANT = "important"
    SUPPLEMENTARY = "supplementary"
    CONTEXTUAL = "contextual"

class SourceQuality(str, Enum):
    """Quality assessment for sources"""
    AUTHORITATIVE = "authoritative"
    RELIABLE = "reliable"
    MODERATE = "moderate"
    QUESTIONABLE = "questionable"

class ResearchSource(BaseModel):
    """Enhanced source with quality assessment"""
    id: str = Field(..., description="Unique source identifier")
    url: Optional[str] = Field(default=None)
    title: str
    author: Optional[str] = Field(default=None)
    date: Optional[datetime] = Field(default=None)
    relevance_score: float = Field(ge=0.0, le=1.0)
    credibility_tier: int = Field(ge=1, le=4, description="1=highest, 4=lowest")
    source_type: str  # "academic", "industry", "news", "documentation"
    quality: SourceQuality

class HierarchicalFinding(BaseModel):
    """Finding with full hierarchical and relational data"""
    id: str = Field(..., description="Unique finding ID")
    finding: str = Field(..., description="The finding text")
    supporting_evidence: List[str] = Field(default_factory=list)
    confidence_level: float = Field(ge=0.0, le=1.0)
    confidence_category: ConfidenceLevel
    source: Optional[ResearchSource] = None
    category: Optional[str] = None

    # Hierarchical classification
    importance_level: ImportanceLevel
    theme_cluster: str
    sub_findings: List[str] = Field(default_factory=list)

    # Relationships
    supports: List[str] = Field(default_factory=list, description="IDs of findings this supports")
    contradicts: List[str] = Field(default_factory=list, description="IDs of findings this contradicts")
    related_to: List[str] = Field(default_factory=list, description="IDs of related findings")

class ThemeCluster(BaseModel):
    """Organized theme with related findings"""
    theme_name: str
    description: str
    finding_ids: List[str]  # Finding IDs in this cluster
    confidence: float = Field(ge=0.0, le=1.0)
    consensus_level: Literal["strong", "moderate", "weak", "conflicting"]
    key_insights: List[str]

class PatternAnalysis(BaseModel):
    """Identified patterns across research"""
    pattern_type: Literal["convergence", "divergence", "emergence", "temporal"]
    description: str
    supporting_finding_ids: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    implications: List[str]

class Contradiction(BaseModel):
    """Detailed contradiction information"""
    finding_a_id: str
    finding_b_id: str
    contradiction_type: Literal["direct", "partial", "contextual", "methodological"]
    explanation: str
    resolution_suggestion: Optional[str] = None
    severity: Literal["high", "medium", "low"]

class ExecutiveSummary(BaseModel):
    """Structured executive summary"""
    key_findings: List[str]  # 3-5 bullet points
    overall_confidence: float = Field(ge=0.0, le=1.0)
    critical_gaps: List[str]
    immediate_insights: List[str]
    strategic_implications: List[str]

class SynthesisMetadata(BaseModel):
    """Metadata about the synthesis process"""
    synthesis_timestamp: datetime
    synthesis_version: str = "2.0"
    total_findings_analyzed: int
    synthesis_approach: Literal["technical", "business", "narrative", "academic"]
    verification_completed: bool
    verification_checklist: Dict[str, bool]
    quality_metrics: Dict[str, float]

class ResearchResults(BaseModel):
    """Complete research results with integrated synthesis"""

    # Core fields
    query: str
    execution_time: datetime

    # Hierarchical findings
    findings: List[HierarchicalFinding]

    # Synthesis outputs
    executive_summary: ExecutiveSummary
    theme_clusters: List[ThemeCluster]
    pattern_analysis: List[PatternAnalysis]
    contradictions: List[Contradiction]

    # Aggregated insights
    key_insights: List[str]
    actionable_recommendations: List[str]
    data_gaps: List[str]

    # Sources and quality
    sources: List[ResearchSource]
    quality_score: float = Field(ge=0.0, le=1.0)
    confidence_metrics: Dict[str, float]
    coverage_assessment: Dict[str, float]

    # Synthesis metadata
    synthesis_metadata: SynthesisMetadata

    # Structured content for report generation
    content_hierarchy: Dict[str, Any]
```

## System Prompts

### Synthesis System Prompt

````python
SYNTHESIS_SYSTEM_PROMPT = """
# Role: Advanced Research Synthesis Expert

You are a Senior Research Synthesis Specialist with expertise in pattern recognition,
evidence evaluation, and systematic analysis across diverse information sources.

## Core Mission
Transform raw search results into actionable research insights through systematic
synthesis, advanced pattern recognition, and quality-assured analysis.

## Process Overview: Tree of Thoughts

The following tree represents the three-phase synthesis process. Each branch will be
executed systematically following the detailed instructions in the Execution Protocol below:

    Research Synthesis Process
    ├── Pattern Recognition (Phase 1)
    │   ├── Convergence Analysis (agreement across sources)
    │   ├── Divergence Mapping (contradictions/conflicts)
    │   └── Emergence Detection (new trends/signals)
    ├── Insight Extraction (Phase 2)
    │   ├── Primary Insights (directly address query)
    │   ├── Secondary Insights (related discoveries)
    │   └── Meta-Insights (patterns about patterns)
    └── Quality Verification (Phase 3)
        ├── Completeness Check
        ├── Coherence Validation
        └── Confidence Calibration

## Execution Protocol: Three-Phase Synthesis

### Phase 1: Pattern Recognition (Tree Branch 1 - Execute ALL sub-branches in parallel)

#### Task 1.1: Convergence Analysis
OBJECTIVE: Identify findings that support each other across sources
STEPS:
1. Group findings by semantic similarity (>70% overlap threshold)
2. Count supporting sources for each finding group
3. Calculate convergence_score = (supporting_sources / total_sources)
4. Classify convergence strength:
   - Strong (>0.8): Multiple independent sources agree
   - Moderate (0.5-0.8): Majority agreement with some variation
   - Weak (<0.5): Limited agreement, consider as preliminary

EXAMPLE:
- Input: 5 sources report "40-60% efficiency improvement with new caching"
- Analysis: High semantic overlap (numbers + "efficiency improvement")
- Output: convergence_score=1.0, strength="strong", confidence_boost=+0.2
- Synthesis: "Strong consensus on 40-60% efficiency improvement from caching implementation"

#### Task 1.2: Divergence Mapping
OBJECTIVE: Identify contradictory or conflicting information
STEPS:
1. Compare all finding pairs for contradiction indicators
2. Detection criteria:
   - Opposite directional words (increase/decrease, positive/negative)
   - Conflicting metrics on same measurement
   - Mutually exclusive claims
3. Classify contradiction type and severity:
   - Direct: Opposite claims about same fact (severity=high)
   - Partial: Different scope or conditions (severity=medium)
   - Contextual: True in different contexts (severity=low)
4. Generate resolution hypotheses

EXAMPLE:
- Finding A: "AI increases job opportunities"
- Finding B: "AI reduces employment"
- Analysis: Contextual contradiction - sector-specific effects
- Resolution: "AI impact on employment varies by sector - creating tech jobs while reducing manufacturing roles"
- Confidence adjustment: Medium (0.5) due to conflicting evidence

#### Task 1.3: Emergence Detection
OBJECTIVE: Spot new trends, patterns, or signals not explicitly stated
STEPS:
1. Scan for temporal markers: "recently", "emerging", "new", dates
2. Identify statistical outliers: unique findings from authoritative sources
3. Detect pattern breaks: sudden changes in metrics or consensus
4. Flag innovation indicators: novel methods, breakthrough claims
5. Output emergence_signals[] with confidence scores

EXAMPLE:
- Input: Single authoritative source provides novel framework not seen elsewhere
- Analysis: Unique insight from credible source, no corroboration yet
- Output: emergence_signal with confidence=0.6 (single source limitation)
- Synthesis: "Novel framework for understanding X (single source, requires validation)"

### Phase 2: Insight Extraction (Tree Branch 2 - Sequential processing by importance)

#### Task 2.1: Primary Insights (Critical findings only)
FILTER: importance_score >= 0.9
PROCESS:
1. Link each critical finding to original research questions
2. Merge convergent findings into unified statements
3. Preserve ALL numerical data and source attribution
4. Generate 3-5 bullet points maximum

TEMPLATE: "[INSIGHT] based on [N sources] showing [convergence_level] agreement"

#### Task 2.2: Secondary Insights (Important findings)
FILTER: importance_score 0.7-0.89
PROCESS:
1. Group by theme cluster
2. Extract patterns that contextualize primary insights
3. Identify surprising connections
4. Generate 3-5 supporting points

#### Task 2.3: Meta-Insights (Pattern analysis)
REQUIRES: Completed Tasks 1.1-1.3 and 2.1-2.2
PROCESS:
1. Analyze finding distribution across categories
2. Identify systematic gaps in research coverage
3. Assess methodology trends and biases
4. Evaluate temporal evolution of consensus

### Phase 3: Quality Verification (Tree Branch 3 - Sequential mandatory checks)

#### Check 3.1: Completeness Verification
□ All SearchQueryBatch queries addressed? (REQUIRED: 100%)
□ Critical findings preserved? (REQUIRED: 100%)
□ Source attribution complete? (REQUIRED: 100%)
□ Contradictions documented? (REQUIRED: 100%)
FAIL → Return to Phase 1 with gap list

#### Check 3.2: Coherence Validation
□ Logical flow from evidence to conclusions?
□ Confidence justified by evidence strength?
□ Temporal consistency maintained?
□ Causal claims properly qualified?
FAIL → Flag specific issues for correction

#### Check 3.3: Confidence Calibration
FORMULA: confidence = (source_quality * convergence_score * (1 - contradiction_penalty))
VERIFY:
□ Single-source claims marked as preliminary?
□ Contradictions reduce confidence appropriately?
□ Distribution reasonable (not all high/low)?
FAIL → Recalibrate using formula

## Information Hierarchy Framework with Scoring Examples

Apply this scoring to EVERY finding:

### Critical (0.9-1.0)
DEFINITION: Core facts, primary conclusions, unique insights that directly answer the research question
CRITERIA:
- Directly answers research question
- Unique insight not found elsewhere
- High-confidence breakthrough finding
- Safety/risk critical information

EXAMPLE:
- Finding: "GPT-4 achieves 86.4% on MMLU benchmark, surpassing human expert performance"
- Score: 0.95
- Rationale: Direct answer to AI performance query, specific metric, authoritative source

### Important (0.7-0.89)
DEFINITION: Supporting evidence, methodologies, key patterns that provide context
CRITERIA:
- Provides essential context
- Validates/challenges critical findings
- Methodology and approach details
- Strong supporting evidence

EXAMPLE:
- Finding: "The benchmark was conducted using zero-shot prompting across 57 subjects"
- Score: 0.75
- Rationale: Methodology detail that validates the critical finding

### Supplementary (0.4-0.69)
DEFINITION: Background information, elaborations, secondary examples
CRITERIA:
- Additional examples and cases
- Extended background information
- Alternative perspectives
- Historical context

### Contextual (0.0-0.39)
DEFINITION: Tangential information, general background
CRITERIA:
- General background
- Tangentially related information
- Common knowledge
- Redundant details

## Preservation Rules

### Must ALWAYS Preserve

✓ All numerical data and statistics with sources
✓ Contradictions and conflicting viewpoints
✓ Unique insights not found elsewhere
✓ Methodological limitations and caveats
✓ Confidence intervals and uncertainty measures
✓ Source attribution for all claims

### Safe to Consolidate

- Similar findings from multiple sources (note convergence)
- Redundant examples (keep most representative)
- Extended background information
- Detailed process descriptions (summarize key steps)

### Never Omit

✗ Contradictory evidence (even if minority view)
✗ Methodological flaws or limitations
✗ Recent updates that override older information
✗ Safety, ethical, or legal concerns
✗ Uncertainty or low confidence indicators

## Self-Verification Checklist

Before finalizing synthesis, verify:
□ Phase 1 (Pattern Recognition) - All three tasks completed?
□ Phase 2 (Insight Extraction) - Findings processed by importance?
□ Phase 3 (Quality Verification) - All checks passed?
□ All SearchQueryBatch queries addressed (100% required)?
□ Information hierarchy correctly applied to all findings?
□ Contradictions explicitly identified and analyzed?
□ Confidence scores justified by evidence and convergence?
□ Source attribution complete for all claims?
□ Executive summary captures essence without losing critical details?
□ Theme clusters logically organized with consensus levels?
□ Gaps and limitations clearly documented?
□ No critical information lost in synthesis?
□ Output follows exact structure requirements?

## Anti-Patterns to Avoid

✗ Cherry-picking evidence that supports a single narrative
✗ Hiding or minimizing contradictions
✗ Over-generalizing from limited sources
✗ Conflating correlation with causation
✗ Ignoring source credibility differences
✗ Creating false consensus where none exists
✗ Losing nuance through oversimplification
✗ Failing to acknowledge uncertainty

## Output Structure Requirements

Generate synthesis with these EXACT sections:

### 1. Executive Summary (2-3 paragraphs)
- Paragraph 1: Core findings and direct answer to research question
- Paragraph 2: Key patterns, convergences, and contradictions identified
- Paragraph 3: Overall confidence assessment and critical gaps

### 2. Hierarchical Findings

Format:
    CRITICAL FINDINGS (0.9-1.0):
    • [Finding 1] (confidence: X, sources: N, convergence: strong/moderate/weak)
    • [Finding 2] (confidence: X, sources: N, convergence: strong/moderate/weak)

    IMPORTANT FINDINGS (0.7-0.89):
    • [Finding 3] (confidence: X, sources: N)
    • [Finding 4] (confidence: X, sources: N)

    SUPPLEMENTARY FINDINGS (0.4-0.69):
    • [Brief listing of additional context]

    CONTEXTUAL INFORMATION (0.0-0.39):
    • [Optional, only if adds value]

### 3. Theme Clusters

Format:
    THEME: [Name]
    - Description: [One sentence explanation]
    - Findings: [Count] findings across [N] sources
    - Consensus: [strong/moderate/weak/conflicting]
    - Key Insight: [Single most important takeaway]
    - Confidence: [0.0-1.0]

### 4. Contradictions Analysis

Format:
    CONTRADICTION 1: [Type - direct/partial/contextual]
    - Finding A: [Description with source]
    - Finding B: [Description with source]
    - Resolution: [Proposed reconciliation]
    - Impact: [high/medium/low] on overall conclusions

### 5. Quality Metrics

Format:
    Synthesis Metrics:
    - Total Findings Analyzed: [N]
    - Convergence Rate: [X]% (findings with multi-source agreement)
    - Contradiction Rate: [Y]% (findings with conflicts)
    - Source Diversity: [Z] unique domains
    - Query Coverage: [%] of original questions addressed
    - Confidence Distribution:
      * High (>0.8): [N] findings ([%])
      * Medium (0.5-0.8): [N] findings ([%])
      * Low (<0.5): [N] findings ([%])
    - Overall Synthesis Confidence: [0.0-1.0] - [one sentence justification]

### 6. Gaps & Limitations

Format:
    INFORMATION GAPS:
    • [Unanswered aspects of research questions]
    • [Missing data or perspectives]

    METHODOLOGICAL LIMITATIONS:
    • [Source quality issues]
    • [Coverage limitations]
    • [Temporal constraints]

## Domain Adaptation Protocol

Detect domain from query content and apply appropriate focus:

### Technical/Scientific Research:
- Preserve ALL version numbers, specifications, and technical details
- Maintain precision in terminology (no paraphrasing technical terms)
- Emphasize reproducibility and methodology
- Track compatibility constraints and dependencies
- Weight peer-reviewed and official documentation higher

### Business/Market Research:
- Extract ROI, cost implications, and business metrics
- Highlight competitive advantages and market positioning
- Focus on actionable recommendations and strategic insights
- Emphasize timing and market readiness
- Consider source recency more heavily

### Academic/Theoretical Research:
- Maintain citation standards and attribution
- Track theoretical frameworks and schools of thought
- Document methodology details and research design
- Preserve academic debates and opposing viewpoints
- Note publication venues and impact factors

### Emerging Technology Research:
- Flag uncertainty and speculation clearly
- Distinguish proven vs. experimental/theoretical
- Track hype indicators vs. actual deployments
- Monitor patent filings and investment signals
- Note early adopters and pilot programs

Remember: Your synthesis will be used for decision-making. Accuracy, completeness, and
transparency are paramount. When uncertain, explicitly state limitations rather than
creating false confidence.
"""

### Dynamic Context Injection

```python
@self.agent.instructions
async def add_enhanced_synthesis_context(ctx: RunContext[ResearchExecutorDependencies]) -> str:
    """Inject enhanced context for synthesis."""

    # Determine query complexity
    complexity = self._assess_query_complexity(ctx.deps.search_queries)

    # Select appropriate few-shot examples
    examples = self._get_relevant_examples(ctx.deps.original_query)

    # Add context-specific rules
    if complexity == "technical":
        additional_rules = """
        ## Technical Synthesis Rules
        - Preserve all technical specifications
        - Maintain precision in terminology
        - Note version dependencies
        - Highlight compatibility issues
        """
    elif complexity == "business":
        additional_rules = """
        ## Business Synthesis Rules
        - Focus on ROI and business impact
        - Highlight actionable recommendations
        - Note market conditions
        - Emphasize competitive advantages
        """
    else:
        additional_rules = ""

    return f"""
    Query Type: {complexity}
    Search Results Count: {len(ctx.deps.search_results.all_results)}
    Priority Distribution: {self._get_priority_distribution(ctx.deps.search_queries)}

    {additional_rules}

    Additional Examples:
    {examples}
    """
````

## Module Structure

```
src/
├── agents/
│   ├── research_executor.py          # Main agent implementation
│   └── __init__.py
├── models/
│   ├── research_executor.py          # All research models
│   └── __init__.py
├── services/
│   ├── __init__.py
│   ├── synthesis_engine.py           # Core synthesis logic
│   ├── contradiction_detector.py     # Contradiction detection
│   ├── pattern_recognizer.py         # Pattern analysis
│   └── confidence_analyzer.py        # Confidence analysis
└── utils/
    ├── __init__.py
    └── research_helpers.py           # Utility functions
```

## Implementation Details

### 1. Research Executor Agent

```python
# src/agents/research_executor.py
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from ..models.research_executor import (
    ResearchResults, HierarchicalFinding, ThemeCluster,
    PatternAnalysis, Contradiction, ExecutiveSummary,
    SynthesisMetadata, ConfidenceLevel, ImportanceLevel
)
from ..services.synthesis_engine import SynthesisEngine
from ..services.contradiction_detector import ContradictionDetector
from ..services.pattern_recognizer import PatternRecognizer
from ..services.confidence_analyzer import ConfidenceAnalyzer

class ResearchExecutorDependencies(BaseModel):
    """Dependencies for research executor agent"""
    search_results: List[SearchResult]
    query_batch: SearchQueryBatch
    synthesis_approach: str = Field(default="technical")
    max_findings: int = Field(default=50)

class ResearchExecutorAgent:
    """Agent for executing research and synthesis"""

    def __init__(self, model_name: str = "openai:gpt-4o"):
        self.agent = Agent(
            model_name,
            deps_type=ResearchExecutorDependencies,
            result_type=ResearchResults,
            system_prompt=self._get_system_prompt()
        )

        # Services
        self.synthesis_engine = SynthesisEngine()
        self.contradiction_detector = ContradictionDetector()
        self.pattern_recognizer = PatternRecognizer()
        self.confidence_analyzer = ConfidenceAnalyzer()

        # Register tools
        self._register_tools()

    def _get_system_prompt(self) -> str:
        return """You are an advanced research synthesis agent with expertise in:
        1. Extracting and organizing findings from multiple sources
        2. Identifying patterns, themes, and relationships
        3. Detecting and analyzing contradictions
        4. Building hierarchical knowledge structures
        5. Assessing confidence and source quality
        6. Generating actionable insights and recommendations

        Apply systematic analysis with critical thinking and evidence-based reasoning."""

    def _register_tools(self):
        """Register agent tools for synthesis operations"""

        @self.agent.tool
        async def extract_hierarchical_findings(
            ctx: RunContext[ResearchExecutorDependencies],
            search_results: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            """Extract findings with hierarchical classification"""
            findings = []

            # Parallel extraction
            tasks = [
                self._extract_and_classify(result)
                for result in search_results
            ]
            extracted = await asyncio.gather(*tasks, return_exceptions=True)

            for finding_list in extracted:
                if isinstance(finding_list, Exception):
                    logfire.error(f"Extraction failed: {finding_list}")
                    continue
                findings.extend(finding_list)

            # Apply importance hierarchy
            classified_findings = self._classify_by_importance(findings)

            return classified_findings

        @self.agent.tool
        async def identify_theme_clusters(
            ctx: RunContext[ResearchExecutorDependencies],
            findings: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            """Group findings into theme clusters"""

            clusters = await self.synthesis_engine.cluster_by_themes(
                findings,
                min_cluster_size=2,
                similarity_threshold=0.7
            )

            # Analyze consensus within clusters
            for cluster in clusters:
                cluster['consensus_level'] = self._analyze_consensus(
                    cluster['findings']
                )
                cluster['key_insights'] = self._extract_cluster_insights(
                    cluster['findings']
                )

            return clusters

        @self.agent.tool
        async def detect_contradictions(
            ctx: RunContext[ResearchExecutorDependencies],
            findings: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            """Detect contradictions between findings"""

            contradictions = await self.contradiction_detector.detect_all(
                findings,
                check_types=["direct", "partial", "contextual", "methodological"]
            )

            # Add severity assessment
            for cont in contradictions:
                cont['severity'] = self._assess_contradiction_severity(cont)
                cont['resolution_suggestion'] = self._suggest_resolution(cont)

            return contradictions

        @self.agent.tool
        async def analyze_patterns(
            ctx: RunContext[ResearchExecutorDependencies],
            findings: List[Dict[str, Any]],
            clusters: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            """Analyze patterns across findings and clusters"""

            patterns = await self.pattern_recognizer.recognize_patterns(
                findings=findings,
                clusters=clusters,
                pattern_types=["convergence", "divergence", "emergence", "temporal"]
            )

            # Add implications
            for pattern in patterns:
                pattern['implications'] = self._derive_implications(pattern)

            return patterns

        @self.agent.tool
        async def generate_executive_summary(
            ctx: RunContext[ResearchExecutorDependencies],
            findings: List[Dict[str, Any]],
            patterns: List[Dict[str, Any]],
            contradictions: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Generate executive summary"""

            critical_findings = [
                f for f in findings
                if f.get('importance_level') == 'critical'
            ]

            return {
                'key_findings': self._summarize_findings(critical_findings[:5]),
                'overall_confidence': self._calculate_overall_confidence(findings),
                'critical_gaps': self._identify_gaps(findings, ctx.deps.query_batch),
                'immediate_insights': self._extract_immediate_insights(patterns),
                'strategic_implications': self._derive_strategic_implications(
                    patterns, contradictions
                )
            }

        @self.agent.tool
        async def create_content_hierarchy(
            ctx: RunContext[ResearchExecutorDependencies],
            clusters: List[Dict[str, Any]],
            findings: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Create hierarchical content structure for reports"""

            hierarchy = {}

            # Build main sections from theme clusters
            for cluster in clusters:
                section_name = cluster['theme_name']
                hierarchy[section_name] = {
                    'description': cluster['description'],
                    'findings': [
                        f for f in findings
                        if f['id'] in cluster['finding_ids']
                    ],
                    'insights': cluster['key_insights'],
                    'confidence': cluster['confidence']
                }

            # Add metadata section
            hierarchy['_metadata'] = {
                'total_sections': len(clusters),
                'total_findings': len(findings),
                'synthesis_approach': ctx.deps.synthesis_approach
            }

            return hierarchy

    async def _extract_and_classify(
        self,
        result: SearchResult
    ) -> List[HierarchicalFinding]:
        """Extract and classify findings from a search result"""
        findings = []

        # Extract findings from content
        if result.content:
            content_findings = await self._extract_from_text(
                result.content,
                source=result
            )

            # Classify each finding
            for finding in content_findings:
                finding.importance_level = self._determine_importance(
                    finding, result.relevance_score
                )
                finding.confidence_category = self._score_to_level(
                    finding.confidence_level
                )
                findings.append(finding)

        return findings

    async def _extract_from_text(
        self,
        text: str,
        source: SearchResult
    ) -> List[HierarchicalFinding]:
        """Extract findings from text with full metadata"""
        findings = []

        # Use NLP or LLM for extraction (simplified here)
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]

        for i, sentence in enumerate(sentences):
            finding = HierarchicalFinding(
                id=f"{source.id}_finding_{i}",
                finding=sentence,
                supporting_evidence=[source.url] if source.url else [],
                confidence_level=source.relevance_score * 0.9,
                confidence_category=ConfidenceLevel.MEDIUM,
                source=self._create_research_source(source),
                importance_level=ImportanceLevel.SUPPLEMENTARY,
                theme_cluster="unassigned",
                sub_findings=[]
            )
            findings.append(finding)

        return findings

    def _determine_importance(
        self,
        finding: HierarchicalFinding,
        source_relevance: float
    ) -> ImportanceLevel:
        """Determine importance level of a finding"""

        # Combine multiple factors
        score = finding.confidence_level * 0.5 + source_relevance * 0.5

        if score >= 0.9:
            return ImportanceLevel.CRITICAL
        elif score >= 0.7:
            return ImportanceLevel.IMPORTANT
        elif score >= 0.4:
            return ImportanceLevel.SUPPLEMENTARY
        else:
            return ImportanceLevel.CONTEXTUAL

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level"""
        if score >= 0.9:
            return ConfidenceLevel.HIGH
        elif score >= 0.7:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN

    def _classify_by_importance(
        self,
        findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Classify findings by importance hierarchy"""

        # Sort by importance and confidence
        sorted_findings = sorted(
            findings,
            key=lambda f: (
                self._importance_to_score(f['importance_level']),
                f['confidence_level']
            ),
            reverse=True
        )

        return sorted_findings

    def _importance_to_score(self, level: str) -> float:
        """Convert importance level to numeric score"""
        scores = {
            'critical': 1.0,
            'important': 0.75,
            'supplementary': 0.5,
            'contextual': 0.25
        }
        return scores.get(level, 0.5)

    async def execute_research(
        self,
        search_results: List[SearchResult],
        query_batch: SearchQueryBatch,
        **kwargs
    ) -> ResearchResults:
        """Main execution method"""

        deps = ResearchExecutorDependencies(
            search_results=search_results,
            query_batch=query_batch,
            **kwargs
        )

        try:
            # Run agent with all tools
            result = await self.agent.run(
                """Execute comprehensive research synthesis:
                1. Extract and classify all findings hierarchically
                2. Identify theme clusters with consensus analysis
                3. Detect and analyze contradictions
                4. Recognize patterns across findings
                5. Generate executive summary
                6. Create content hierarchy for reporting

                Focus on actionable insights and evidence-based conclusions.""",
                deps=deps
            )

            return result

        except Exception as e:
            logfire.error(f"Research execution failed: {e}")
            raise
```

### 2. Synthesis Engine Service

```python
# src/services/synthesis_engine.py
from typing import List, Dict, Any, Optional, Set
import asyncio
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import logfire

from ..models.research_executor import (
    HierarchicalFinding, ThemeCluster, ImportanceLevel
)

class SynthesisEngine:
    """Advanced synthesis engine with clustering and analysis"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.clustering_cache = {}

    async def cluster_by_themes(
        self,
        findings: List[Dict[str, Any]],
        min_cluster_size: int = 2,
        similarity_threshold: float = 0.7
    ) -> List[ThemeCluster]:
        """Cluster findings by themes using text similarity"""

        if len(findings) < min_cluster_size:
            return self._create_single_cluster(findings)

        # Extract text for clustering
        texts = [f['finding'] for f in findings]

        # Vectorize findings
        try:
            vectors = self.vectorizer.fit_transform(texts)
        except Exception as e:
            logfire.error(f"Vectorization failed: {e}")
            return self._create_single_cluster(findings)

        # Determine optimal number of clusters
        n_clusters = min(len(findings) // min_cluster_size, 10)
        n_clusters = max(2, n_clusters)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(vectors)

        # Build theme clusters
        clusters = []
        for i in range(n_clusters):
            cluster_findings = [
                findings[j] for j, label in enumerate(cluster_labels)
                if label == i
            ]

            if len(cluster_findings) >= min_cluster_size:
                theme_cluster = self._create_theme_cluster(
                    cluster_findings,
                    cluster_id=i
                )
                clusters.append(theme_cluster)

        return clusters

    def _create_theme_cluster(
        self,
        findings: List[Dict[str, Any]],
        cluster_id: int
    ) -> ThemeCluster:
        """Create a theme cluster from findings"""

        # Extract common terms for theme name
        theme_name = self._extract_theme_name(findings)

        # Calculate cluster confidence
        confidence = np.mean([f['confidence_level'] for f in findings])

        # Determine consensus level
        consensus = self._analyze_consensus_level(findings)

        # Extract key insights
        insights = self._extract_cluster_insights(findings)

        return ThemeCluster(
            theme_name=theme_name,
            description=f"Cluster of {len(findings)} related findings",
            finding_ids=[f['id'] for f in findings],
            confidence=confidence,
            consensus_level=consensus,
            key_insights=insights
        )

    def _extract_theme_name(self, findings: List[Dict[str, Any]]) -> str:
        """Extract theme name from common terms in findings"""

        # Simple approach: find most common meaningful words
        all_words = []
        for finding in findings:
            words = finding['finding'].lower().split()
            # Filter out common words
            meaningful = [
                w for w in words
                if len(w) > 4 and w not in {'these', 'those', 'which', 'where', 'there'}
            ]
            all_words.extend(meaningful)

        if not all_words:
            return f"Theme_{id(findings) % 1000}"

        # Find most common word
        from collections import Counter
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(3)

        return "_".join([word for word, _ in top_words])

    def _analyze_consensus_level(
        self,
        findings: List[Dict[str, Any]]
    ) -> str:
        """Analyze consensus level within findings"""

        # Check for contradictions
        contradiction_count = sum(
            1 for f in findings
            if f.get('contradicts') and len(f['contradicts']) > 0
        )

        ratio = contradiction_count / len(findings)

        if ratio > 0.3:
            return "conflicting"
        elif ratio > 0.1:
            return "weak"
        elif ratio > 0.05:
            return "moderate"
        else:
            return "strong"

    def _extract_cluster_insights(
        self,
        findings: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract key insights from a cluster of findings"""

        insights = []

        # Get critical findings
        critical = [
            f for f in findings
            if f.get('importance_level') == 'critical'
        ]

        if critical:
            insights.append(
                f"Contains {len(critical)} critical findings"
            )

        # Check confidence distribution
        high_conf = sum(
            1 for f in findings
            if f.get('confidence_category') == 'high'
        )

        if high_conf > len(findings) * 0.5:
            insights.append("High confidence cluster")

        # Add top finding as insight
        if findings:
            top_finding = findings[0]['finding'][:100]
            insights.append(f"Key: {top_finding}...")

        return insights[:5]  # Limit to 5 insights

    def _create_single_cluster(
        self,
        findings: List[Dict[str, Any]]
    ) -> List[ThemeCluster]:
        """Create a single cluster when clustering isn't possible"""

        if not findings:
            return []

        return [ThemeCluster(
            theme_name="primary",
            description="All findings in single theme",
            finding_ids=[f['id'] for f in findings],
            confidence=np.mean([f['confidence_level'] for f in findings]),
            consensus_level="moderate",
            key_insights=self._extract_cluster_insights(findings)
        )]
```

### 3. Contradiction Detector Service

```python
# src/services/contradiction_detector.py
from typing import List, Dict, Any, Tuple
import asyncio
from difflib import SequenceMatcher
import logfire

from ..models.research_executor import Contradiction

class ContradictionDetector:
    """Service for detecting contradictions between findings"""

    def __init__(self):
        self.contradiction_keywords = {
            'direct': [
                ('increase', 'decrease'), ('positive', 'negative'),
                ('success', 'failure'), ('improve', 'worsen'),
                ('grow', 'shrink'), ('expand', 'contract')
            ],
            'partial': [
                ('sometimes', 'always'), ('partially', 'completely'),
                ('may', 'will'), ('could', 'must')
            ]
        }

    async def detect_all(
        self,
        findings: List[Dict[str, Any]],
        check_types: List[str] = None
    ) -> List[Contradiction]:
        """Detect all contradictions in findings"""

        if check_types is None:
            check_types = ["direct", "partial", "contextual", "methodological"]

        contradictions = []

        # Check each pair of findings
        for i in range(len(findings)):
            for j in range(i + 1, len(findings)):
                contradiction = await self._check_contradiction(
                    findings[i],
                    findings[j],
                    check_types
                )

                if contradiction:
                    contradictions.append(contradiction)

        return contradictions

    async def _check_contradiction(
        self,
        finding_a: Dict[str, Any],
        finding_b: Dict[str, Any],
        check_types: List[str]
    ) -> Optional[Contradiction]:
        """Check if two findings contradict each other"""

        text_a = finding_a['finding'].lower()
        text_b = finding_b['finding'].lower()

        # Check different types of contradictions
        for check_type in check_types:
            if check_type == "direct":
                if self._is_direct_contradiction(text_a, text_b):
                    return Contradiction(
                        finding_a_id=finding_a['id'],
                        finding_b_id=finding_b['id'],
                        contradiction_type="direct",
                        explanation=f"Direct opposing claims detected",
                        severity="high"
                    )

            elif check_type == "partial":
                if self._is_partial_contradiction(text_a, text_b):
                    return Contradiction(
                        finding_a_id=finding_a['id'],
                        finding_b_id=finding_b['id'],
                        contradiction_type="partial",
                        explanation=f"Partial disagreement in scope or degree",
                        severity="medium"
                    )

            elif check_type == "contextual":
                if self._is_contextual_contradiction(text_a, text_b):
                    return Contradiction(
                        finding_a_id=finding_a['id'],
                        finding_b_id=finding_b['id'],
                        contradiction_type="contextual",
                        explanation=f"Contradictory in specific contexts",
                        severity="low"
                    )

            elif check_type == "methodological":
                if self._is_methodological_contradiction(finding_a, finding_b):
                    return Contradiction(
                        finding_a_id=finding_a['id'],
                        finding_b_id=finding_b['id'],
                        contradiction_type="methodological",
                        explanation=f"Different methodologies lead to different conclusions",
                        severity="medium"
                    )

        return None

    def _is_direct_contradiction(self, text_a: str, text_b: str) -> bool:
        """Check for direct contradictions using keyword pairs"""

        for word_a, word_b in self.contradiction_keywords['direct']:
            if (word_a in text_a and word_b in text_b) or \
               (word_b in text_a and word_a in text_b):
                # Check if talking about same subject
                if self._similar_subject(text_a, text_b):
                    return True

        return False

    def _is_partial_contradiction(self, text_a: str, text_b: str) -> bool:
        """Check for partial contradictions in degree or scope"""

        for word_a, word_b in self.contradiction_keywords['partial']:
            if (word_a in text_a and word_b in text_b) or \
               (word_b in text_a and word_a in text_b):
                return True

        return False

    def _is_contextual_contradiction(self, text_a: str, text_b: str) -> bool:
        """Check for contextual contradictions"""

        # Simple heuristic: high similarity but opposite sentiment
        similarity = SequenceMatcher(None, text_a, text_b).ratio()

        if similarity > 0.6:  # High similarity
            # Check for negation words
            negations = ['not', 'no', 'never', 'neither', 'none', "don't", "doesn't"]
            has_negation_a = any(neg in text_a for neg in negations)
            has_negation_b = any(neg in text_b for neg in negations)

            # XOR - one has negation, other doesn't
            if has_negation_a != has_negation_b:
                return True

        return False

    def _is_methodological_contradiction(
        self,
        finding_a: Dict[str, Any],
        finding_b: Dict[str, Any]
    ) -> bool:
        """Check for methodological contradictions"""

        # Check if sources have different quality tiers
        source_a = finding_a.get('source')
        source_b = finding_b.get('source')

        if source_a and source_b:
            tier_diff = abs(
                source_a.get('credibility_tier', 3) -
                source_b.get('credibility_tier', 3)
            )

            # Different methodology if tier difference > 1
            if tier_diff > 1:
                # Check if conclusions differ
                text_a = finding_a['finding'].lower()
                text_b = finding_b['finding'].lower()

                if self._similar_subject(text_a, text_b):
                    similarity = SequenceMatcher(None, text_a, text_b).ratio()
                    if similarity < 0.5:  # Different conclusions
                        return True

        return False

    def _similar_subject(self, text_a: str, text_b: str) -> bool:
        """Check if two texts discuss similar subjects"""

        # Extract main nouns/subjects (simplified)
        words_a = set(text_a.split())
        words_b = set(text_b.split())

        # Find common meaningful words
        common = words_a & words_b
        meaningful_common = [
            w for w in common
            if len(w) > 4 and w not in {'these', 'those', 'which', 'where'}
        ]

        # If significant overlap, likely same subject
        return len(meaningful_common) >= 2
```

## Testing Strategy

### Unit Tests

```python
# tests/unit/agents/test_research_executor.py
import pytest
from unittest.mock import Mock, patch
import asyncio
from datetime import datetime

from agents.research_executor import ResearchExecutorAgent
from models.research_executor import (
    ResearchResults, HierarchicalFinding, ThemeCluster,
    Contradiction, ImportanceLevel, ConfidenceLevel
)

@pytest.fixture
def research_executor():
    """Fixture for research executor agent"""
    return ResearchExecutorAgent(model_name="openai:gpt-4o")

@pytest.fixture
def mock_search_results():
    """Fixture for mock search results"""
    return [
        SearchResult(
            id="result1",
            url="https://example.com/1",
            title="Research Paper 1",
            content="AI improves productivity by 40%. Studies show significant gains.",
            relevance_score=0.9
        ),
        SearchResult(
            id="result2",
            url="https://example.com/2",
            title="Industry Report",
            content="AI decreases productivity in creative tasks. Mixed results observed.",
            relevance_score=0.8
        )
    ]

@pytest.mark.asyncio
async def test_execute_research_full_pipeline(research_executor, mock_search_results):
    """Test full research execution pipeline"""

    query_batch = SearchQueryBatch(
        queries=["AI productivity impact"],
        context="technology research"
    )

    # Execute research
    results = await research_executor.execute_research(
        search_results=mock_search_results,
        query_batch=query_batch
    )

    # Validate structure
    assert isinstance(results, ResearchResults)
    assert len(results.findings) > 0
    assert all(isinstance(f, HierarchicalFinding) for f in results.findings)

    # Check synthesis outputs
    assert results.executive_summary is not None
    assert len(results.theme_clusters) > 0
    assert isinstance(results.theme_clusters[0], ThemeCluster)

    # Verify contradictions detected
    assert len(results.contradictions) > 0
    assert results.contradictions[0].contradiction_type in [
        "direct", "partial", "contextual", "methodological"
    ]

    # Check hierarchical classification
    importance_levels = {f.importance_level for f in results.findings}
    assert len(importance_levels) > 1  # Multiple levels present

@pytest.mark.asyncio
async def test_finding_extraction_and_classification(research_executor):
    """Test finding extraction with classification"""

    search_result = SearchResult(
        id="test1",
        url="https://test.com",
        title="Test",
        content="Critical finding about AI. Important evidence presented. Additional context provided.",
        relevance_score=0.95
    )

    findings = await research_executor._extract_and_classify(search_result)

    assert len(findings) > 0
    assert all(hasattr(f, 'importance_level') for f in findings)
    assert all(hasattr(f, 'confidence_category') for f in findings)
    assert all(f.source is not None for f in findings)

@pytest.mark.asyncio
async def test_theme_clustering(research_executor):
    """Test theme clustering functionality"""

    findings = [
        {
            'id': f'f{i}',
            'finding': f'Finding about {"AI" if i % 2 == 0 else "ML"} topic {i}',
            'confidence_level': 0.8,
            'importance_level': 'important'
        }
        for i in range(10)
    ]

    clusters = await research_executor.synthesis_engine.cluster_by_themes(
        findings,
        min_cluster_size=2
    )

    assert len(clusters) > 0
    assert all(isinstance(c, ThemeCluster) for c in clusters)
    assert all(len(c.finding_ids) >= 2 for c in clusters)
    assert all(c.consensus_level in ["strong", "moderate", "weak", "conflicting"] for c in clusters)

@pytest.mark.asyncio
async def test_contradiction_detection(research_executor):
    """Test contradiction detection"""

    findings = [
        {
            'id': 'f1',
            'finding': 'AI increases productivity significantly',
            'confidence_level': 0.9
        },
        {
            'id': 'f2',
            'finding': 'AI decreases productivity in many cases',
            'confidence_level': 0.8
        }
    ]

    contradictions = await research_executor.contradiction_detector.detect_all(findings)

    assert len(contradictions) > 0
    assert contradictions[0].contradiction_type == "direct"
    assert contradictions[0].severity in ["high", "medium", "low"]
```

## Enhanced Synthesis Tools Implementation

```python
@synthesis_agent.tool
async def score_information_hierarchy(
    findings: list[ResearchFinding]
) -> dict[str, list[ResearchFinding]]:
    """Score and categorize findings by importance."""
    return {
        "critical": [f for f in findings if score(f) >= 0.9],
        "important": [f for f in findings if 0.7 <= score(f) < 0.9],
        "supplementary": [f for f in findings if 0.4 <= score(f) < 0.7],
        "contextual": [f for f in findings if score(f) < 0.4]
    }

@synthesis_agent.tool
async def verify_synthesis_quality(
    synthesis: SynthesisResults
) -> VerificationReport:
    """Self-verification protocol for synthesis quality."""
    return VerificationReport(
        all_critical_preserved=True,
        logical_flow_maintained=True,
        no_ambiguity_introduced=True,
        confidence_justified=True,
        gaps_identified=True
    )

@synthesis_agent.tool
async def detect_contradictions(
    findings: list[ResearchFinding]
) -> list[Contradiction]:
    """Programmatically detect contradictions in findings."""
    # Advanced contradiction detection logic
    contradictions = []
    for i, finding_a in enumerate(findings):
        for finding_b in findings[i+1:]:
            if is_contradictory(finding_a, finding_b):
                contradictions.append(Contradiction(
                    finding_a_id=finding_a.id,
                    finding_b_id=finding_b.id,
                    type=determine_contradiction_type(finding_a, finding_b),
                    explanation=explain_contradiction(finding_a, finding_b)
                ))
    return contradictions

@synthesis_agent.tool
async def generate_synthesis_metrics(
    synthesis: SynthesisResults
) -> SynthesisMetrics:
    """Generate comprehensive metrics for synthesis quality."""
    return SynthesisMetrics(
        convergence_score=calculate_convergence(synthesis),
        confidence_distribution=analyze_confidence_distribution(synthesis),
        source_diversity=calculate_source_diversity(synthesis),
        pattern_strength=assess_pattern_strength(synthesis)
    )
```

## Implementation Plan

1. **Day 1-2**: Implement enhanced synthesis prompt

   - Add information hierarchy framework
   - Include few-shot examples
   - Add self-verification checklist

2. **Day 3-4**: Create SynthesisTools class

   - Implement hierarchy scoring
   - Add basic contradiction detection
   - Create metrics generation

3. **Day 5**: Integration testing
   - Test enhanced prompt with workflow
   - Verify all components integrate

### Phase 2: Tool Integration (Week 2)

1. **Day 1-2**: Implement synthesis tools

   - Score information hierarchy tool
   - Verify synthesis quality tool
   - Detect contradictions tool

2. **Day 3-4**: Enhanced metrics

   - Implement convergence analysis
   - Add confidence calibration
   - Create synthesis metrics

3. **Day 5**: Performance optimization
   - Add caching for repeated patterns
   - Implement parallel tool execution

### Phase 3: Quality Assurance (Week 3)

1. **Day 1-2**: Testing framework

   - Create comprehensive test suite
   - Implement quality benchmarks

2. **Day 3-4**: A/B testing

   - Compare enhanced vs baseline
   - Measure improvements

3. **Day 5**: Documentation and rollout
   - Update documentation
   - Create deployment guide

## Configuration

```python
# src/config/research_executor.py
from pydantic import BaseModel, Field

class ResearchExecutorConfig(BaseModel):
    """Configuration for research executor"""

    # Model settings
    model_name: str = Field(default="openai:gpt-4o")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    synthesis_model: str = Field(default="openai:gpt-4o")  # or gpt-5 when available
    synthesis_max_retries: int = Field(default=2)

    # Search execution
    primary_search_provider: str = Field(default="tavily")
    fallback_providers: list[str] = Field(default_factory=lambda: ["mock"])
    max_retries_per_query: int = Field(default=3)
    retry_delay_seconds: int = Field(default=2)

    # Processing settings
    max_findings: int = Field(default=50, ge=10, le=200)
    min_cluster_size: int = Field(default=2, ge=2, le=10)
    similarity_threshold: float = Field(default=0.7, ge=0.5, le=0.95)

    # Information hierarchy thresholds
    critical_threshold: float = Field(default=0.9)
    important_threshold: float = Field(default=0.7)
    supplementary_threshold: float = Field(default=0.4)

    # Feature flags
    enable_hierarchy_scoring: bool = Field(default=True)
    enable_contradiction_detection: bool = Field(default=True)
    enable_pattern_recognition: bool = Field(default=True)
    enable_theme_clustering: bool = Field(default=True)
    enable_quality_verification: bool = Field(default=True)

    # Quality thresholds
    min_execution_rate: float = Field(default=0.9)
    min_quality_score: float = Field(default=0.7)
    min_convergence_score: float = Field(default=0.6)
    min_confidence_for_critical: float = Field(default=0.9, ge=0.7, le=1.0)
    min_source_quality: str = Field(default="moderate")

    # Performance optimization
    enable_synthesis_cache: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600)
    parallel_tool_execution: bool = Field(default=True)

    # Cost management
    max_cost_per_research: float = Field(default=1.00)
    cost_per_search: float = Field(default=0.02)
    cost_per_synthesis: float = Field(default=0.10)
```

## Summary

This implementation plan provides a phased approach to building a comprehensive Research Executor Agent:
