# Clarification System Improvement Implementation Plan

## Executive Summary

The current clarification agent is failing to identify broad, ambiguous queries like "What is quantum computing?" that require clarification. Analysis of LangGraph's proven implementation reveals critical gaps in our approach. This plan outlines comprehensive improvements to create a robust clarification system.

## Problem Analysis

### Current Issues

1. **Over-permissive clarification logic**: Simple questions that should trigger clarification are being accepted
2. **Weak scope assessment**: No systematic approach to identify missing essential dimensions
3. **Missing transformation step**: No process to convert conversation into detailed research questions
4. **Insufficient guidance**: Prompt lacks specific criteria for information gathering

### Root Causes

- **Prompt engineering deficiency**: Current system prompt too generic and lenient
- **Missing architectural component**: No research question transformation after clarification
- **Inadequate validation logic**: Confidence thresholds and assessment criteria too weak
- **Insufficient output guidance**: Current prompt lacks explicit conditional instructions for response fields

## LangGraph Analysis Insights

### Technical Implementation Comparison

**Important Finding**: Both LangGraph and our implementation use Pydantic BaseModel for structured output:
- **LangGraph**: Uses `ClarifyWithUser(BaseModel)` at `/src/open_deep_research/state.py:30-41`
- **LangGraph**: Applies via `.with_structured_output(ClarifyWithUser)` at `/src/open_deep_research/deep_researcher.py:91`
- **Our Implementation**: Uses `ClarifyWithUser(BaseModel)` at `/src/open_deep_research_with_pydantic_ai/agents/clarification.py:18-32`

Both approaches have the same structural validation advantages. The key differences lie in prompt engineering and workflow design.

### Technical Comparison Table

| Aspect | LangGraph | Our Current Implementation | Assessment |
|--------|-----------|---------------------------|------------|
| **Structure** | Pydantic BaseModel | Pydantic BaseModel | ✅ Same (both excellent) |
| **Type Safety** | Full type checking | Full type checking | ✅ Same |
| **Validation** | Automatic via Pydantic | Automatic via Pydantic | ✅ Same |
| **Prompt Clarity** | Explicit conditionals | Generic instructions | ❌ Ours needs improvement |
| **Output Guidance** | Clear if/then rules | Vague requirements | ❌ Ours needs improvement |
| **Question Transform** | Separate component | Missing | ❌ We need to add this |
| **Scope Assessment** | Implicit in prompt | Weak criteria | ❌ Ours needs strengthening |

### Key Differentiators in LangGraph's Approach

1. **Explicit Conditional Output Instructions**:
   ```
   From /src/open_deep_research/prompts.py:21-34:

   If you need to ask a clarifying question, return:
   "need_clarification": true,
   "question": "<your clarifying question>",
   "verification": ""

   If you do not need to ask a clarifying question, return:
   "need_clarification": false,
   "question": "",
   "verification": "<acknowledgement message>"
   ```
   This explicit conditional logic tells the LLM exactly what to output in each scenario.

2. **Structured Information Gathering Guidelines**:
   ```
   - Be concise while gathering all necessary information
   - Make sure to gather all the information needed to carry out the research task
   - Use bullet points or numbered lists if appropriate for clarity
   ```

3. **Research Question Transformation Component**:
   ```python
   # From /src/open_deep_research/prompts.py:44-77
   # Separate step that maximizes specificity:
   # 1. Include all known user preferences
   # 2. Fill in unstated but necessary dimensions as open-ended
   # 3. Avoid unwarranted assumptions
   # 4. Use first person perspective
   # 5. Specify source prioritization
   ```

4. **Clear Verification Requirements**: Specific guidance on acknowledgment messages when no clarification is needed

## Technical Implementation Plan

### Phase 1: Enhanced Clarification System Prompt

**File**: `src/open_deep_research_with_pydantic_ai/agents/clarification.py`

#### 1.1 Enhance System Prompt While Keeping Pydantic Structure

**Current prompt issues**:
- Too generic: "assess whether you need to ask a clarifying question"
- No specific guidance on scope assessment
- Weak information gathering criteria
- Missing explicit conditional output instructions

**Important**: We keep our Pydantic `ClarifyWithUser(BaseModel)` structure as it provides superior type safety and validation. We only need to improve the prompt engineering.

**New prompt structure** (adapted from LangGraph's explicit conditional approach):
```python
def _get_default_system_prompt(self) -> str:
    return """You are a research clarification assistant. Today's date is {date}.

Your role is to assess whether you need to ask a clarifying question, or if the user
has already provided enough information for comprehensive research.

IMPORTANT: If you can see in the messages history that you have already asked a
clarifying question, you almost always do not need to ask another one. Only ask
another question if ABSOLUTELY NECESSARY.

CRITICAL ASSESSMENT CRITERIA:
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

EXAMPLES OF QUERIES REQUIRING CLARIFICATION:
- "What is [broad topic]?" → Ask about audience, purpose, specific focus
- "Compare [general category]" → Ask which specific items, criteria, timeframe
- "Research [technology]" → Ask about technical depth, applications, audience
- "Analyze [industry]" → Ask about specific aspects, timeframe, geographic scope

INFORMATION GATHERING GUIDELINES:
- Be concise while gathering all necessary information
- Use bullet points or numbered lists for clarity in questions
- Don't ask for unnecessary information already provided
- Focus on gathering information needed for comprehensive research

OUTPUT REQUIREMENTS (Explicit Conditional Instructions):

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

IMPORTANT: Only populate the field relevant to your decision. Leave the other field as an empty string."""
```

#### 1.2 Strengthen Assessment Logic

**Add scope breadth detection**:
```python
async def _assess_query_breadth(self, query: str, conversation: list[str]) -> dict:
    """Assess if query is too broad or missing essential context."""

    broad_indicators = [
        "what is", "how does", "explain", "overview of",
        "research", "analyze", "compare", "study"
    ]

    missing_context_flags = {
        "audience_level": not any(word in " ".join(conversation).lower()
                               for word in ["beginner", "expert", "academic", "business"]),
        "purpose": not any(word in " ".join(conversation).lower()
                          for word in ["for", "because", "need", "purpose", "use"]),
        "scope": len(query.split()) < 8,  # Very short queries likely lack scope
        "specificity": any(indicator in query.lower() for indicator in broad_indicators)
    }

    return {
        "is_broad": any(indicator in query.lower() for indicator in broad_indicators),
        "missing_context": missing_context_flags,
        "breadth_score": sum(missing_context_flags.values()) / len(missing_context_flags)
    }
```

### Phase 2: Research Question Transformation Component

**New file**: `src/open_deep_research_with_pydantic_ai/agents/question_transformer.py`

#### 2.1 Create Research Question Transformer Agent

```python
class ResearchQuestionTransformer(BaseResearchAgent):
    """Transforms conversation into detailed, specific research question.

    Based on LangGraph's transform_messages_into_research_topic_prompt approach.
    Ensures maximum specificity and identifies open-ended dimensions.
    """

    def _get_default_system_prompt(self) -> str:
        return """You transform user conversations into detailed research questions.

TRANSFORMATION PRINCIPLES:

1. **Maximize Specificity and Detail**:
   - Include all known user preferences and requirements
   - List key attributes or dimensions to consider explicitly
   - Preserve all details from user messages

2. **Handle Missing Dimensions**:
   - If essential attributes are missing, mark as open-ended
   - Don't invent details not provided by user
   - State lack of specification explicitly

3. **Avoid Unwarranted Assumptions**:
   - If user hasn't specified something, don't assume
   - Guide researcher to treat missing info as flexible
   - Accept all possible options when not specified

4. **Use First Person Perspective**:
   - Phrase from user's perspective: "I want to understand..."
   - Make it clear what the user is asking for

5. **Source Prioritization Guidelines**:
   - Academic queries: prefer original papers over summaries
   - Product research: prefer official sites over aggregators
   - People research: prefer LinkedIn/personal websites
   - Language-specific: prioritize sources in same language

OUTPUT FORMAT: Single detailed research question that guides comprehensive research."""

    async def transform_conversation(
        self,
        conversation: list[str],
        deps: ResearchDependencies
    ) -> str:
        """Transform conversation into detailed research question."""

        messages_context = "\n".join([
            f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}"
            for i, msg in enumerate(conversation)
        ])

        prompt = f"""Transform these messages into a detailed research question:
<Messages>
{messages_context}
</Messages>

Create a comprehensive research question that includes all user requirements
and explicitly states any dimensions left open-ended."""

        result = await self.run(prompt, deps, stream=False)
        return result
```

#### 2.2 Integration with Workflow

**Update**: `src/open_deep_research_with_pydantic_ai/core/workflow.py`

Add transformation step after clarification:
```python
# After clarification is complete, transform conversation into research question
transformer = ResearchQuestionTransformer()
detailed_question = await transformer.transform_conversation(
    conversation_messages, deps
)

# Store the detailed question for downstream agents
research_state.metadata["detailed_research_question"] = detailed_question
```

### Phase 3: Improved Brief Generation Integration

**File**: `src/open_deep_research_with_pydantic_ai/agents/brief_generator.py`

#### 3.1 Enhance Brief Generation with Transformed Question

```python
async def generate_from_conversation(self, deps: ResearchDependencies) -> BriefResult:
    """Generate brief using both conversation and transformed question."""

    # Get the detailed research question if available
    detailed_question = deps.research_state.metadata.get("detailed_research_question")

    if detailed_question:
        # Use the transformed question for better specificity
        prompt = f"""Generate a research brief for this detailed question:
{detailed_question}

Original conversation context:
{conversation_context}

Focus on the specific requirements and dimensions identified."""
    else:
        # Fallback to original conversation-based approach
        prompt = self._build_conversation_prompt(deps)

    # Continue with existing logic...
```

### Phase 4: Testing and Validation Framework

**New file**: `tests/test_clarification_improvement.py`

#### 4.1 Comprehensive Test Cases

```python
class TestClarificationImprovement:
    """Test suite for improved clarification system."""

    @pytest.mark.parametrize("query,should_clarify,reason", [
        # Broad queries that SHOULD require clarification
        ("What is quantum computing?", True, "missing_audience_and_scope"),
        ("Research artificial intelligence", True, "missing_focus_and_purpose"),
        ("Compare blockchain technologies", True, "missing_criteria_and_scope"),
        ("Analyze the healthcare industry", True, "missing_geographic_and_time_scope"),

        # Specific queries that should NOT require clarification
        ("I need a beginner-friendly explanation of quantum computing principles for a college presentation", False, "sufficient_context"),
        ("Compare OpenAI, Anthropic, and DeepMind's approaches to AI safety for academic research", False, "specific_entities_and_purpose"),

        # Edge cases
        ("What is ML?", True, "acronym_and_broad_scope"),
        ("Tell me about cats", True, "extremely_broad_scope"),
    ])
    async def test_clarification_assessment(self, query, should_clarify, reason):
        """Test that clarification logic correctly identifies when clarification is needed."""

        deps = create_mock_dependencies(query)
        result = await clarification_agent.assess_query(query, deps)

        assert result.need_clarification == should_clarify, f"Failed for {reason}: {query}"

        if should_clarify:
            assert result.question != "", "Should provide a clarifying question"
            assert len(result.question.split()) > 10, "Question should be detailed"
        else:
            assert result.verification != "", "Should provide verification message"

    async def test_question_transformation_specificity(self):
        """Test that question transformer adds appropriate specificity."""

        conversation = ["What is quantum computing?", "I need this for a business presentation"]

        transformer = ResearchQuestionTransformer()
        detailed_question = await transformer.transform_conversation(conversation, deps)

        # Should include business context and presentation purpose
        assert "business" in detailed_question.lower()
        assert "presentation" in detailed_question.lower()

        # Should identify missing dimensions
        assert any(word in detailed_question.lower()
                  for word in ["audience", "level", "depth", "focus"])
```

#### 4.2 Integration Tests

```python
class TestWorkflowIntegration:
    """Test complete clarification workflow."""

    async def test_broad_query_workflow(self):
        """Test that broad queries go through full clarification process."""

        # Test with "What is quantum computing?"
        result = await workflow.execute_research(
            user_query="What is quantum computing?",
            api_keys=mock_keys,
            stream_callback=None
        )

        # Should pause for clarification in HTTP mode
        assert result.metadata.get("awaiting_clarification") == True
        assert "clarification_question" in result.metadata

        # Question should be comprehensive
        question = result.metadata["clarification_question"]
        assert len(question) > 100, "Clarification question should be detailed"
        assert "audience" in question.lower()
        assert "purpose" in question.lower()

    async def test_specific_query_workflow(self):
        """Test that specific queries proceed without clarification."""

        specific_query = """I need a comprehensive analysis of quantum computing's
        commercial applications for enterprise decision-makers, focusing on
        current capabilities and 2-year outlook in finance and logistics sectors."""

        result = await workflow.execute_research(
            user_query=specific_query,
            api_keys=mock_keys,
            stream_callback=None
        )

        # Should proceed directly to research
        assert result.current_stage != ResearchStage.CLARIFICATION
        assert not result.metadata.get("awaiting_clarification")
```

### Phase 5: Configuration and Tuning

**File**: `src/open_deep_research_with_pydantic_ai/core/config.py`

#### 5.1 Add Clarification Configuration

```python
class APIConfig(BaseModel):
    # ... existing config ...

    # Enhanced clarification settings
    clarification_breadth_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CLARIFICATION_BREADTH_THRESHOLD", "0.6")),
        ge=0.0, le=1.0,
        description="Threshold for detecting broad queries (0.6 = 60% missing context triggers clarification)"
    )

    enable_question_transformation: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_QUESTION_TRANSFORMATION", "true").lower() == "true",
        description="Enable research question transformation after clarification"
    )

    clarification_detail_level: Literal["minimal", "standard", "comprehensive"] = Field(
        default_factory=lambda: os.getenv("CLARIFICATION_DETAIL_LEVEL", "standard"),
        description="How detailed clarification questions should be"
    )
```

## Implementation Timeline

### Week 1: Core Improvements
- [ ] Implement enhanced clarification system prompt
- [ ] Add scope breadth detection logic
- [ ] Create comprehensive test cases
- [ ] Test with problem queries like "What is quantum computing?"

### Week 2: Question Transformation
- [ ] Implement ResearchQuestionTransformer agent
- [ ] Integrate transformer into workflow
- [ ] Add transformation step after clarification
- [ ] Test transformation output quality

### Week 3: Integration and Testing
- [ ] Update brief generation to use transformed questions
- [ ] Implement comprehensive test suite
- [ ] Add configuration options
- [ ] Performance and integration testing

### Week 4: Validation and Optimization
- [ ] A/B test against current system
- [ ] Measure clarification accuracy improvements
- [ ] Fine-tune thresholds and prompts
- [ ] Documentation and deployment

## Success Metrics

### Quantitative Metrics
- **Clarification Accuracy**: % of broad queries correctly identified (target: >90%)
- **False Positive Rate**: % of specific queries incorrectly flagged (target: <10%)
- **Question Quality**: Average length and detail of clarifying questions (target: >150 words)
- **User Satisfaction**: Rating of clarification relevance (target: >4.5/5)

### Qualitative Metrics
- Broad queries like "What is quantum computing?" should consistently trigger clarification
- Clarifying questions should address specific missing dimensions (audience, purpose, scope)
- Research quality should improve due to better initial specification

## Risk Assessment

### High Risk
- **Breaking existing functionality**: Mitigation through comprehensive testing
- **Over-clarification**: Risk of asking too many questions. Mitigation through strict limits and quality thresholds

### Medium Risk
- **Performance impact**: Additional LLM calls for transformation. Mitigation through caching and optimization
- **Prompt engineering complexity**: Multiple interacting prompts. Mitigation through modular design and testing

### Low Risk
- **Configuration complexity**: Additional settings. Mitigation through sensible defaults
- **Integration complexity**: New components in workflow. Mitigation through clean interfaces

## Monitoring and Observability

### Key Metrics to Track
```python
# Add to research_state.metadata for monitoring
clarification_metrics = {
    "breadth_score": float,  # 0-1 score of query breadth
    "missing_dimensions": list[str],  # Which dimensions were missing
    "clarification_triggered": bool,  # Whether clarification was needed
    "transformation_applied": bool,  # Whether question transformation was used
    "question_complexity": int,  # Word count of transformed question
}
```

### Logging and Debugging
- Log all clarification decisions with reasoning
- Track false positives/negatives for continuous improvement
- Monitor transformation quality and user feedback
- A/B test different prompt variations

## Architectural Decision: Pydantic vs JSON

### Why We Keep Pydantic BaseModel

After analyzing both implementations, we found that **both LangGraph and our system use Pydantic BaseModel** for structured output. This is the superior approach because:

1. **Type Safety**: Compile-time type checking and IDE support
2. **Automatic Validation**: Built-in field validation, type coercion, and error handling
3. **Graceful Degradation**: Structured error messages and partial validation
4. **Integration**: Works seamlessly with Pydantic-AI's `.run()` method

### The Real Issue: Prompt Engineering

The problem was never about structure - it was about **insufficient behavioral guidance** in our prompts. LangGraph excels at:
- Explicit conditional instructions (if X then output Y)
- Clear field population rules
- Detailed examples and edge cases

## Conclusion

This implementation plan addresses the core issue of over-permissive clarification by:

1. **Improved prompt engineering** with explicit conditional output instructions (keeping Pydantic structure)
2. **Systematic scope assessment** using explicit criteria
3. **Research question transformation** to maximize specificity
4. **Comprehensive testing** to ensure reliability
5. **Configurable thresholds** for different use cases

The result will be a clarification system that properly identifies broad queries like "What is quantum computing?" while maintaining efficiency for well-specified requests.

**Expected outcome**: Broad, ambiguous queries will consistently trigger appropriate clarification, leading to higher-quality research outputs and better user experience.
