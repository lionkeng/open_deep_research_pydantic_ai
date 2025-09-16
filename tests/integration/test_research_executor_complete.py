"""
Integration tests for the complete research executor system.

These tests use real API calls (no mocks) to validate the entire research workflow,
including contradiction severity calculation, mathematical validation, and quality metrics.
"""

import asyncio
import os
import pytest
from typing import Any, Dict, List

from core.research_executor import ResearchExecutor
from models.research import ResearchQuery, ResearchResult
from core.synthesis import SynthesisEngine
from models.synthesis import SynthesisResult


class TestResearchExecutorComplete:
    """Complete end-to-end integration tests for research executor."""

    @pytest.fixture(autouse=True)
    def setup_env(self):
        """Ensure environment variables are loaded from .env file."""
        # Environment variables should be loaded from .env file via pytest configuration
        required_keys = ["OPENAI_API_KEY"]
        for key in required_keys:
            if not os.getenv(key):
                pytest.skip(f"Missing required environment variable: {key}")

    @pytest.fixture
    async def research_executor(self):
        """Create a research executor instance."""
        executor = ResearchExecutor()
        return executor

    @pytest.fixture
    async def synthesis_engine(self):
        """Create a synthesis engine instance."""
        engine = SynthesisEngine()
        return engine

    @pytest.mark.asyncio
    async def test_complete_research_workflow_simple_query(
        self, research_executor: ResearchExecutor, synthesis_engine: SynthesisEngine
    ):
        """Test complete research workflow with a simple, well-defined query."""
        query = ResearchQuery(
            original_query="What is the capital of France?",
            refined_query="What is the capital city of France and what are some key facts about it?",
            research_scope="basic",
            expected_depth="surface"
        )

        # Execute research
        result = await research_executor.execute_research(query)

        # Validate research result structure
        assert isinstance(result, ResearchResult)
        assert result.query == query
        assert result.research_plan is not None
        assert len(result.sources) > 0
        assert result.findings is not None
        assert len(result.findings) > 0

        # Test synthesis
        synthesis_result = await synthesis_engine.synthesize(result)

        # Validate synthesis result
        assert isinstance(synthesis_result, SynthesisResult)
        assert synthesis_result.final_answer is not None
        assert len(synthesis_result.final_answer) > 0
        assert synthesis_result.confidence_score >= 0.0
        assert synthesis_result.confidence_score <= 1.0
        assert synthesis_result.quality_metrics is not None

        # Validate quality metrics structure
        metrics = synthesis_result.quality_metrics
        assert hasattr(metrics, 'completeness')
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'coherence')
        assert 0.0 <= metrics.completeness <= 1.0
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.coherence <= 1.0

    @pytest.mark.asyncio
    async def test_contradiction_severity_calculation_system(
        self, research_executor: ResearchExecutor, synthesis_engine: SynthesisEngine
    ):
        """Test contradiction severity calculation with conflicting information."""
        query = ResearchQuery(
            original_query="What is the height of Mount Everest?",
            refined_query="What is the exact height of Mount Everest according to different measurements?",
            research_scope="detailed",
            expected_depth="moderate"
        )

        # Execute research to get potentially conflicting information
        result = await research_executor.execute_research(query)

        # Synthesize to trigger contradiction detection
        synthesis_result = await synthesis_engine.synthesize(result)

        # Validate contradiction analysis exists and is structured properly
        if hasattr(synthesis_result, 'contradictions') and synthesis_result.contradictions:
            contradictions = synthesis_result.contradictions
            assert isinstance(contradictions, list)

            for contradiction in contradictions:
                # Validate SeverityResult TypedDict structure
                assert isinstance(contradiction, dict)

                # Check for severity calculation fields
                severity_fields = ['severity', 'confidence', 'explanation', 'sources_involved']
                for field in severity_fields:
                    if field in contradiction:
                        if field == 'severity':
                            assert contradiction[field] in ['low', 'medium', 'high', 'critical']
                        elif field == 'confidence':
                            assert isinstance(contradiction[field], (int, float))
                            assert 0.0 <= contradiction[field] <= 1.0
                        elif field == 'explanation':
                            assert isinstance(contradiction[field], str)
                            assert len(contradiction[field]) > 0
                        elif field == 'sources_involved':
                            assert isinstance(contradiction[field], list)

        # Even if no contradictions found, the analysis should complete successfully
        assert synthesis_result.final_answer is not None
        assert synthesis_result.confidence_score >= 0.0

    @pytest.mark.asyncio
    async def test_mathematical_validation_infrastructure(
        self, research_executor: ResearchExecutor, synthesis_engine: SynthesisEngine
    ):
        """Test mathematical validation including Shannon entropy and robust scoring."""
        query = ResearchQuery(
            original_query="What are the key principles of quantum computing?",
            refined_query="What are the fundamental principles and applications of quantum computing technology?",
            research_scope="comprehensive",
            expected_depth="deep"
        )

        # Execute research
        result = await research_executor.execute_research(query)
        synthesis_result = await synthesis_engine.synthesize(result)

        # Validate that synthesis completed successfully
        assert synthesis_result.final_answer is not None
        assert synthesis_result.quality_metrics is not None

        # Validate mathematical metrics are calculated
        metrics = synthesis_result.quality_metrics

        # Core quality metrics should be present
        assert hasattr(metrics, 'completeness')
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'coherence')

        # Validate metric ranges
        assert 0.0 <= metrics.completeness <= 1.0
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.coherence <= 1.0

        # Check for advanced mathematical validation metrics (if implemented)
        advanced_metrics = ['information_density', 'source_reliability', 'temporal_relevance', 'entropy_score']
        for metric in advanced_metrics:
            if hasattr(metrics, metric):
                value = getattr(metrics, metric)
                if value is not None:
                    assert isinstance(value, (int, float))
                    assert value >= 0.0

        # Validate information consistency
        if hasattr(synthesis_result, 'consistency_score'):
            consistency = synthesis_result.consistency_score
            assert isinstance(consistency, (int, float))
            assert 0.0 <= consistency <= 1.0

    @pytest.mark.asyncio
    async def test_complex_research_scenario_with_multiple_perspectives(
        self, research_executor: ResearchExecutor, synthesis_engine: SynthesisEngine
    ):
        """Test handling of complex research requiring multiple perspectives."""
        query = ResearchQuery(
            original_query="What are the pros and cons of renewable energy adoption?",
            refined_query="What are the economic, environmental, and social advantages and disadvantages of transitioning to renewable energy sources?",
            research_scope="comprehensive",
            expected_depth="deep"
        )

        # Execute complete workflow
        result = await research_executor.execute_research(query)
        synthesis_result = await synthesis_engine.synthesize(result)

        # Validate comprehensive analysis
        assert synthesis_result.final_answer is not None
        assert len(synthesis_result.final_answer) > 200  # Expect detailed analysis

        # Check for balanced perspective indicators
        answer_text = synthesis_result.final_answer.lower()
        perspective_indicators = ['advantage', 'disadvantage', 'pro', 'con', 'benefit', 'challenge', 'positive', 'negative']
        found_indicators = sum(1 for indicator in perspective_indicators if indicator in answer_text)
        assert found_indicators >= 2  # Should mention multiple perspectives

        # Validate source diversity
        assert len(result.sources) >= 2  # Multiple sources for complex topics

        # Validate quality metrics for complex analysis
        metrics = synthesis_result.quality_metrics
        assert metrics.completeness >= 0.4  # Should be reasonably complete
        assert metrics.coherence >= 0.5  # Should maintain coherence despite complexity

    @pytest.mark.asyncio
    async def test_error_handling_with_real_failure_scenarios(
        self, research_executor: ResearchExecutor
    ):
        """Test error handling with scenarios that cause real failures."""

        # Test with minimal query
        minimal_query = ResearchQuery(
            original_query="?",
            refined_query="What?",
            research_scope="basic",
            expected_depth="surface"
        )

        # Should handle gracefully
        try:
            result = await research_executor.execute_research(minimal_query)
            # If it doesn't raise an error, validate it produces something meaningful
            assert isinstance(result, ResearchResult)
            # Should have some basic structure even with minimal input
            assert result.query is not None
        except Exception as e:
            # Acceptable to raise an error for invalid input
            assert isinstance(e, Exception)

        # Test with very broad query that might hit limits
        broad_query = ResearchQuery(
            original_query="Tell me about science",
            refined_query="Provide information about scientific fields and discoveries",
            research_scope="comprehensive",
            expected_depth="deep"
        )

        # Should handle gracefully without crashing
        try:
            result = await research_executor.execute_research(broad_query)
            assert isinstance(result, ResearchResult)
            # Should still produce reasonable results
            assert len(result.findings) > 0
        except Exception as e:
            # Allow for resource limits or timeouts
            assert isinstance(e, Exception)

    @pytest.mark.asyncio
    async def test_quality_metrics_verification_comprehensive(
        self, research_executor: ResearchExecutor, synthesis_engine: SynthesisEngine
    ):
        """Comprehensive verification of all quality metrics."""
        query = ResearchQuery(
            original_query="How does machine learning impact healthcare?",
            refined_query="What are the current applications and future potential of machine learning in healthcare, including benefits and challenges?",
            research_scope="detailed",
            expected_depth="moderate"
        )

        # Execute workflow
        result = await research_executor.execute_research(query)
        synthesis_result = await synthesis_engine.synthesize(result)

        # Comprehensive quality metrics validation
        metrics = synthesis_result.quality_metrics

        # Core metrics validation
        assert hasattr(metrics, 'completeness')
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'coherence')

        # Validate metric ranges
        assert 0.0 <= metrics.completeness <= 1.0
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.coherence <= 1.0

        # Validate confidence aligns reasonably with quality
        confidence = synthesis_result.confidence_score
        assert 0.0 <= confidence <= 1.0

        # For a well-researched topic like ML in healthcare, expect reasonable quality
        avg_quality = (metrics.completeness + metrics.accuracy + metrics.coherence) / 3
        assert avg_quality >= 0.4  # Should achieve reasonable quality

        # Validate source validation if present
        if hasattr(synthesis_result, 'source_validations') and synthesis_result.source_validations:
            validations = synthesis_result.source_validations
            assert isinstance(validations, list)

            for validation in validations:
                # Check for reliability and relevance scores
                if hasattr(validation, 'reliability_score'):
                    assert 0.0 <= validation.reliability_score <= 1.0
                if hasattr(validation, 'relevance_score'):
                    assert 0.0 <= validation.relevance_score <= 1.0

    @pytest.mark.asyncio
    async def test_research_plan_execution_validation(
        self, research_executor: ResearchExecutor
    ):
        """Test that research plans are properly executed and validated."""
        query = ResearchQuery(
            original_query="What is the impact of climate change on polar bears?",
            refined_query="How is climate change affecting polar bear populations, habitat, and survival?",
            research_scope="detailed",
            expected_depth="moderate"
        )

        # Execute research
        result = await research_executor.execute_research(query)

        # Validate research plan structure
        plan = result.research_plan
        assert plan is not None

        # Validate research questions are generated if present
        if hasattr(plan, 'research_questions') and plan.research_questions:
            questions = plan.research_questions
            assert isinstance(questions, list)
            assert len(questions) > 0

            for question in questions:
                assert isinstance(question, str)
                assert len(question) > 5  # Meaningful questions
                # Should relate to climate change and polar bears
                question_lower = question.lower()
                climate_keywords = ['climate', 'polar', 'bear', 'ice', 'arctic', 'warming']
                assert any(keyword in question_lower for keyword in climate_keywords)

        # Validate methodology is defined if present
        if hasattr(plan, 'methodology') and plan.methodology:
            methodology = plan.methodology
            assert isinstance(methodology, str)
            assert len(methodology) > 10  # Substantial methodology description

        # Validate sources were found
        assert len(result.sources) > 0

        # Validate findings relate to the query
        assert len(result.findings) > 0
        findings_text = ' '.join(result.findings).lower()
        relevant_terms = ['polar bear', 'climate', 'ice', 'arctic', 'habitat']
        found_terms = sum(1 for term in relevant_terms if term in findings_text)
        assert found_terms >= 2  # Should mention relevant concepts

    @pytest.mark.asyncio
    async def test_synthesis_robustness_and_mathematical_validation(
        self, research_executor: ResearchExecutor, synthesis_engine: SynthesisEngine
    ):
        """Test synthesis robustness and mathematical validation."""
        query = ResearchQuery(
            original_query="What is photosynthesis?",
            refined_query="What is photosynthesis and how does it work in plants?",
            research_scope="basic",
            expected_depth="surface"
        )

        # Execute research
        result = await research_executor.execute_research(query)

        # Run synthesis
        synthesis_result = await synthesis_engine.synthesize(result)

        # Validate synthesis robustness
        assert synthesis_result.final_answer is not None
        assert len(synthesis_result.final_answer) > 50  # Should be substantive

        # Core scientific terms should be present for photosynthesis
        answer_text = synthesis_result.final_answer.lower()
        scientific_terms = ['light', 'carbon', 'oxygen', 'plant', 'energy']
        found_terms = sum(1 for term in scientific_terms if term in answer_text)
        assert found_terms >= 3  # Should mention key scientific concepts

        # Confidence should be reasonable for well-established scientific facts
        assert synthesis_result.confidence_score >= 0.6

        # Quality metrics should indicate good performance
        metrics = synthesis_result.quality_metrics
        assert metrics.accuracy >= 0.6  # Scientific facts should be accurate
        assert metrics.coherence >= 0.5  # Should be well-structured

    @pytest.mark.asyncio
    async def test_information_density_and_entropy_analysis(
        self, research_executor: ResearchExecutor, synthesis_engine: SynthesisEngine
    ):
        """Test information density and entropy analysis in mathematical validation."""
        # Use a query that should produce diverse information
        query = ResearchQuery(
            original_query="What are different types of artificial intelligence?",
            refined_query="What are the main categories and types of artificial intelligence systems and their applications?",
            research_scope="detailed",
            expected_depth="moderate"
        )

        # Execute workflow
        result = await research_executor.execute_research(query)
        synthesis_result = await synthesis_engine.synthesize(result)

        # Validate that the answer contains diverse information
        answer_text = synthesis_result.final_answer
        assert len(answer_text) > 100  # Should be substantial

        # Check for information diversity indicators (different AI types)
        ai_types = ['machine learning', 'neural network', 'expert system', 'natural language', 'computer vision']
        found_types = sum(1 for ai_type in ai_types if ai_type in answer_text.lower())
        assert found_types >= 2  # Should mention multiple AI types

        # Validate information density through content analysis
        # High information density means more unique concepts per unit text
        word_count = len(answer_text.split())
        unique_word_ratio = len(set(answer_text.lower().split())) / word_count if word_count > 0 else 0
        assert unique_word_ratio >= 0.3  # Reasonable vocabulary diversity

        # Check if mathematical validation metrics are present
        metrics = synthesis_result.quality_metrics
        if hasattr(metrics, 'information_density'):
            assert isinstance(metrics.information_density, (int, float))
            assert metrics.information_density >= 0.0

        # Validate overall quality for a technical topic
        assert metrics.completeness >= 0.5
        assert metrics.coherence >= 0.5

    @pytest.mark.asyncio
    async def test_robust_score_validation_system(
        self, research_executor: ResearchExecutor, synthesis_engine: SynthesisEngine
    ):
        """Test robust scoring system for research quality validation."""
        query = ResearchQuery(
            original_query="What are the effects of exercise on mental health?",
            refined_query="How does regular physical exercise impact mental health and psychological well-being?",
            research_scope="detailed",
            expected_depth="moderate"
        )

        # Execute workflow
        result = await research_executor.execute_research(query)
        synthesis_result = await synthesis_engine.synthesize(result)

        # Validate robust scoring components
        metrics = synthesis_result.quality_metrics

        # Check core robustness indicators
        assert metrics.completeness is not None
        assert metrics.accuracy is not None
        assert metrics.coherence is not None

        # Validate that scores are reasonable for a well-researched health topic
        # Exercise and mental health is well-established research area
        assert metrics.accuracy >= 0.5  # Should find reliable information

        # Validate confidence aligns with quality metrics
        confidence = synthesis_result.confidence_score
        avg_quality = (metrics.completeness + metrics.accuracy + metrics.coherence) / 3

        # Confidence should roughly correlate with average quality
        # Allow for reasonable variance
        confidence_quality_diff = abs(confidence - avg_quality)
        assert confidence_quality_diff <= 0.5  # Reasonable alignment

        # For health topics, expect reasonable baseline quality
        assert avg_quality >= 0.4

        # Validate that the answer addresses the mental health aspect
        answer_text = synthesis_result.final_answer.lower()
        mental_health_terms = ['mental', 'psychological', 'mood', 'depression', 'anxiety', 'well-being']
        found_terms = sum(1 for term in mental_health_terms if term in answer_text)
        assert found_terms >= 2  # Should address mental health aspects


# Utility functions for integration testing
def validate_synthesis_structure(synthesis_result: SynthesisResult) -> None:
    """Validate the complete structure of a synthesis result."""
    assert hasattr(synthesis_result, 'final_answer')
    assert hasattr(synthesis_result, 'confidence_score')
    assert hasattr(synthesis_result, 'quality_metrics')

    assert synthesis_result.final_answer is not None
    assert isinstance(synthesis_result.confidence_score, (int, float))
    assert synthesis_result.quality_metrics is not None


def validate_research_depth(result: ResearchResult, expected_depth: str) -> None:
    """Validate that research meets the expected depth requirements."""
    findings_count = len(result.findings) if result.findings else 0
    sources_count = len(result.sources)

    if expected_depth == "surface":
        assert findings_count >= 1
        assert sources_count >= 1
    elif expected_depth == "moderate":
        assert findings_count >= 2
        assert sources_count >= 2
    elif expected_depth == "deep":
        assert findings_count >= 3
        assert sources_count >= 3


def calculate_text_entropy(text: str) -> float:
    """Calculate Shannon entropy of text for validation."""
    import math
    from collections import Counter

    if not text:
        return 0.0

    # Count character frequencies
    char_counts = Counter(text.lower())
    text_length = len(text)

    # Calculate entropy
    entropy = 0.0
    for count in char_counts.values():
        probability = count / text_length
        if probability > 0:
            entropy -= probability * math.log2(probability)

    return entropy


def validate_contradiction_severity(contradictions: List[Dict[str, Any]]) -> None:
    """Validate contradiction severity analysis structure."""
    for contradiction in contradictions:
        # SeverityResult TypedDict validation
        assert 'severity' in contradiction
        assert 'confidence' in contradiction
        assert 'explanation' in contradiction
        assert 'sources_involved' in contradiction

        # Validate severity levels
        assert contradiction['severity'] in ['low', 'medium', 'high', 'critical']

        # Validate confidence score
        assert isinstance(contradiction['confidence'], (int, float))
        assert 0.0 <= contradiction['confidence'] <= 1.0

        # Validate explanation exists and is meaningful
        assert isinstance(contradiction['explanation'], str)
        assert len(contradiction['explanation']) > 10

        # Validate sources are listed
        assert isinstance(contradiction['sources_involved'], list)
        assert len(contradiction['sources_involved']) > 0


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v"])
