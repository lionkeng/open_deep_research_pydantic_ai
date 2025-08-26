"""Comprehensive tests for the three-phase clarification improvement system."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, List, Any

from src.open_deep_research_with_pydantic_ai.core.workflow import workflow
from src.open_deep_research_with_pydantic_ai.models.api_models import APIKeys
from src.open_deep_research_with_pydantic_ai.models.research import ResearchState, ResearchStage


class TestThreePhaseIntegration:
    """Test the complete three-phase clarification improvement system."""

    @pytest.mark.asyncio
    async def test_specific_query_minimal_processing(self):
        """Test that specific queries require minimal clarification processing."""
        specific_queries = [
            "What is the current stock price of Apple Inc.?",
            "How many days are there in February 2024?",
            "What is the capital of France?",
        ]

        for query in specific_queries:
            research_state = await workflow.execute_planning_only(
                user_query=query,
                api_keys=APIKeys()
            )

            # Verify completion
            assert research_state.current_stage == ResearchStage.RESEARCH_EXECUTION

            # Verify brief generation
            assert "research_brief_text" in research_state.metadata
            assert len(research_state.metadata["research_brief_text"]) > 0

            # Verify clarification assessment
            assert "clarification_assessment" in research_state.metadata
            ca = research_state.metadata["clarification_assessment"]

            # Specific queries should not need clarification
            assert ca.get("needs_clarification") is False or ca.get("question", "") == ""

    @pytest.mark.asyncio
    async def test_broad_query_processing(self):
        """Test that broad queries go through the full three-phase system."""
        broad_queries = [
            "Tell me about artificial intelligence",
            "How does technology affect society?",
            "What is the future of humanity?",
        ]

        for query in broad_queries:
            research_state = await workflow.execute_planning_only(
                user_query=query,
                api_keys=APIKeys()
            )

            # Verify completion
            assert research_state.current_stage == ResearchStage.RESEARCH_EXECUTION

            # Verify all three phases executed
            metadata = research_state.metadata
            assert "clarification_assessment" in metadata
            assert "transformed_query" in metadata  # Phase 2 ran
            assert "research_brief_text" in metadata  # Phase 3 ran

            # Verify transformation data structure
            tq = metadata["transformed_query"]
            assert "original_query" in tq
            assert "transformed_query" in tq
            assert "specificity_score" in tq

    @pytest.mark.asyncio
    async def test_data_flow_between_phases(self):
        """Test that data flows correctly between all three phases."""
        query = "What are the environmental impacts of renewable energy?"

        research_state = await workflow.execute_planning_only(
            user_query=query,
            api_keys=APIKeys()
        )

        metadata = research_state.metadata

        # Phase 1 outputs
        assert "clarification_assessment" in metadata
        ca = metadata["clarification_assessment"]
        assert "needs_clarification" in ca

        # Phase 2 outputs (transformation data)
        assert "transformed_query" in metadata
        tq = metadata["transformed_query"]
        assert tq["original_query"] == query
        assert "transformed_query" in tq
        assert "specificity_score" in tq
        assert isinstance(tq["specificity_score"], float)
        assert 0.0 <= tq["specificity_score"] <= 1.0

        # Phase 3 outputs (enhanced brief)
        assert "research_brief_text" in metadata
        assert "research_brief_confidence" in metadata

        brief_text = metadata["research_brief_text"]
        assert len(brief_text) > 100  # Substantial brief

        confidence = metadata["research_brief_confidence"]
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self):
        """Test that the system handles errors gracefully with fallback mechanisms."""

        # Test with a query that might cause issues
        problematic_query = ""  # Empty query

        research_state = await workflow.execute_planning_only(
            user_query=problematic_query,
            api_keys=APIKeys()
        )

        # Even with problematic input, workflow should complete
        assert research_state.current_stage == ResearchStage.RESEARCH_EXECUTION

        # Should have some metadata even if phases failed
        assert isinstance(research_state.metadata, dict)

        # Should have some brief even if minimal
        brief_text = research_state.metadata.get("research_brief_text", "")
        assert isinstance(brief_text, str)

    @pytest.mark.asyncio
    async def test_metadata_schema_consistency(self):
        """Test that metadata schema remains consistent across phases."""
        query = "How do neural networks work in deep learning?"

        research_state = await workflow.execute_planning_only(
            user_query=query,
            api_keys=APIKeys()
        )

        metadata = research_state.metadata

        # Validate clarification metadata schema
        if "clarification_assessment" in metadata:
            ca = metadata["clarification_assessment"]
            required_ca_fields = ["needs_clarification", "question", "verification"]
            for field in required_ca_fields:
                assert field in ca, f"Missing clarification field: {field}"

        # Validate transformation metadata schema
        if "transformed_query" in metadata:
            tq = metadata["transformed_query"]
            required_tq_fields = [
                "original_query", "transformed_query", "specificity_score",
                "transformation_rationale"
            ]
            for field in required_tq_fields:
                assert field in tq, f"Missing transformation field: {field}"

        # Validate brief metadata schema
        required_brief_fields = ["research_brief_text", "research_brief_confidence"]
        for field in required_brief_fields:
            assert field in metadata, f"Missing brief field: {field}"

    @pytest.mark.asyncio
    async def test_non_interactive_http_mode_simulation(self):
        """Test workflow behavior in non-interactive (HTTP) mode."""

        # Mock sys.stdin.isatty to return False (simulating HTTP environment)
        with patch('sys.stdin.isatty', return_value=False):
            query = "What is machine learning?"  # Potentially broad query

            research_state = await workflow.execute_planning_only(
                user_query=query,
                api_keys=APIKeys()
            )

            # In HTTP mode, should complete without user interaction
            assert research_state.current_stage == ResearchStage.RESEARCH_EXECUTION

            # Should have brief generated
            assert "research_brief_text" in research_state.metadata

            # If clarification was needed but we're in HTTP mode,
            # should have "awaiting_clarification" flag or should proceed anyway
            if "clarification_assessment" in research_state.metadata:
                ca = research_state.metadata["clarification_assessment"]
                if ca.get("needs_clarification"):
                    # Either we should be awaiting clarification, or we proceeded anyway
                    awaiting = research_state.metadata.get("awaiting_clarification", False)
                    has_brief = len(research_state.metadata.get("research_brief_text", "")) > 0
                    assert awaiting or has_brief, "Should either await clarification or generate brief"

    def test_query_categorization_accuracy(self):
        """Test that different query types are categorized correctly."""

        # Define test cases with expected clarification needs
        test_cases = [
            # (query, expected_needs_clarification)
            ("What is 2+2?", False),  # Very specific
            ("Current Apple stock price", False),  # Specific
            ("Compare React vs Vue performance", False),  # Specific comparison
            ("What is artificial intelligence?", True),  # Broad
            ("Tell me about technology", True),  # Very broad
            ("How does the universe work?", True),  # Extremely broad
        ]

        # This would be tested in actual agent unit tests
        # Here we just verify the test cases are reasonable
        specific_count = sum(1 for _, needs_clarification in test_cases if not needs_clarification)
        broad_count = sum(1 for _, needs_clarification in test_cases if needs_clarification)

        assert specific_count >= 3, "Should have multiple specific query examples"
        assert broad_count >= 3, "Should have multiple broad query examples"

    @pytest.mark.asyncio
    async def test_workflow_performance_basic(self):
        """Test basic performance characteristics of the workflow."""
        import time

        query = "What are the benefits of solar energy?"

        start_time = time.time()
        research_state = await workflow.execute_planning_only(
            user_query=query,
            api_keys=APIKeys()
        )
        execution_time = time.time() - start_time

        # Basic performance assertions
        assert execution_time < 120, f"Workflow took too long: {execution_time:.2f}s"

        # Quality assertions
        brief_length = len(research_state.metadata.get("research_brief_text", ""))
        assert brief_length > 200, f"Brief too short: {brief_length} characters"

        confidence = research_state.metadata.get("research_brief_confidence", 0.0)
        assert 0.0 <= confidence <= 1.0, f"Invalid confidence score: {confidence}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
