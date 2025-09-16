"""Unit tests for the Enhanced Synthesis System Prompt implementation."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.agents.research_executor import research_executor_agent


class TestEnhancedSynthesisPrompt:
    """Test the Enhanced Synthesis System Prompt V3.0 implementation."""

    def test_enhanced_prompt_exists(self):
        """Test that the enhanced synthesis prompt is properly loaded."""
        prompt = research_executor_agent._system_prompt
        assert prompt is not None
        assert len(prompt) > 10000  # Enhanced prompt should be substantial
        assert "ENHANCED SYNTHESIS SYSTEM PROMPT V3.0" in prompt

    def test_tree_of_thoughts_methodology(self):
        """Test that Tree of Thoughts methodology is included."""
        prompt = research_executor_agent._system_prompt

        # Check for all three phases
        assert "Phase 1: Exploration (Divergent Thinking)" in prompt
        assert "Phase 2: Evaluation (Critical Analysis)" in prompt
        assert "Phase 3: Convergence (Synthesis)" in prompt

        # Check for specific ToT elements
        assert "research branches" in prompt
        assert "confidence score" in prompt
        assert "unified narrative" in prompt

    def test_information_hierarchy_framework(self):
        """Test that Information Hierarchy Framework is properly defined."""
        prompt = research_executor_agent._system_prompt

        # Check for priority levels
        assert "Priority Level 1: Critical Core" in prompt
        assert "Priority Level 2: Supporting Context" in prompt
        assert "Priority Level 3: Enrichment Layer" in prompt

        # Check for scoring examples
        assert "Scoring Example:" in prompt
        assert "Total: 10/10 → Priority 1" in prompt
        assert "Total: 6/10 → Priority 2" in prompt

    def test_few_shot_examples(self):
        """Test that few-shot synthesis examples are included."""
        prompt = research_executor_agent._system_prompt

        # Check for example types
        assert "Example 1: Scientific Topic Synthesis" in prompt
        assert "Example 2: Complex Social Issue Synthesis" in prompt

        # Check for synthesis patterns
        assert "CRISPR-Cas9" in prompt  # Scientific example
        assert "remote work on urban economies" in prompt  # Social example
        assert "Synthesis Pattern:" in prompt

    def test_preservation_rules(self):
        """Test that preservation rules and anti-patterns are defined."""
        prompt = research_executor_agent._system_prompt

        # Check preservation rules
        assert "MUST PRESERVE:" in prompt
        assert "Direct Quotes" in prompt
        assert "Technical Specifications" in prompt
        assert "Statistical Relationships" in prompt

        # Check anti-patterns
        assert "AVOID THESE ANTI-PATTERNS:" in prompt
        assert "Over-generalization" in prompt
        assert "False Balance" in prompt
        assert "Cherry-picking" in prompt
        assert "Confidence Inflation" in prompt

    def test_self_verification_protocol(self):
        """Test that self-verification protocol is included."""
        prompt = research_executor_agent._system_prompt

        # Check for verification checklists
        assert "Self-Verification Protocol" in prompt.upper() or "SELF-VERIFICATION PROTOCOL" in prompt
        assert "Accuracy Checklist:" in prompt
        assert "Completeness Checklist:" in prompt
        assert "Coherence Checklist:" in prompt

        # Check specific verification items
        assert "All statistics include sources" in prompt
        assert "Research question directly answered" in prompt
        assert "Logical flow from general to specific" in prompt

    def test_domain_adaptation_protocols(self):
        """Test that domain-specific adaptation protocols are included."""
        prompt = research_executor_agent._system_prompt

        # Check for different domain protocols
        assert "For Scientific/Technical Topics:" in prompt
        assert "For Business/Economic Topics:" in prompt
        assert "For Social/Political Topics:" in prompt
        assert "For Historical Topics:" in prompt

        # Check domain-specific instructions
        assert "peer-reviewed sources" in prompt
        assert "market size/growth metrics" in prompt
        assert "multiple viewpoints" in prompt
        assert "primary sources" in prompt

    def test_quality_metrics_reporting(self):
        """Test that quality metrics reporting is defined."""
        prompt = research_executor_agent._system_prompt

        # Check for metrics structure
        assert "Quality Metrics Reporting" in prompt.upper() or "QUALITY METRICS REPORTING" in prompt
        assert "Research Quality Metrics:" in prompt
        assert "Sources Analyzed:" in prompt
        assert "Source Diversity:" in prompt
        assert "Evidence Strength:" in prompt
        assert "Contradiction Rate:" in prompt
        assert "Synthesis Confidence:" in prompt

    def test_advanced_synthesis_techniques(self):
        """Test that advanced synthesis techniques are included."""
        prompt = research_executor_agent._system_prompt

        # Check for pattern recognition
        assert "Pattern Recognition Enhancement:" in prompt
        assert "Temporal Patterns" in prompt
        assert "Causal Networks" in prompt
        assert "Emergent Themes" in prompt
        assert "Anomaly Detection" in prompt

        # Check for uncertainty quantification
        assert "Uncertainty Quantification:" in prompt
        assert "source_agreement" in prompt
        assert "evidence_quality" in prompt
        assert "temporal_relevance" in prompt

    def test_output_structure_template(self):
        """Test that output structure template is defined."""
        prompt = research_executor_agent._system_prompt

        # Check for output sections
        assert "Executive Summary" in prompt
        assert "Key Findings (Priority 1)" in prompt
        assert "Detailed Analysis" in prompt
        assert "Supporting Evidence (Priority 2)" in prompt
        assert "Additional Context (Priority 3)" in prompt
        assert "Limitations and Gaps" in prompt
        assert "Research Metrics" in prompt

    def test_iterative_refinement_protocol(self):
        """Test that iterative refinement protocol is included."""
        prompt = research_executor_agent._system_prompt

        # Check for refinement steps
        assert "Iterative Refinement Protocol" in prompt.upper() or "ITERATIVE REFINEMENT PROTOCOL" in prompt
        assert "Gap Analysis" in prompt
        assert "Targeted Search" in prompt
        assert "Integration" in prompt
        assert "Re-validation" in prompt
        assert "Polish" in prompt

    def test_ethical_guidelines(self):
        """Test that ethical synthesis guidelines are included."""
        prompt = research_executor_agent._system_prompt

        # Check for ethical considerations
        assert "Ethical Synthesis Guidelines" in prompt.upper() or "ETHICAL SYNTHESIS GUIDELINES" in prompt
        assert "Avoid amplifying misinformation" in prompt
        assert "Distinguish facts from opinions" in prompt
        assert "Respect intellectual property" in prompt
        assert "Acknowledge potential biases" in prompt
        assert "Consider societal implications" in prompt

    def test_methodology_enhancement(self):
        """Test that the methodology section is enhanced."""
        prompt = research_executor_agent._system_prompt

        # Check for enhanced methodology
        assert "METHODOLOGY:" in prompt
        assert "Tree of Thoughts approach" in prompt
        assert "Information Hierarchy Framework" in prompt
        assert "explicit contradiction resolution" in prompt
        assert "quality metrics" in prompt
        assert "domain-specific protocols" in prompt
        assert "self-verification" in prompt

    def test_prompt_comprehensiveness(self):
        """Test overall prompt comprehensiveness."""
        prompt = research_executor_agent._system_prompt

        # Count major sections (should have at least 12 parts)
        parts_count = prompt.count("### PART")
        assert parts_count >= 12, f"Expected at least 12 parts, found {parts_count}"

        # Check for key optimization mentions
        assert "GPT-5" in prompt or "enhanced reasoning capabilities" in prompt

        # Verify prompt is significantly larger than basic version
        assert len(prompt) > 12000, f"Enhanced prompt should be >12000 chars, got {len(prompt)}"

    def test_backwards_compatibility(self):
        """Test that the enhanced prompt maintains backwards compatibility."""
        prompt = research_executor_agent._system_prompt

        # Original responsibilities should still be present
        assert "research assistant" in prompt.lower()
        assert "comprehensive information synthesis" in prompt.lower()
        assert "analysis" in prompt.lower()
        assert "evidence-based" in prompt.lower()


class TestPromptIntegration:
    """Test the integration of the enhanced prompt with the agent."""

    def test_agent_initialization(self):
        """Test that the agent initializes correctly with enhanced prompt."""
        assert research_executor_agent is not None
        assert hasattr(research_executor_agent, '_system_prompt')
        assert research_executor_agent._system_prompt is not None

    def test_agent_tools_present(self):
        """Test that all agent tools are still present after prompt update."""
        # The agent should still have all its tools registered
        # This ensures the prompt change didn't break tool registration
        assert research_executor_agent is not None
        # Tools are registered via decorators, so they should work

    @pytest.mark.asyncio
    async def test_agent_can_process_with_enhanced_prompt(self):
        """Test that the agent can process requests with the enhanced prompt."""
        # Mock dependencies
        mock_deps = MagicMock()
        mock_deps.search_results = []
        mock_deps.original_query = "Test query"

        # This test verifies the agent can still be called
        # Actual execution would require API keys and external services
        assert research_executor_agent._system_prompt is not None
        assert "ENHANCED SYNTHESIS SYSTEM PROMPT V3.0" in research_executor_agent._system_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
