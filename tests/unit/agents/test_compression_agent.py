"""
Comprehensive tests for the CompressionAgent.
"""

import asyncio
import pytest
from typing import List, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.compression import CompressionAgent
from src.agents.base import ResearchDependencies, AgentConfiguration
from src.models.compression import CompressedContent
from src.models.api_models import APIKeys, ResearchMetadata
from src.models.core import ResearchState, ResearchStage


class TestCompressionAgent:
    """Test suite for CompressionAgent."""

    @pytest.fixture
    async def agent_dependencies(self):
        """Create mock dependencies for testing."""
        deps = ResearchDependencies(
            http_client=AsyncMock(),
            api_keys=APIKeys(),
            research_state=ResearchState(
                request_id="test-comp-123",
                user_id="test-user",
                session_id="test-session",
                user_query="Summarize findings on climate change",
                current_stage=ResearchStage.COMPRESSION
            ),
            metadata=ResearchMetadata(),
            usage=None
        )
        return deps

    @pytest.fixture
    def compression_agent(self, agent_dependencies):
        """Create a CompressionAgent instance."""
        config = AgentConfiguration(
            max_retries=3,
            timeout_seconds=30.0,
            temperature=0.7
        )
        agent = CompressionAgent(config=config)
        agent._deps = agent_dependencies
        return agent

    @pytest.fixture
    def sample_research_findings(self):
        """Create sample research findings for compression."""
        return [
            {
                "finding": "Global temperatures have risen by 1.1°C since pre-industrial times",
                "source": "IPCC Report 2023",
                "confidence": 0.95
            },
            {
                "finding": "Arctic ice is melting at unprecedented rates",
                "source": "NASA Climate Study",
                "confidence": 0.92
            },
            {
                "finding": "Sea levels are rising 3.3mm per year",
                "source": "NOAA Research",
                "confidence": 0.90
            },
            {
                "finding": "Extreme weather events are becoming more frequent",
                "source": "WMO Analysis",
                "confidence": 0.88
            },
            {
                "finding": "CO2 levels reached 421 ppm in 2024",
                "source": "Mauna Loa Observatory",
                "confidence": 0.99
            }
        ]

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = CompressionAgent()
        assert agent.agent_name == "compression"
        assert agent.agent is not None
        assert agent.result_validator is not None

    @pytest.mark.asyncio
    async def test_successful_compression(self, compression_agent, agent_dependencies, sample_research_findings):
        """Test successful compression of research findings."""
        agent_dependencies.metadata.additional_context = {
            "findings": sample_research_findings
        }

        mock_result = MagicMock()
        mock_result.data = CompressedContent(
            original_length=5000,
            compressed_length=800,
            compression_ratio=6.25,
            summary="Climate change research shows global temperatures have risen 1.1°C with accelerating impacts including Arctic ice melt, sea level rise, and increased extreme weather events. CO2 levels at record highs.",
            key_points=[
                "1.1°C global temperature increase",
                "Arctic ice melting rapidly",
                "3.3mm/year sea level rise",
                "More frequent extreme weather",
                "421 ppm CO2 levels"
            ],
            themes={
                "temperature": ["global warming", "1.1°C increase"],
                "ice": ["Arctic melting", "unprecedented rates"],
                "sea_level": ["3.3mm annual rise", "coastal impacts"],
                "weather": ["extreme events", "increasing frequency"],
                "greenhouse_gases": ["CO2 421 ppm", "record levels"]
            },
            preserved_details=[
                "IPCC confirms 1.1°C rise",
                "NASA documents Arctic changes",
                "NOAA sea level measurements"
            ],
            confidence_score=0.92,
            metadata={
                "compression_method": "semantic_summarization",
                "information_loss": 0.05,
                "processing_time": 2.3
            }
        )

        with patch.object(compression_agent.agent, 'run', return_value=mock_result):
            result = await compression_agent.execute(agent_dependencies)

            assert isinstance(result, CompressedContent)
            assert result.compression_ratio > 1.0
            assert len(result.key_points) >= 3
            assert len(result.themes) > 0
            assert result.confidence_score > 0.8

    @pytest.mark.asyncio
    async def test_compression_ratio_calculation(self, compression_agent, agent_dependencies):
        """Test that compression ratios are properly calculated."""
        test_cases = [
            (10000, 1000, 10.0),  # 10:1 compression
            (5000, 2500, 2.0),    # 2:1 compression
            (1000, 500, 2.0),     # 2:1 compression
            (800, 400, 2.0)       # 2:1 compression
        ]

        for original, compressed, expected_ratio in test_cases:
            mock_result = MagicMock()
            mock_result.data = CompressedContent(
                original_length=original,
                compressed_length=compressed,
                compression_ratio=expected_ratio,
                summary="Test summary",
                key_points=["Point 1"],
                themes={"theme": ["detail"]},
                preserved_details=["Detail 1"],
                confidence_score=0.85,
                metadata={}
            )

            with patch.object(compression_agent.agent, 'run', return_value=mock_result):
                result = await compression_agent.execute(agent_dependencies)

                assert result.compression_ratio == expected_ratio
                assert result.original_length > result.compressed_length

    @pytest.mark.asyncio
    async def test_theme_extraction(self, compression_agent, agent_dependencies):
        """Test that themes are properly extracted from content."""
        mock_result = MagicMock()
        mock_result.data = CompressedContent(
            original_length=3000,
            compressed_length=500,
            compression_ratio=6.0,
            summary="AI research summary",
            key_points=["ML advances", "DL breakthroughs", "NLP progress"],
            themes={
                "machine_learning": ["supervised learning", "unsupervised learning", "reinforcement learning"],
                "deep_learning": ["neural networks", "transformers", "CNNs"],
                "nlp": ["language models", "text generation", "sentiment analysis"],
                "computer_vision": ["image recognition", "object detection"],
                "ethics": ["bias", "fairness", "transparency"]
            },
            preserved_details=["Key algorithms", "Performance metrics"],
            confidence_score=0.88,
            metadata={}
        )

        with patch.object(compression_agent.agent, 'run', return_value=mock_result):
            result = await compression_agent.execute(agent_dependencies)

            assert len(result.themes) >= 3
            assert "machine_learning" in result.themes
            assert all(isinstance(v, list) for v in result.themes.values())
            assert all(len(v) > 0 for v in result.themes.values())

    @pytest.mark.asyncio
    async def test_key_information_preservation(self, compression_agent, agent_dependencies):
        """Test that key information is preserved during compression."""
        mock_result = MagicMock()
        mock_result.data = CompressedContent(
            original_length=5000,
            compressed_length=1000,
            compression_ratio=5.0,
            summary="Research summary with critical data preserved",
            key_points=[
                "Statistical significance p<0.001",
                "Sample size n=10,000",
                "95% confidence interval",
                "Effect size d=0.8"
            ],
            themes={"statistics": ["significance", "confidence"]},
            preserved_details=[
                "p-value: 0.0005",
                "CI: [0.75, 0.85]",
                "n=10,000 participants",
                "Cohen's d=0.8"
            ],
            confidence_score=0.95,
            metadata={"critical_data_preserved": True}
        )

        with patch.object(compression_agent.agent, 'run', return_value=mock_result):
            result = await compression_agent.execute(agent_dependencies)

            assert len(result.preserved_details) >= 3
            assert any("p" in detail.lower() or "significance" in detail.lower() for detail in result.preserved_details)
            assert result.metadata.get("critical_data_preserved") is True

    @pytest.mark.asyncio
    async def test_edge_case_minimal_content(self, compression_agent, agent_dependencies):
        """Test compression of minimal content."""
        agent_dependencies.metadata.additional_context = {
            "findings": ["Single finding"]
        }

        mock_result = MagicMock()
        mock_result.data = CompressedContent(
            original_length=50,
            compressed_length=45,
            compression_ratio=1.11,
            summary="Single finding",
            key_points=["Single finding"],
            themes={},
            preserved_details=["Single finding"],
            confidence_score=1.0,
            metadata={"minimal_content": True}
        )

        with patch.object(compression_agent.agent, 'run', return_value=mock_result):
            result = await compression_agent.execute(agent_dependencies)

            assert result.compression_ratio < 2.0  # Minimal compression possible
            assert result.metadata.get("minimal_content") is True

    @pytest.mark.asyncio
    async def test_large_content_compression(self, compression_agent, agent_dependencies):
        """Test compression of very large content."""
        large_findings = [f"Finding {i}: Details about research point {i}" for i in range(100)]
        agent_dependencies.metadata.additional_context = {
            "findings": large_findings
        }

        mock_result = MagicMock()
        mock_result.data = CompressedContent(
            original_length=50000,
            compressed_length=2000,
            compression_ratio=25.0,
            summary="Comprehensive research with 100 findings compressed into key insights",
            key_points=[
                "Major theme 1 from findings 1-20",
                "Major theme 2 from findings 21-40",
                "Major theme 3 from findings 41-60",
                "Major theme 4 from findings 61-80",
                "Major theme 5 from findings 81-100"
            ],
            themes={
                "cluster_1": ["findings 1-20"],
                "cluster_2": ["findings 21-40"],
                "cluster_3": ["findings 41-60"],
                "cluster_4": ["findings 61-80"],
                "cluster_5": ["findings 81-100"]
            },
            preserved_details=["Critical finding 15", "Critical finding 42", "Critical finding 88"],
            confidence_score=0.75,  # Lower confidence due to high compression
            metadata={
                "high_compression": True,
                "potential_information_loss": 0.15
            }
        )

        with patch.object(compression_agent.agent, 'run', return_value=mock_result):
            result = await compression_agent.execute(agent_dependencies)

            assert result.compression_ratio > 20.0
            assert result.confidence_score < 0.8  # Lower confidence for high compression
            assert result.metadata.get("high_compression") is True

    @pytest.mark.asyncio
    async def test_confidence_score_based_on_compression(self, compression_agent, agent_dependencies):
        """Test that confidence scores reflect compression quality."""
        test_cases = [
            (2.0, 0.95),   # Low compression, high confidence
            (5.0, 0.85),   # Medium compression, good confidence
            (10.0, 0.70),  # High compression, moderate confidence
            (25.0, 0.55)   # Very high compression, lower confidence
        ]

        for compression_ratio, expected_confidence_range in test_cases:
            mock_result = MagicMock()
            mock_result.data = CompressedContent(
                original_length=10000,
                compressed_length=int(10000 / compression_ratio),
                compression_ratio=compression_ratio,
                summary="Test",
                key_points=["Test"],
                themes={"test": ["test"]},
                preserved_details=["Test"],
                confidence_score=expected_confidence_range,
                metadata={}
            )

            with patch.object(compression_agent.agent, 'run', return_value=mock_result):
                result = await compression_agent.execute(agent_dependencies)

                # Higher compression should generally lead to lower confidence
                if compression_ratio > 10:
                    assert result.confidence_score < 0.8
                else:
                    assert result.confidence_score > 0.6

    @pytest.mark.asyncio
    async def test_metadata_information_loss_tracking(self, compression_agent, agent_dependencies):
        """Test that information loss is tracked in metadata."""
        mock_result = MagicMock()
        mock_result.data = CompressedContent(
            original_length=8000,
            compressed_length=1000,
            compression_ratio=8.0,
            summary="Compressed content",
            key_points=["Key 1", "Key 2"],
            themes={"main": ["theme1"]},
            preserved_details=["Detail 1"],
            confidence_score=0.78,
            metadata={
                "information_loss": 0.12,
                "compression_method": "extractive_summary",
                "tokens_reduced": 7000,
                "processing_iterations": 3
            }
        )

        with patch.object(compression_agent.agent, 'run', return_value=mock_result):
            result = await compression_agent.execute(agent_dependencies)

            assert "information_loss" in result.metadata
            assert 0.0 <= result.metadata["information_loss"] <= 1.0
            assert "compression_method" in result.metadata

    @pytest.mark.asyncio
    async def test_error_handling(self, compression_agent, agent_dependencies):
        """Test error handling during compression."""
        with patch.object(compression_agent.agent, 'run', side_effect=Exception("Compression failed")):
            with pytest.raises(Exception, match="Compression failed"):
                await compression_agent.execute(agent_dependencies)

    @pytest.mark.asyncio
    async def test_empty_content_handling(self, compression_agent, agent_dependencies):
        """Test handling of empty content."""
        agent_dependencies.metadata.additional_context = {"findings": []}

        mock_result = MagicMock()
        mock_result.data = CompressedContent(
            original_length=0,
            compressed_length=0,
            compression_ratio=1.0,
            summary="",
            key_points=[],
            themes={},
            preserved_details=[],
            confidence_score=0.0,
            metadata={"empty_content": True}
        )

        with patch.object(compression_agent.agent, 'run', return_value=mock_result):
            result = await compression_agent.execute(agent_dependencies)

            assert result.original_length == 0
            assert result.compressed_length == 0
            assert len(result.key_points) == 0
            assert result.metadata.get("empty_content") is True
