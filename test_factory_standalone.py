"""Standalone test for the factory system to verify it works independently."""

import asyncio
from typing import Optional, Dict, Any
from pydantic import BaseModel
from pydantic_ai import Agent

# Import our factory system
from src.open_deep_research_with_pydantic_ai.agents.factory import (
    AgentType,
    AgentFactory,
    AgentPoolConfig,
    register_agent_type
)

# Import the dependencies model directly
from src.open_deep_research_with_pydantic_ai.models.dependencies import ResearchDependencies

# Create a minimal base agent for testing
class TestAgentConfiguration(BaseModel):
    """Test configuration model."""
    agent_name: str = "test_agent"
    agent_type: str = "test"


class TestOutput(BaseModel):
    """Test output model."""
    result: str
    confidence: float = 0.95


class MockBaseResearchAgent:
    """Mock base agent for testing factory functionality."""
    
    def __init__(self, config: TestAgentConfiguration, dependencies: ResearchDependencies):
        self.config = config
        self.dependencies = dependencies
        print(f"Created agent: {config.agent_name}")
    
    def __repr__(self):
        return f"MockAgent(name={self.config.agent_name})"


async def test_factory_standalone():
    """Test the factory system independently."""
    print("Testing Agent Factory System Standalone")
    print("=" * 50)
    
    # Create factory
    factory = AgentFactory()
    print("✓ Factory created")
    
    # Create pool configuration
    pool_config = AgentPoolConfig(
        max_size=3,
        initial_size=1,
        enable_monitoring=True
    )
    print("✓ Pool config created")
    
    # Create test configuration
    test_config = TestAgentConfiguration()
    print("✓ Test config created")
    
    # Register agent type
    factory.register_agent_type(
        AgentType.RESEARCH_EXECUTOR,
        MockBaseResearchAgent,
        default_config=test_config,
        pool_config=pool_config
    )
    print("✓ Agent type registered")
    
    # Get factory stats
    stats = factory.get_factory_stats()
    print(f"✓ Factory stats: {stats['registry_stats']['total_registered']} registered types")
    
    # Test creating dependencies
    dependencies = ResearchDependencies()
    print("✓ Dependencies created")
    
    # Test agent creation
    agent = await factory.create_agent(
        AgentType.RESEARCH_EXECUTOR,
        dependencies,
        use_pool=False
    )
    print(f"✓ Agent created: {agent}")
    
    # Test batch creation
    specs = [
        {"type": "research_executor"},
        {"type": "research_executor"},
    ]
    
    batch_agents = await factory.create_agents_batch(specs, dependencies, parallel=False)
    print(f"✓ Batch creation: {len(batch_agents)} agents")
    
    # Test pool functionality
    pooled_agent = await factory.create_agent(
        AgentType.RESEARCH_EXECUTOR,
        dependencies,
        use_pool=True
    )
    print(f"✓ Pooled agent created: {pooled_agent}")
    
    # Return agent to pool
    await factory.return_agent_to_pool(pooled_agent, AgentType.RESEARCH_EXECUTOR)
    print("✓ Agent returned to pool")
    
    # Get final stats
    final_stats = factory.get_factory_stats()
    print(f"✓ Final stats: {final_stats['global_stats']}")
    
    # Shutdown
    await factory.shutdown()
    print("✓ Factory shut down")
    
    print("\n" + "=" * 50)
    print("All factory tests passed!")


if __name__ == "__main__":
    asyncio.run(test_factory_standalone())