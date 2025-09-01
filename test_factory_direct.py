"""Direct test of factory without going through main package imports."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


# Direct imports to avoid circular dependencies
from agents.factory import AgentFactory, AgentPoolConfig, AgentType
from models.dependencies import ResearchDependencies

print("✓ All imports successful")

# Test creation
factory = AgentFactory()
print("✓ Factory created")

# Test enum
agent_type = AgentType.RESEARCH_EXECUTOR
print(f"✓ Agent type: {agent_type}")

# Test pool config
pool_config = AgentPoolConfig(max_size=5)
print(f"✓ Pool config: {pool_config}")

# Test dependencies
deps = ResearchDependencies()
print(f"✓ Dependencies: {deps}")

print("🎉 Factory system is working correctly!")
print(f"Available agent types: {[t.value for t in AgentType]}")
