"""Direct test of factory without going through main package imports."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
from typing import Optional, Dict, Any
from pydantic import BaseModel

# Direct imports to avoid circular dependencies
from open_deep_research_with_pydantic_ai.agents.factory import (
    AgentType,
    AgentFactory,
    AgentPoolConfig
)

from open_deep_research_with_pydantic_ai.models.dependencies import ResearchDependencies

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