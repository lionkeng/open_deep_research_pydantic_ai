#!/usr/bin/env python
"""Simple test of multi-question clarification."""

import asyncio
import os

import httpx
from pydantic import SecretStr

from src.agents.base import ResearchDependencies
from src.agents.clarification import ClarificationAgent
from src.models.api_models import APIKeys, ResearchMetadata
from src.models.core import ResearchStage, ResearchState

# Configure environment
# Note: .env is loaded automatically when importing from src
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"


async def test_simple():
    """Single simple test."""

    # Create agent
    agent = ClarificationAgent()

    # Setup
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OPENAI_API_KEY found")
        return False

    api_keys = APIKeys(openai=SecretStr(api_key))

    # Test query
    query = "I want to learn about machine learning"

    print(f"🚀 Testing clarification for: {query}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            state = ResearchState(
                request_id="simple-test",
                user_id="test-user",
                session_id="test-session",
                user_query=query,
                current_stage=ResearchStage.CLARIFICATION,
                metadata=ResearchMetadata(),
            )

            deps = ResearchDependencies(
                http_client=http_client, api_keys=api_keys, research_state=state, usage=None
            )

            print("📡 Calling agent...")
            result = await agent.agent.run(query, deps=deps)

            print("✅ Got response!")
            output = result.output

            if output.needs_clarification and output.request:
                print(f"💭 Needs clarification with {len(output.request.questions)} questions:")
                for i, q in enumerate(output.request.questions[:3], 1):
                    print(f"   {i}. {q.question}")
                    print(f"      Type: {q.question_type}, Required: {q.is_required}")
            else:
                print("✅ No clarification needed")
                print(f"   Reasoning: {output.reasoning[:100]}...")

            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_simple())
    exit(0 if success else 1)
