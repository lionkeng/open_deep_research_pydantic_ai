#!/usr/bin/env python
"""Quick test of evaluation framework with minimal cases."""

import asyncio
import os
import sys
from pathlib import Path

from pydantic_ai.run import AgentRunResult

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
_ = load_dotenv()

os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

import httpx
from src.agents.clarification import ClarificationAgent
from src.agents.base import ResearchDependencies
from src.models.api_models import APIKeys
from src.models.core import ResearchState
from pydantic import SecretStr
from src.agents.clarification import ClarifyWithUser
from typing import Any

openai_api_key = os.getenv("OPENAI_API_KEY")
api_keys = APIKeys(openai=SecretStr(openai_api_key) if openai_api_key is not None else None)

async def quick_test():
    """Quick test with just a few cases."""

    agent = ClarificationAgent()

    test_cases = [
        ("What is the current Bitcoin price?", False, "Specific query"),
        ("What is AI?", True, "Broad query"),
        ("Tell me about Python", True, "Ambiguous query"),
    ]

    print("QUICK CLARIFICATION TEST")
    print("=" * 40)

    for query, expected, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Query: {query}")

        async with httpx.AsyncClient() as http_client:
            state = ResearchState(
                request_id=f"quick-test",
                user_id="test-user",
                session_id="quick-session",  # Added session_id argument
                user_query=query
            )
            deps = ResearchDependencies(
                http_client=http_client,
                api_keys=api_keys,
                research_state=state
            )

            try:
                result: AgentRunResult[ClarifyWithUser | dict[Any, Any]] = await agent.agent.run(deps=deps)
                # print(f"Raw result: {result}")
                if isinstance(result.output, dict):
                    output = ClarifyWithUser(**result.output)
                else:
                    output = result.output
                correct = output.need_clarification == expected
                print(f"Expected clarification: {expected}")
                print(f"Got clarification: {output.need_clarification}")
                print(f"Result: {'✅ PASS' if correct else '❌ FAIL'}")

                if output.need_clarification:
                    print(f"Question: {output.question}")
                    print(f"Dimensions: {', '.join(output.missing_dimensions)}")
                else:
                    print(f"Verification: {output.verification[:100]}...")

            except Exception as e:
                print(f"❌ Error: {e}")

    print("\n" + "=" * 40)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(quick_test())
