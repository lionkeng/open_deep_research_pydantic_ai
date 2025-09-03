#!/usr/bin/env python
"""Direct test of ClarificationAgent."""

import asyncio
import os

from dotenv import load_dotenv
from pydantic_ai import Agent

from src.agents.clarification import ClarifyWithUser

# Configure environment
load_dotenv()
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"


async def test_direct():
    """Test direct agent creation."""

    print("Creating agent directly...")

    agent = Agent(
        model="openai:gpt-4o-mini",
        output_type=ClarifyWithUser,
        system_prompt="You analyze queries and determine if clarification is needed.",
    )

    print("Running agent...")

    try:
        result = await agent.run("I want to learn about machine learning")
        output = result.output
        print("✅ Got response!")
        print(f"Needs clarification: {output.needs_clarification}")
        if output.request:
            print(f"Number of questions: {len(output.request.questions)}")
            for i, q in enumerate(output.request.questions[:3], 1):
                print(f"  {i}. {q.question[:60]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_direct())
    exit(0 if success else 1)
