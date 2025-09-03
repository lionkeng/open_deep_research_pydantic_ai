#!/usr/bin/env python
"""Minimal test to debug hanging issue."""

import asyncio
import os

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent

# Configure environment
load_dotenv()
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

print(f"API Key present: {bool(os.getenv('OPENAI_API_KEY'))}")


class SimpleOutput(BaseModel):
    answer: str


async def test_minimal():
    """Test minimal Pydantic AI agent."""

    agent = Agent(
        model="openai:gpt-4o-mini",
        output_type=SimpleOutput,
        system_prompt="You are a helpful assistant. Answer in one sentence.",
    )

    print("Running minimal agent...")

    try:
        result = await agent.run("What is 2+2?")
        print(f"Result: {result.output}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_minimal())
    print(f"Success: {success}")
    exit(0 if success else 1)
