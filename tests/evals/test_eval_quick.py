#!/usr/bin/env python
"""Quick test of multi-question clarification evaluation."""

import asyncio
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
_ = load_dotenv()

os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

import httpx
from src.agents.clarification import ClarificationAgent
from src.agents.base import ResearchDependencies
from src.models.api_models import APIKeys, ResearchMetadata
from src.models.core import ResearchState, ResearchStage
from pydantic import SecretStr

openai_api_key = os.getenv("OPENAI_API_KEY")
api_keys = APIKeys(openai=SecretStr(openai_api_key) if openai_api_key is not None else None)

async def quick_test():
    """Quick test of multi-question clarification."""

    agent = ClarificationAgent()

    test_cases = [
        {
            "query": "What is the current Bitcoin price?",
            "expected_clarification": False,
            "description": "Specific query - no clarification needed"
        },
        {
            "query": "I want to learn about machine learning",
            "expected_clarification": True,
            "min_questions": 2,
            "description": "Broad ML query - needs multiple clarifications"
        },
        {
            "query": "Help me design my system architecture",
            "expected_clarification": True,
            "min_questions": 3,
            "description": "Architecture query - needs detailed clarifications"
        },
        {
            "query": "Tell me about Python",
            "expected_clarification": True,
            "min_questions": 1,
            "description": "Ambiguous query - could be language or snake"
        },
    ]

    print("🚀 MULTI-QUESTION CLARIFICATION TEST")
    print("=" * 60)

    results = []

    async with httpx.AsyncClient() as http_client:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test_case['description']}")
            print(f"   Query: {test_case['query'][:60]}...")

            state = ResearchState(
                request_id=f"quick-test-{i}",
                user_id="test-user",
                session_id="quick-session",
                user_query=test_case['query'],
                current_stage=ResearchStage.CLARIFICATION,
                metadata=ResearchMetadata()
            )
            deps = ResearchDependencies(
                http_client=http_client,
                api_keys=api_keys,
                research_state=state,
                usage=None
            )

            try:
                result = await agent.agent.run(test_case['query'], deps=deps)
                output = result.output  # This will be a ClarifyWithUser object

                # Check if clarification expectation matches
                correct_decision = output.needs_clarification == test_case.get('expected_clarification', False)

                # Check question count if clarification is needed
                question_count_ok = True
                if output.needs_clarification and output.request:
                    num_questions = len(output.request.questions)
                    min_expected = test_case.get('min_questions', 1)
                    if isinstance(min_expected, int):
                        question_count_ok = num_questions >= min_expected
                    else:
                        question_count_ok = True

                    print(f"   ✅ Needs clarification: {num_questions} questions")

                    # Display first 3 questions
                    for j, q in enumerate(output.request.questions[:3], 1):
                        print(f"      {j}. {q.question[:60]}...")
                        print(f"         Type: {q.question_type}, Required: {q.is_required}")

                    if num_questions > 3:
                        print(f"      ... and {num_questions - 3} more questions")

                    # Check question types
                    question_types = set(q.question_type for q in output.request.questions)
                    print(f"   📊 Question types: {', '.join(question_types)}")

                    # Check required vs optional
                    required = sum(1 for q in output.request.questions if q.is_required)
                    optional = num_questions - required
                    print(f"   📌 Required: {required}, Optional: {optional}")

                    if not question_count_ok:
                        print(f"   ⚠️  Expected at least {min_expected} questions, got {num_questions}")
                else:
                    print(f"   ✅ No clarification needed - query is clear")
                    print(f"   Reasoning: {output.reasoning[:100]}...")

                # Overall result
                success = correct_decision and question_count_ok
                print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")

                results.append({
                    "test": test_case['description'],
                    "success": success,
                    "needs_clarification": output.needs_clarification,
                    "num_questions": len(output.request.questions) if output.request else 0
                })

            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    "test": test_case['description'],
                    "success": False,
                    "error": str(e)
                })

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results if r['success'])
    total = len(results)

    print(f"✅ Passed: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")

    # Multi-question statistics
    multi_q_results = [r for r in results if r.get('num_questions', 0) > 0]
    if multi_q_results:
        avg_questions = sum(r['num_questions'] for r in multi_q_results) / len(multi_q_results)
        print(f"\n📈 Multi-Question Statistics:")
        print(f"   Average questions: {avg_questions:.1f}")
        print(f"   Max questions: {max(r['num_questions'] for r in multi_q_results)}")
        print(f"   Min questions: {min(r['num_questions'] for r in multi_q_results)}")

    # Save results
    with open('quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to quick_test_results.json")

    print("\n" + "=" * 60)
    print("Test completed!")

    return successful == total


if __name__ == "__main__":
    asyncio.run(quick_test())
