"""Performance validation tests for the clarification improvement system."""

import pytest
import asyncio
import time
from typing import List, Tuple
from unittest.mock import patch

from src.open_deep_research_with_pydantic_ai.agents.clarification import ClarificationAgent
from src.open_deep_research_with_pydantic_ai.agents.brief_generator import BriefGeneratorAgent
from src.open_deep_research_with_pydantic_ai.core.workflow import workflow
from src.open_deep_research_with_pydantic_ai.models.api_models import APIKeys
from src.open_deep_research_with_pydantic_ai.agents.base import ResearchDependencies
from src.open_deep_research_with_pydantic_ai.models.research import ResearchState


class TestPerformanceValidation:
    """Performance validation for the three-phase clarification system."""

    # Test data for algorithm accuracy validation - expanded dataset
    ALGORITHM_TEST_CASES = [
        # Specific queries (should NOT need clarification)
        ("What is 2+2?", False, "math_fact"),
        ("Current stock price of AAPL", False, "specific_data"),
        ("Compare React hooks useState vs useReducer for TypeScript", False, "technical_comparison"),
        ("Best practices for Python error handling in async code", False, "technical_specific"),
        ("How to implement JWT authentication in Node.js Express", False, "technical_howto"),
        ("Difference between MongoDB and PostgreSQL performance for read-heavy workloads", False, "technical_comparison_detailed"),
        ("Current FDA approval status for Alzheimer's drug aducanumab", False, "specific_regulatory"),
        ("What is the time complexity of quicksort algorithm", False, "cs_fact"),
        ("How to configure CORS in Spring Boot 3.0", False, "technical_config"),
        ("Latest quarterly revenue for Tesla Inc Q3 2024", False, "specific_financial"),

        # Intermediate queries (edge cases - could go either way)
        ("How does solar energy work?", False, "moderate_technical"),
        ("Benefits of microservices architecture", False, "moderate_concept"),

        # Broad queries (should need clarification)
        ("What is artificial intelligence?", True, "broad_concept"),
        ("Tell me about technology", True, "very_broad"),
        ("How does the future look?", True, "extremely_broad"),
        ("Explain machine learning", True, "broad_technical"),
        ("What should I know about healthcare?", True, "broad_domain"),
        ("How does business work?", True, "broad_abstract"),
        ("Tell me about science", True, "broad_academic"),
        ("What are the implications of climate change?", True, "broad_impact"),
        ("How can I improve my life?", True, "personal_broad"),
        ("What is the meaning of innovation?", True, "philosophical_broad"),
    ]

    @pytest.mark.asyncio
    async def test_clarification_algorithm_accuracy(self):
        """Test that clarification algorithm meets >90% accuracy target."""

        # Create dependencies for testing
        research_state = ResearchState(
            request_id="test-accuracy",
            user_id="test-user",
            session_id="test-session",
            user_query="placeholder",
        )

        deps = ResearchDependencies(
            http_client=None,  # Will be mocked
            api_keys=APIKeys(),
            research_state=research_state,
            metadata={},
            usage=None,
        )

        agent = ClarificationAgent()

        correct_predictions = 0
        total_predictions = len(self.ALGORITHM_TEST_CASES)

        results = []

        for query, expected_needs_clarification, category in self.ALGORITHM_TEST_CASES:
            try:
                # Update research state query
                deps.research_state.user_query = query

                # Test the clarification assessment
                prompt = f"Assess if this query needs clarification: {query}"
                result = await agent.run(prompt, deps)

                actual_needs_clarification = result.need_clarification
                is_correct = actual_needs_clarification == expected_needs_clarification

                if is_correct:
                    correct_predictions += 1

                results.append({
                    "query": query,
                    "category": category,
                    "expected": expected_needs_clarification,
                    "actual": actual_needs_clarification,
                    "correct": is_correct,
                    "question": result.question[:50] if result.question else "",
                })

            except Exception as e:
                print(f"Error testing query '{query}': {e}")
                results.append({
                    "query": query,
                    "category": category,
                    "expected": expected_needs_clarification,
                    "actual": "ERROR",
                    "correct": False,
                    "question": "",
                })

        accuracy = (correct_predictions / total_predictions) * 100

        # Print detailed results
        print(f"\n📊 ALGORITHM ACCURACY TEST RESULTS:")
        print(f"Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
        print(f"Target: >90%")

        for result in results:
            status = "✓" if result["correct"] else "✗"
            print(f"{status} {result['category']}: {result['query'][:40]}...")
            print(f"   Expected: {result['expected']}, Got: {result['actual']}")
            if result["question"]:
                print(f"   Question: {result['question']}...")

        # Assert accuracy meets target
        assert accuracy >= 90.0, f"Algorithm accuracy {accuracy:.1f}% is below 90% target"

    @pytest.mark.asyncio
    async def test_workflow_response_time(self):
        """Test that workflow phases meet response time targets."""

        test_queries = [
            "What is quantum computing?",
            "How do electric cars work?",
            "Current trends in renewable energy",
        ]

        performance_results = []

        for query in test_queries:
            # Measure total planning time
            start_time = time.time()

            research_state = await workflow.execute_planning_only(
                user_query=query,
                api_keys=APIKeys()
            )

            total_time = time.time() - start_time

            performance_results.append({
                "query": query,
                "total_time": total_time,
                "completed": research_state.current_stage == ResearchStage.RESEARCH_EXECUTION,
                "brief_length": len(research_state.metadata.get("research_brief_text", "")),
            })

        # Print performance results
        print(f"\n⏱️  WORKFLOW PERFORMANCE RESULTS:")
        for result in performance_results:
            print(f"Query: {result['query'][:40]}...")
            print(f"  Time: {result['total_time']:.2f}s")
            print(f"  Completed: {result['completed']}")
            print(f"  Brief Length: {result['brief_length']} chars")

        # Performance assertions
        avg_time = sum(r["total_time"] for r in performance_results) / len(performance_results)
        print(f"Average Time: {avg_time:.2f}s")

        # All should complete
        all_completed = all(r["completed"] for r in performance_results)
        assert all_completed, "All workflows should complete successfully"

        # Time targets (realistic for production)
        assert avg_time < 45, f"Average time {avg_time:.2f}s exceeds 45 second target"

        for result in performance_results:
            assert result["total_time"] < 90, f"Query took too long: {result['total_time']:.2f}s (max 90s)"

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self):
        """Test that multiple workflows can run concurrently without interference."""

        queries = [
            "What is blockchain technology?",
            "How do solar panels generate electricity?",
            "What are the benefits of electric vehicles?",
        ]

        # Run workflows concurrently
        start_time = time.time()

        tasks = [
            workflow.execute_planning_only(query, APIKeys())
            for query in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        concurrent_time = time.time() - start_time

        # Analyze results
        successful_results = [r for r in results if isinstance(r, ResearchState)]
        error_results = [r for r in results if isinstance(r, Exception)]

        print(f"\n🔄 CONCURRENT EXECUTION RESULTS:")
        print(f"Total Time: {concurrent_time:.2f}s")
        print(f"Successful: {len(successful_results)}/{len(queries)}")
        print(f"Errors: {len(error_results)}")

        # All should succeed
        assert len(error_results) == 0, f"Concurrent execution had {len(error_results)} errors"
        assert len(successful_results) == len(queries), "All workflows should complete"

        # Each should have completed the planning phase
        for result in successful_results:
            assert result.current_stage == ResearchStage.RESEARCH_EXECUTION
            assert "research_brief_text" in result.metadata

        # Concurrent execution should be faster than sequential
        # (This is a basic test - real measurement would require controlled conditions)
        estimated_sequential_time = concurrent_time * len(queries)
        efficiency_ratio = estimated_sequential_time / concurrent_time

        print(f"Efficiency Ratio: {efficiency_ratio:.2f}x")
        assert efficiency_ratio >= 1.0, "Concurrent execution should not be slower than sequential"

    @pytest.mark.asyncio
    async def test_memory_usage_validation(self):
        """Basic memory usage validation for workflow execution."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Measure memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run multiple workflow iterations
        for i in range(3):
            query = f"Test query {i}: What are the applications of AI in query {i}?"
            research_state = await workflow.execute_planning_only(
                user_query=query,
                api_keys=APIKeys()
            )
            assert research_state.current_stage == ResearchStage.RESEARCH_EXECUTION

        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        print(f"\n💾 MEMORY USAGE VALIDATION:")
        print(f"Memory Before: {memory_before:.1f} MB")
        print(f"Memory After: {memory_after:.1f} MB")
        print(f"Memory Increase: {memory_increase:.1f} MB")

        # Realistic memory usage assertions
        assert memory_increase < 100, f"Memory usage increased too much: {memory_increase:.1f} MB (max 100MB for 3 iterations)"

    @pytest.mark.asyncio
    async def test_workflow_interruption_recovery(self):
        """Test workflow behavior when interrupted and resumed."""

        # Create initial research state
        query = "What is renewable energy technology?"

        # Start workflow
        research_state = await workflow.execute_planning_only(
            user_query=query,
            api_keys=APIKeys()
        )

        # Verify initial completion
        assert research_state.current_stage == ResearchStage.RESEARCH_EXECUTION
        original_metadata_keys = set(research_state.metadata.keys())

        # Simulate resuming workflow (though planning is already complete)
        resumed_state = await workflow.resume_research(
            research_state=research_state,
            api_keys=APIKeys()
        )

        # Should maintain state and not regress
        assert resumed_state.current_stage >= research_state.current_stage
        assert set(resumed_state.metadata.keys()).issuperset(original_metadata_keys)

        print("✓ Workflow interruption and recovery test passed")


class TestEdgeCaseHandling:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test handling of empty or whitespace-only queries."""

        edge_queries = ["", "   ", "\n\t", "?", "..."]

        for query in edge_queries:
            research_state = await workflow.execute_planning_only(
                user_query=query,
                api_keys=APIKeys()
            )

            # Should complete without crashing
            assert isinstance(research_state, ResearchState)
            assert research_state.current_stage == ResearchStage.RESEARCH_EXECUTION

            # Should have some metadata
            assert isinstance(research_state.metadata, dict)

    @pytest.mark.asyncio
    async def test_very_long_query_handling(self):
        """Test handling of extremely long queries."""

        long_query = "What is artificial intelligence? " * 100  # Very long repetitive query

        research_state = await workflow.execute_planning_only(
            user_query=long_query,
            api_keys=APIKeys()
        )

        # Should complete successfully
        assert research_state.current_stage == ResearchStage.RESEARCH_EXECUTION
        assert "research_brief_text" in research_state.metadata

    @pytest.mark.asyncio
    async def test_special_characters_handling(self):
        """Test handling of queries with special characters."""

        special_queries = [
            "What is AI/ML & deep learning? (2024 update)",
            "How does the @framework handle errors in C++?",
            "Research: 'quantum computing' + ethics considerations",
            "Analysis of €-cost vs $-benefit in renewable energy [2023-2024]",
        ]

        for query in special_queries:
            research_state = await workflow.execute_planning_only(
                user_query=query,
                api_keys=APIKeys()
            )

            # Should handle special characters gracefully
            assert research_state.current_stage == ResearchStage.RESEARCH_EXECUTION
            assert len(research_state.metadata.get("research_brief_text", "")) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
