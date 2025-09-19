#!/usr/bin/env python3
"""Test runner script for the two-phase clarification system."""

import subprocess
import sys
import time
import argparse
import os
import shlex
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🧪 {description}")
    print(f"Command: {cmd}")
    print("-" * 60)

    # Parse command safely by splitting on spaces
    # Handle common prefixes safely
    cmd_parts = []
    if cmd.startswith("LOGFIRE_IGNORE_NO_CONFIG=1 "):
        # Set environment variable
        env = {**os.environ, "LOGFIRE_IGNORE_NO_CONFIG": "1"}
        cmd = cmd[len("LOGFIRE_IGNORE_NO_CONFIG=1 "):]
    else:
        env = None

    # Split command into arguments safely
    import shlex
    try:
        cmd_parts = shlex.split(cmd)
    except ValueError:
        # Fallback for malformed commands
        print(f"❌ {description} - FAILED (Invalid command format)")
        return False

    start_time = time.time()
    try:
        result = subprocess.run(cmd_parts, capture_output=False, env=env)
        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {description} - PASSED ({execution_time:.1f}s)")
            return True
        else:
            print(f"❌ {description} - FAILED ({execution_time:.1f}s)")
            return False
    except FileNotFoundError:
        execution_time = time.time() - start_time
        print(f"❌ {description} - FAILED (Command not found) ({execution_time:.1f}s)")
        return False
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ {description} - FAILED ({e}) ({execution_time:.1f}s)")
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run two-phase clarification system tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Always run from project root with tests/ prefix
    test_prefix = "LOGFIRE_IGNORE_NO_CONFIG=1 uv run python -m pytest tests/"

    verbose_flag = "-v" if args.verbose else ""

    print("🚀 Two-phase Clarification System Test Runner")
    print("=" * 60)

    total_tests = 0
    passed_tests = 0

    if args.quick:
        # Quick smoke tests
        tests = [
            (f"{test_prefix} test_new_agents.py::TestClarificationAgent::test_agent_initialization {verbose_flag}",
             "Quick: Agent Initialization"),
            (f"python -c 'from tests.conftest import MockWorkflowComponents; print(\"✓ Mock components imported successfully\")'",
             "Quick: Mock Components Validation"),
        ]

    elif args.integration:
        # Integration tests only
        tests = [
            (f"{test_prefix} test_two_phase_integration.py {verbose_flag}",
             "Integration: Two-phase Workflow"),
        ]

    elif args.performance:
        # Performance tests only
        tests = [
            (f"{test_prefix} test_performance_validation.py::TestPerformanceValidation::test_workflow_response_time {verbose_flag}",
             "Performance: Response Time"),
            (f"{test_prefix} test_performance_validation.py::TestPerformanceValidation::test_memory_usage_validation {verbose_flag}",
             "Performance: Memory Usage"),
        ]

    elif args.accuracy:
        # Accuracy tests only
        tests = [
            (f"{test_prefix} test_performance_validation.py::TestPerformanceValidation::test_clarification_algorithm_accuracy {verbose_flag}",
             "Accuracy: Algorithm Validation"),
        ]

    else:
        # Full test suite
        tests = [
            # Quick validation tests
            (f"{test_prefix} test_new_agents.py::TestClarificationAgent::test_agent_initialization -q",
             "Validation: Agent Initialization"),

            # Core integration tests
            (f"{test_prefix} test_two_phase_integration.py::TestTwoPhaseIntegration::test_specific_query_minimal_processing {verbose_flag}",
             "Integration: Specific Query Processing"),

            (f"{test_prefix} test_two_phase_integration.py::TestTwoPhaseIntegration::test_metadata_schema_consistency {verbose_flag}",
             "Integration: Metadata Schema"),

            # Performance validation
            (f"{test_prefix} test_performance_validation.py::TestPerformanceValidation::test_workflow_response_time {verbose_flag}",
             "Performance: Response Time"),

            # Edge case validation
            (f"{test_prefix} test_performance_validation.py::TestEdgeCaseHandling::test_empty_query_handling {verbose_flag}",
             "Edge Cases: Empty Query Handling"),
        ]

    # Run all tests
    for cmd, description in tests:
        total_tests += 1
        if run_command(cmd, description):
            passed_tests += 1
        else:
            if not args.verbose:
                print("💡 Tip: Use --verbose flag for detailed error output")

    # Summary
    print("\n" + "=" * 60)
    print(f"📊 TEST SUMMARY")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
