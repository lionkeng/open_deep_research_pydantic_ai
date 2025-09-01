#!/usr/bin/env python
"""Test script to verify CLI dual-mode functionality."""

import asyncio
import sys

from cli import HTTPResearchClient, _http_mode_available


def test_http_mode_availability():
    """Test if HTTP mode dependencies are available."""
    if _http_mode_available:
        print("✓ HTTP mode is available (httpx-sse installed)")
        return True
    else:
        print("✗ HTTP mode is NOT available (httpx-sse not installed)")
        return False


async def test_http_client_init():
    """Test HTTPResearchClient initialization."""
    if not _http_mode_available:
        print("Skipping HTTP client test - dependencies not available")
        return False

    try:
        client = HTTPResearchClient("http://localhost:8000")
        print("✓ HTTPResearchClient initialized successfully")
        await client.close()
        return True
    except Exception as e:
        print(f"✗ HTTPResearchClient initialization failed: {e}")
        return False


async def test_cli_help():
    """Test CLI help command."""
    import subprocess

    result = subprocess.run(
        ["uv", "run", "deep-research", "research", "--help"], capture_output=True, text=True
    )

    if "--mode" in result.stdout and "--server-url" in result.stdout:
        print("✓ CLI has --mode and --server-url options")
        return True
    else:
        print("✗ CLI missing dual-mode options")
        print(f"Output: {result.stdout}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing CLI Dual-Mode Implementation")
    print("=" * 60)

    tests = [
        ("HTTP Mode Availability", test_http_mode_availability()),
        ("HTTP Client Initialization", test_http_client_init()),
        ("CLI Help Options", test_cli_help()),
    ]

    results = []
    for name, coro in tests:
        print(f"\nTesting: {name}")
        if asyncio.iscoroutine(coro):
            result = await coro
        else:
            result = coro
        results.append(result)

    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed! CLI dual-mode implementation is working.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
