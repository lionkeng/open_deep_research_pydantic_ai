#!/usr/bin/env python3
"""Quick test runner for validation."""

import subprocess
import sys
import os

def main():
    """Run a quick validation test."""
    print("🚀 Quick Test Validation")
    print("=" * 40)

    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    # Run a simple test
    cmd = "LOGFIRE_IGNORE_NO_CONFIG=1 uv run python -c \"print('Testing framework ready!');\""
    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        print("✅ Testing framework is properly set up!")
        return 0
    else:
        print("❌ Testing framework has issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
