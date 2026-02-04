#!/usr/bin/env python3
"""
Test runner for the agentic trading system.

This script runs the comprehensive test suite that addresses:
1. Unit tests for individual tools (most important)
2. Agent reasoning and tool usage tests
3. Golden path integration tests
4. Failure and resilience testing
"""

import sys
import os
import subprocess
import argparse


def run_tests(test_type=None, verbose=False):
    """Run the test suite."""

    # Add the current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Base test command
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    # Add specific test file
    test_file = os.path.join(os.path.dirname(__file__), "tests", "test_agentic_system.py")
    cmd.append(test_file)

    # Add test type filter if specified
    if test_type:
        cmd.extend(["-k", test_type])

    # Add coverage if available
    try:
        import coverage

        cmd.extend(["--cov=tradr", "--cov-report=term-missing"])
    except ImportError:
        print("Coverage not available. Install with: pip install coverage")

    print(f"Running tests with command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, check=True)
        print("=" * 60)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"❌ Tests failed with exit code {e.returncode}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run agentic trading system tests")
    parser.add_argument(
        "--type", choices=["tools", "agent", "integration", "resilience", "memory"], help="Run specific test type"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Map test types to pytest filters
    test_type_map = {
        "tools": "TestIndividualTools",
        "agent": "TestAgentReasoning",
        "integration": "TestGoldenPathIntegration",
        "resilience": "TestFailureAndResilience",
        "memory": "TestMemorySystem",
    }

    test_filter = test_type_map.get(args.type) if args.type else None

    print("🧪 Agentic Trading System Test Suite")
    print("=" * 60)

    if args.type:
        print(f"Running {args.type} tests...")
    else:
        print("Running all tests...")

    success = run_tests(test_filter, args.verbose)

    if success:
        print("\n🎉 Test suite completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test suite failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
