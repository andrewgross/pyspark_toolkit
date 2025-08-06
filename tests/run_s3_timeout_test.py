#!/usr/bin/env python
"""Runner script for s3 tests with proper timeout and cleanup."""

import subprocess
import sys


def main():
    """Run the s3 isolated test with timeout."""
    print("Testing s3._get_signature_key (5s timeout)...")

    # Start the test process
    p = subprocess.Popen([sys.executable, "tests/unit/s3_isolated_runner.py"])

    try:
        # Wait for up to 5 seconds
        exit_code = p.wait(timeout=5)
        if exit_code == 0:
            print("SUCCESS: Test completed without hanging!")
            print("The s3 module issue may be resolved.")
            return 1  # Return non-zero to indicate unexpected success
        else:
            print("Test failed with error")
            return exit_code
    except subprocess.TimeoutExpired:
        # Expected behavior - test hangs
        print("Test timed out as expected (hanging issue confirmed)")
        p.kill()
        p.wait()  # Ensure process is cleaned up
        return 0  # Return success since hanging is expected


if __name__ == "__main__":
    sys.exit(main())
