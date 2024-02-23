#!env/bin/python3
"""Script to run all test cases in the random order."""

import os
import signal
import subprocess


def main():
    """Run all test cases in the random order."""
    # Run the test cases in the random order
    test_proc = subprocess.Popen(
        "./startup/challenge/start.sh",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        preexec_fn=os.setsid,
    )

    # Wait for the test cases to finish
    try:
        test_proc.wait()
    except KeyboardInterrupt:
        os.killpg(os.getpgid(test_proc.pid), signal.SIGTERM)


if __name__ == "__main__":
    main()
