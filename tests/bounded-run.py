#!/usr/bin/env python3
"""Run a diagnostic command with a wall-clock bound and retained output."""
import argparse
import os
import signal
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seconds", type=float, default=300)
parser.add_argument("--log", required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if args.seconds <= 0 or not command:
    parser.error("use a positive timeout and a command after --")
with open(args.log, "w") as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        status = process.wait(timeout=args.seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as condition:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        status = 124 if isinstance(condition, subprocess.TimeoutExpired) else 130
        log.write(f"\nDiagnostic stopped after {args.seconds}s; exit={status}\n")
print(f"Diagnostic exit={status}; log={args.log}")
sys.exit(status if status >= 0 else 128 - status)
