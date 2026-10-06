#!/usr/bin/env python3
"""Retain bounded serial and simultaneous CCL/Rosetta diagnostic trials."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--ccl", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--cache-seed", type=Path)
    parser.add_argument("--serial-trials", type=int, default=20)
    parser.add_argument("--paired-trials", type=int, default=20)
    parser.add_argument("--full-trials", type=int, default=3)
    parser.add_argument("--background-checkout", type=Path)
    parser.add_argument("--background-revision")
    args = parser.parse_args()
    root = args.checkout.resolve()
    background = args.background_checkout.resolve() if args.background_checkout else root
    logs = args.log_dir.resolve()
    if min(args.serial_trials, args.paired_trials, args.full_trials) < 0:
        parser.error("trial counts must be nonnegative")
    if args.background_checkout and not args.background_revision:
        parser.error("provide the background checkout's revision")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("run on an ARM64 Mac with Rosetta")
    if not (root / "build/libtrivial_simd.dylib").is_file() or not args.ccl.is_file():
        parser.error("provide a built x86-64 checkout and a CCL executable")
    if logs.exists():
        parser.error("use a new log directory to preserve earlier trials")
    if args.cache_seed and not args.cache_seed.is_dir():
        parser.error("cache seed must be an existing ASDF cache directory")
    library = subprocess.check_output(["file", str(root / "build/libtrivial_simd.dylib")], text=True)
    if "x86_64" not in library:
        parser.error("the native library must be built for x86-64")
    background_library = subprocess.check_output(["file", str(background / "build/libtrivial_simd.dylib")], text=True)
    if "x86_64" not in background_library:
        parser.error("the background native library must be built for x86-64")
    logs.mkdir(parents=True)
    command = ["arch", "-x86_64", str(args.ccl.resolve()), "--no-init", "--batch",
               "--eval", '(format t "~&CCL ~A architecture=~A~%" (lisp-implementation-version) (machine-type))',
               "--load", "tests/run.lisp", "--eval", "(quit)"]
    metadata = {"checkout": str(root), "command": command, "revision": args.revision,
                "native_library": library,
                "serial_trials": args.serial_trials, "paired_trials": args.paired_trials,
                "full_trials": args.full_trials,
                "background_checkout": str(background),
                "background_revision": args.background_revision or args.revision,
                "background_native_library": background_library,
                "uname": subprocess.check_output(["uname", "-a"], text=True),
                "macos": subprocess.check_output(["sw_vers"], text=True),
                "rosetta": subprocess.run(["pkgutil", "--pkg-info", "com.apple.pkg.RosettaUpdateAuto"],
                                          capture_output=True, text=True).stdout}
    (logs / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.cache_seed:
        shutil.copytree(args.cache_seed, logs / "cache-serial")
    records = []

    def trial(name, lane, *, full=False, warm=False, seconds=300):
        checkout = background if lane == "full" else root
        env = os.environ.copy()
        env["ASDF_OUTPUT_TRANSLATIONS"] = (
            f'(:output-translations (t ("{logs}/cache-{lane}/" :implementation)) '
            ':ignore-inherited-configuration)')
        env["TRIVIAL_SIMD_TEST_TIMING"] = "1"
        env["TRIVIAL_SIMD_TEST_RUN"] = name
        env["TRIVIAL_SIMD_BACKEND"] = "native"
        if full:
            env.pop("TRIVIAL_SIMD_TESTS", None)
        else:
            env["TRIVIAL_SIMD_TESTS"] = ("complex-native-constant-ownership" if warm else
                "complex-mask-spilling-and-concurrent-calls complex-native-constant-ownership")
        started = time.monotonic()
        wrapper = ["python3", str(checkout / "tests/bounded-run.py"), "--seconds", str(seconds),
                   "--log", str(logs / f"{name}.log"), "--", *command]
        process = subprocess.Popen(wrapper, cwd=checkout, env=env)
        status = process.wait()
        return {"name": name, "lane": lane, "full": full, "wrapper_pid": process.pid,
                "checkout": str(checkout),
                "revision": (args.background_revision or args.revision) if lane == "full" else args.revision,
                "status": status, "seconds": round(time.monotonic() - started, 3)}

    def retain(results):
        records.extend(results)
        (logs / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        for result in results:
            print(json.dumps(result), flush=True)

    warm = trial("warm-serial", "serial", warm=True, seconds=900)
    retain([warm])
    if warm["status"]:
        raise SystemExit("warm-up failed; inspect the retained log")
    for lane in ("pair-a", "pair-b", "target", "full"):
        shutil.copytree(logs / "cache-serial", logs / f"cache-{lane}")
        warm = trial("warm-" + lane, lane, warm=True,
                     seconds=900 if lane == "full" and background != root else 300)
        retain([warm])
        if warm["status"]:
            raise SystemExit("lane warm-up failed; inspect the retained log")
    for index in range(args.serial_trials):
        retain([trial(f"serial-{index + 1:02}", "serial")])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for index in range(args.paired_trials):
            futures = [pool.submit(trial, f"pair-{index + 1:02}-{lane}", "pair-" + lane)
                       for lane in ("a", "b")]
            retain([future.result() for future in futures])
        for index in range(args.full_trials):
            background_trial = pool.submit(trial, f"full-{index + 1}", "full", full=True, seconds=900)
            time.sleep(3)
            targeted = trial(f"with-full-{index + 1}", "target")
            retain([targeted, background_trial.result()])


if __name__ == "__main__":
    main()
