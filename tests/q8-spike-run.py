"""Retain five fresh-process Q8_0 trials and their aggregate medians."""
import argparse
import collections
import datetime
import os
import pathlib
import statistics
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parent.parent
environment = dict(os.environ, VECLIB_MAXIMUM_THREADS="1")
environment.pop("Q8_CHECK_ONLY", None)
results = collections.defaultdict(list)
repack = collections.defaultdict(list)

with args.output.open("w") as log:
    log.write(f"UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
    for command in (["git", "describe", "--always", "--dirty"],
                    ["sw_vers"], ["sysctl", "-n", "machdep.cpu.brand_string"],
                    ["xcrun", "clang", "--version"]):
        log.write(subprocess.check_output(command, cwd=root, text=True))
    cache = (root / "build/CMakeCache.txt").read_text()
    flags = (root / "build/CMakeFiles/q8_spike.dir/flags.make").read_text()
    log.write("\n".join(line for line in cache.splitlines()
                        if line.startswith(("CMAKE_C_COMPILER:", "CMAKE_BUILD_TYPE:", "BUILD_Q8_SPIKE:"))))
    log.write("\n" + flags + "\n")
    log.write("Warm repeated matvec; one calling thread; VECLIB_MAXIMUM_THREADS=1.\n"
              "Accelerate internal worker count is not observed.\n"
              "Repack includes validation, extraction and f16 conversion into reused arrays.\n"
              "Each process rotates timing order; helper uses >=50ms calibration and three-batch median.\n")
    log.flush()
    for trial in range(5):
        process = subprocess.run(
            ["sbcl", "--dynamic-space-size", "4096", "--script", "tests/q8-spike.lisp"],
            cwd=root, env=dict(environment, Q8_TRIAL=str(trial)), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = process.stdout
        log.write(f"\nPROCESS {trial}\n{output}")
        log.flush()
        process.check_returncode()
        for line in output.splitlines():
            fields = line.split()
            if len(fields) == 8 and fields[0] == "RESULT" and fields[1].isdigit():
                _, _, storage, rows, width, method, us, _ = fields
                results[(storage, int(rows), int(width), method)].append(float(us))
            elif fields and fields[0] == "REPACK":
                _, rows, width, us, *_ = fields
                repack[(int(rows), int(width))].append(float(us))
        print(f"Trial {trial + 1}/5 passed", flush=True)
    medians = {key: statistics.median(values) for key, values in results.items()}
    assert len(medians) == 16
    log.write("\nSUMMARY columns: storage rows width method median-us min-us max-us effective-GFLOP/s kernel/C-throughput\n")
    for key, median in sorted(medians.items()):
        storage, rows, width, method = key
        values = results[key]
        assert len(values) == 5
        ratio = medians[(storage, rows, width, "NEON")] / medians[(storage, rows, width, "KERNEL")]
        line = (f"SUMMARY {storage} {rows} {width} {method} {median:.6f} "
                f"{min(values):.6f} {max(values):.6f} {2 * rows * width / (1000 * median):.6f} "
                f"{ratio:.6f}\n")
        log.write(line)
        print(line, end="")
    for (rows, width), values in sorted(repack.items()):
        log.write(f"REPACK-MEDIAN {rows} {width} {statistics.median(values):.6f} us; samples={len(values)}\n")
