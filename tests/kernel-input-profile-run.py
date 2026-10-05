"""Compare declared-input preparation and VM costs in five fresh processes."""
import argparse
import collections
import datetime
import pathlib
import statistics
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--build", type=pathlib.Path, default=pathlib.Path("build"))
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parent.parent
build = args.build.resolve()
results = collections.defaultdict(list)
variants = ("baseline", "optimized")
executables = (build / "native_kernel_input_profile_baseline", build / "native_kernel_input_profile")
with args.output.open("w") as log:
    log.write(f"UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
    for command in (["git", "describe", "--always", "--dirty"], ["uname", "-sm"],
                    ["xcrun", "clang", "--version"]):
        log.write(subprocess.check_output(command, cwd=root, text=True))
    cache = (build / "CMakeCache.txt").read_text()
    log.write("\n".join(line for line in cache.splitlines() if line.startswith(
        ("CMAKE_BUILD_TYPE:", "CMAKE_C_COMPILER:", "KERNEL_PROFILE_BASELINE_SOURCE:"))))
    log.write("\n" + (build / "CMakeFiles/native_kernel_input_profile.dir/flags.make").read_text())
    log.write("\nOne calling thread; warm buffers; CPU time; >=50ms calibration, three-batch median.\n"
              "Preparation uses private reused buffers and an indirect call per block to retain stores.\n"
              "VM-only uses pre-expanded float inputs; complete descriptor calls include allocation.\n"
              "All cases include scales repeated over 32 elements; phase=5 cases have tails.\n")
    for trial in range(5):
        for variant in (range(2) if trial % 2 == 0 else reversed(range(2))):
            process = subprocess.run([str(executables[variant])], cwd=root, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.write(f"\nPROCESS {trial} {variants[variant]}\n{process.stdout}")
            log.flush()
            process.check_returncode()
            count = 0
            for line in process.stdout.splitlines():
                fields = line.split()
                if len(fields) == 8 and fields[0] == "PROFILE":
                    _, precision, source, rows, width, phase, mode, us = fields
                    results[(variants[variant], precision, source, rows, width, phase, mode)].append(float(us))
                    count += 1
            assert count == 30
        print(f"Trial {trial + 1}/5 passed", flush=True)
    log.write("\nSUMMARY columns: variant precision source rows width phase mode median-us min-us max-us\n")
    for key, values in sorted(results.items()):
        assert len(values) == 5
        line = f"SUMMARY {' '.join(key)} {statistics.median(values):.6f} {min(values):.6f} {max(values):.6f}\n"
        log.write(line)
        print(line, end="")
