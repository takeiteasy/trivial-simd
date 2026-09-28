# Testing

The FiveAM suite checks the Lisp reference and every locally available SIMD
backend for both float types, empty vectors, SIMD tails, aliasing, and errors.
Direct copy tests check both precisions, round trips, empty ranges, and partial
copy-back. Native program tests check partial-initialization cleanup, finalization,
retained functions after redefinition, and collection during active native calls.
Threaded tests exercise concurrent first use with private call buffers. ECL tests
also check helper compilation, reuse, and compiler-failure fallback. Kernel
tests cover register spilling, scratch-slot reuse, encoding boundaries, square-root domain errors, signed zeros, single-rounding FMA, and
scalar-returning sums with slices and spilling. ECL also exercises interpreted
spilling and reduction definitions.

```sh
sbcl --script tests/run.lisp
ccl --no-init --load tests/run.lisp --eval '(quit)'
ecl --load tests/run.lisp --eval '(quit)'
```

Set `TRIVIAL_SIMD_BACKEND=lisp` or `native` to require that backend. GitHub
Actions runs the suite on the combinations in [backend coverage](backends.md#tested-combinations),
including ARM64 Linux, macOS, and Windows. ARM64 jobs require the native backend
and verify that the Lisp process runs on ARM64. Most jobs run on request; see
[CI](ci.md).

For a local ARM64 native check, build the library first, then run:

```sh
TRIVIAL_SIMD_BACKEND=native sbcl --script tests/run.lisp
```

## Kernel example

With Quicklisp loaded and this project registered with ASDF:

```lisp
(load (compile-file "examples/kernels.lisp" :output-file "/tmp/kernels-example.fasl"))
```

## Native VM checks

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The native harness runs SIMD and scalar C checks for both float types. It covers
scratch indexes through 65,535, partial blocks, math operators, scalar reductions,
allocation failure, size overflow, and cleanup after successful or failed calls.
Final-output checks compare every supported reduction operation bit-for-bit
with materialised block sums, including cancellation-sensitive inputs. Borrowed
scratch checks cover capacity, overflow, zero-length calls, ownership, and recovery
after domain errors.
GitHub Actions and sourcehut builds run it alongside the Lisp suite.

## Benchmark

See [measured results](benchmarks.md) for array comparisons, sum kernels, and
the scratch-storage evaluation.

```sh
sbcl --script tests/bench.lisp
ccl --no-init --batch --eval '(load (compile-file "tests/bench.lisp" :output-file "/tmp/trivial-simd-bench.fasl"))' --eval '(quit)'
ecl --eval '(load (compile-file "tests/bench.lisp" :output-file "/tmp/trivial-simd-bench.fas"))' --eval '(quit)'
```

Run this script explicitly when you want measurements. The ASDF test system
and GitHub Actions do not invoke it.

The array comparison measures addition, ordinary multiply-add (`a*b+c`), and
dot products against compiled, typed scalar Lisp loops. All cases use identical
typed arrays at 32, 1,024, and 65,536 elements, for both float types. The table
includes the Lisp fallback and available SIMD backends; `NATIVE-COPY` includes
the copying fallback's costs.

| Output | Meaning |
|---|---|
| `SCALAR` | Typed Lisp loop without explicit SIMD[^scalar] |
| `us/call` | Median microseconds per complete operation |
| `Speedup` | Scalar time divided by this implementation's time |

Illustrative output:

```text
Operation    Type         Elements Implementation us/call    Speedup
ADD          SINGLE-FLOAT     1024 SCALAR              2.000      1.00x
ADD          SINGLE-FLOAT     1024 NATIVE              0.500      4.00x
```

A value below `1.00x` means the implementation takes longer than the scalar
loop. The header identifies the Lisp version, machine type, selected backend,
and native array-access mode.

Arrays are allocated and initialized outside timing, and output buffers are
reused. Result checks run before measurement, including empty and SIMD-tail
arrays. Timings include library dispatch and array-access costs, and exclude
compilation and first-use setup. Each case warms up, doubles its iteration count
until a batch lasts at least 50 ms, and reports the median of three further
batches.[^timing] Speedup is a measurement, not a pass/fail threshold.

Separate sections compare ordinary multiply-add kernels with two bulk calls,
true FMA, and both float types for `sum(a*b)` implementations. Compile the
benchmark on ECL so the measurement loop and ordinary kernel definitions use
compiled code.[^compilation]
Results depend on the Lisp implementation, compiler, CPU, and array access cost.

## Kernel profiling

The optional profile measures identical native bytecode with per-call allocation
and explicitly owned reusable scratch. It reports complete Lisp calls separately
from C execution, with instructions, spills/reloads, scratch slots, and bytes.
It also checks overlapping calls using four workers. See
[spill measurements](kernel-spilling.md).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_KERNEL_PROFILE=ON
cmake --build build --config Release
TRIVIAL_SIMD_PROFILE_LIBRARY="$PWD/build/libnative_kernel_profile.dylib" \
  sbcl --script tests/kernel-profile.lisp
```

Use `.so` on Linux or `native_kernel_profile.dll` on Windows. On CCL/ECL,
compile and load `tests/kernel-profile.lisp`, following the benchmark commands
above. The profile is opt-in and does not run in ASDF tests or CI.
Set `TRIVIAL_SIMD_PROFILE_SHORT=1` to measure only 32/1,024-element cases for
repeated scratch-cache trials.

For a before/after comparison, export an earlier native source and configure
the matching baseline library:

```sh
git show 6289ec4:native/trivial_simd.c > /tmp/trivial-simd-baseline.c
cmake -S . -B build -DBUILD_KERNEL_PROFILE=ON \
  -DKERNEL_PROFILE_BASELINE_SOURCE=/tmp/trivial-simd-baseline.c
cmake --build build --config Release
TRIVIAL_SIMD_PROFILE_LIBRARY="$PWD/build/libnative_kernel_profile.dylib" \
TRIVIAL_SIMD_PROFILE_BASELINE="$PWD/build/libnative_kernel_profile_baseline.dylib" \
  sbcl --script tests/kernel-profile.lisp
```

The profile reports warmed median timings.[^profile] Its Lisp cache is a
benchmark-only prototype with explicit cleanup; normal calls retain per-call
allocation.

[^scalar]: Scalar functions are specialized for each float type and compiled
    explicitly when necessary. The Lisp compiler may choose SIMD instructions;
    the label describes source-level loops, not a guarantee about machine code.

[^timing]: Calibration uses `get-internal-real-time`. Each measured batch retains
    its final result outside the timed interval so reduction results remain
    observable. Existing diagnostic sections use the same calibrated timing.

[^compilation]: Compiled benchmark artifacts retain the source-system location
    and load from `/tmp`. The ECL section separately measures an `eval`-defined
    kernel and reports its first native call, including helper compilation,
    before warmed timings.

[^profile]: Direct C timings use calibrated CPU-time batches; Lisp timings use
    the shared benchmark timer. Both report the median of three batches after
    calibration. Baseline symbols have distinct names because some Lisp
    implementations resolve foreign symbols globally. The Lisp baseline uses
    a cached-pointer wrapper; direct C columns compare equivalent call paths.
