# Testing

The runner loads declared test dependencies through Quicklisp when available.
The FiveAM suite checks the Lisp reference and every locally available SIMD
backend for both float types and eight integer types, empty vectors, SIMD
tails, aliasing, wrapping arithmetic, division errors, and type boundaries.
Extended operation tests cover unary and bounded arithmetic, all conversion
type pairs and rounding modes, byte masks, selection, and mask kernels.
bf16 and f16 tests widen all 65536 bit patterns of each encoding on every
backend, and narrow every encoded value, the midpoints between neighbours and
one unit either side, and pseudo-random single-floats. They compare the results
with an independent rational reference for nearest-even, truncate, floor and
ceiling, and cover subnormals, infinities, overflow, signed zeros, NaN payloads,
slices, strides, tails and errors. Extended conversions widen every encoded bit
pattern to f64 and check both encoded cross-format directions in all rounding
modes. f64 narrowing checks every destination midpoint and its adjacent f64
values against the rational reference, plus random values and extreme magnitudes.
Capability tests check native dispatch and the Lisp fallback for
rounding modes unsupported by an older native library, including foreign-memory
views. Signaling NaN inputs are skipped on Lisps that cannot hold one (ECL).
A test fails when the native library is loaded but lacks the bf16/f16 symbols,
which loading also reports with a warning; rebuild with CMake to fix it.
The suite disables the [inline small-array path](small-arrays.md) so that small
calls exercise the backends; the inline tests enable it and compare every inline
operation with the backends for lengths around the limit, and check that
mismatches and errors still reach the normal path.
Whole-vector call tests compare keyword-free calls with explicit-slice calls and
check that mismatched lengths, types, masks and scalars still signal errors.
Copy tests cover `copy!`, `fill!` and `swap!` on every element type and backend,
including overlap, large unaligned slices and error cases.
Stride tests compare every bulk operation and reduction with a contiguous call on
plainly gathered copies for strides 1, 2, 3 and negative values on every
backend, and cover bounds, empty slices, scalar operands and aliased outputs.
Reduction tests cover `minimum`, `maximum`, `argmin`, `argmax`, `asum`, `nrm2`
and `:accumulate` on every backend, including ties, signed zeros, absolute
indexes, overflow, wrapping and error cases.
Kernel reducer tests compare `minimum`, `maximum`, `argmin`, `argmax`, `asum`
and `nrm2` kernels with the bulk reducers on every backend, covering slices,
block boundaries, ties, selections, spilling, complex vectors and overflow.
Declared-input tests cover all eight integer sources, both float precisions,
conversion rounding, repeated scales, sliced block alignment, masks, reducers,
row strides, views with small staging buffers, overlap rejection, native-symbol
fallback, spilling, redefinition and concurrent calls. C tests also check
descriptor bounds, empty calls, allocation failure and scratch cleanup.
BLAS Level 1 tests cover real and complex precisions, offsets, increments,
errors, rotations, and convenience wrappers. They compare against an available
system CBLAS (Accelerate on macOS, or OpenBLAS/BLAS on other platforms); only
the reference comparison is skipped when no CBLAS is installed.
Level 2 tests cover matrix views, layouts, packed and band storage, transpose,
triangular solves, Hermitian updates, and reference CBLAS comparisons for
dense, banded, packed, symmetric, and rank-update operations.
Level 3 tests cover all precision variants, matrix overlap, padded views,
operation flags, and CBLAS comparisons across its nine routine families.
Randomized CBLAS tests compare every Level 2 and 3 family on padded, offset
subviews in both layouts with all transpose, triangle, side, and diagonal flags
and negative increments. Zero-scalar tests check that `alpha` and `beta` of
zero never read the operands they scale. Native BLAS tests force the native and
Lisp kernels in turn for `gemm`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm`,
`gemv`, `ger`, and `trsv` across layouts, transposes, offsets, increments, and
block-edge sizes, and check the fallback when the library lacks the symbols.
Vector view tests run every bulk operation, reduction, conversion, mask
operation, copy, kernel and BLAS family on views of foreign copies of Lisp
vectors, alone and mixed with Lisp vectors, on every backend and access mode,
with 5-element and default buffer blocks. They compare the results and final
memory with the same call on Lisp vectors, and cover slices, strides, ties
across blocks, overlapping copies within one buffer, matrix overlap by address,
errors, direct native pointer use, and (on SBCL) that a large view is read
without a full-length copy.
`ctest` checks bf16/f16 narrowing against an independent representable-value
reference in SIMD and scalar builds, including mixed-sign lanes, boundaries,
tails, random single-floats and preservation of caller floating-point settings
and exception flags under available rounding, flush-to-zero and trap controls.
It also checks the C kernels, including the scalar build and allocation
failure. On x86-64 it checks the SSE2 BLAS kernels directly and the AVX+FMA
kernels where the CPU supports them; the software-FMA target forces SSE2.
Direct copy tests check both precisions, round trips, empty ranges, and partial
copy-back, compact slice transfers, and overlapping copy-mode input snapshots. Native program tests check partial-initialization cleanup, finalization,
retained functions after redefinition, and collection during active native calls.
Threaded tests exercise concurrent first use with private call buffers. ECL tests
also check helper sharing across definitions, signature separation, both
precisions and access modes, concurrent initialization, compiler-failure fallback,
and collection with populated helper caches. Kernel tests cover register spilling, scratch-slot reuse, encoding boundaries, square-root domain errors, signed zeros, single-rounding FMA, deterministic exact-reference comparisons, forced FMA fallbacks, and
scalar-returning sums with slices and spilling. ECL also exercises interpreted
spilling and reduction definitions.

SBCL test runs reserve a 4 GiB heap for cold compilation.[^sbcl-heap]

```sh
sbcl --dynamic-space-size 4096 --script tests/run.lisp
ccl --no-init --load tests/run.lisp --eval '(quit)'
ecl --load tests/run.lisp --eval '(quit)'
```

Set `TRIVIAL_SIMD_BACKEND=lisp` or `native` to require that backend. GitHub
Actions runs the suite on the combinations in [backend coverage](backends.md#tested-combinations),
including ARM64 Linux and Windows. macOS testing is local only. ARM64 jobs
require the native backend and verify that the Lisp process runs on ARM64.
Most jobs run on request; see [CI](ci.md).

For a local ARM64 native check, build the library first, then run:

```sh
TRIVIAL_SIMD_BACKEND=native sbcl --dynamic-space-size 4096 --script tests/run.lisp
```

## Local platform runs

All macOS combinations run locally. ARM64 scripts use the installed Lisp on
`PATH`, build the ARM64 native library, run `ctest`, and run the Lisp suite.
They require an ARM64 Mac, CMake, Xcode command line tools, and test dependencies
available through Quicklisp or ASDF. The default required backend is `native`;
set `TRIVIAL_SIMD_BACKEND=lisp` to check the Lisp fallback. The runner verifies
that the Lisp process is ARM64.

```sh
tests/arm64-sbcl.sh
tests/arm64-ecl.sh
tests/arm64-ccl.sh
TRIVIAL_SIMD_BACKEND=lisp tests/arm64-sbcl.sh
```

| Installed ARM64 Lisp | Version used locally |
|---|---|
| SBCL | 2.6.8[^sbcl-heap] |
| ECL | 26.5.5 |
| CCL | `1.13 (v1.13-459-g690ff7ea)` preview |

The x86-64 scripts run under Rosetta and cache their Lisp installations.
They need `curl`, `rsync`, CMake, and Rosetta. ECL also needs `make` and Xcode
command line tools; its first run builds ECL from source and takes several minutes.
Each script copies the tree to its cache, builds an x86-64 library, runs `ctest`,
and runs the Lisp suite.[^x86-copy]

```sh
tests/x86-sbcl.sh
tests/x86-ecl.sh
tests/x86-ccl.sh
```

| Script | Version override (default) | Cache override (default under `~/.cache/trivial-simd/`) |
|---|---|---|
| `tests/x86-sbcl.sh` | `SBCL_X86_VERSION` (`2.6.8`) | `TRIVIAL_SIMD_X86_CACHE` (`x86-sbcl`) |
| `tests/x86-ecl.sh` | `ECL_X86_VERSION` (`26.5.5`) | `TRIVIAL_SIMD_X86_ECL_CACHE` (`x86-ecl`) |
| `tests/x86-ccl.sh` | `CCL_X86_VERSION` (`1.13`) | `TRIVIAL_SIMD_X86_CCL_CACHE` (`x86-ccl`) |

CCL uses the complete macOS x86 release archive from
[Clozure CL releases](https://github.com/Clozure/ccl/releases/tag/v1.13).
Both CCL scripts also run 500 lifetime trials and three additional full suites
in fresh processes. Extra script arguments are passed to the Lisp after the
first suite loads. ECL writes installation build logs in its cache.
Use `trash` on a cache directory to reset it.

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

The native harness runs SIMD, scalar C, and forced software-FMA checks for floats,
and arithmetic, division, and kernel checks for every integer type. It covers
scratch indexes through 65,535, partial blocks, math operators, scalar reductions,
allocation failure, size overflow, and cleanup after successful or failed calls.
Final-output checks compare every supported reduction operation bit-for-bit
with materialised block sums, including cancellation-sensitive inputs. Borrowed
scratch checks cover capacity, overflow, zero-length calls, ownership, and recovery
after domain errors.
GitHub Actions and sourcehut builds run it alongside the Lisp suite.

## Native program lifetime

The redefinition test keeps one older function callable while four discarded
programs are reclaimed. After dropping the retained function, all five owners
are collected and each program's buffers are released exactly once. Weak
observations identify any live owners without retaining them.[^lifetime]

The optional stress runner executes the full suite, followed by 500 lifetime
trials. It requires the native library and exits with failure on any error.

```sh
ccl --no-init --batch --load tests/run-lifetime-stress.lisp --eval '(quit)'
```

Both local CCL scripts run 500 trials and five full suites in fresh processes.
CCL is excluded from GitHub workflows.

## Test diagnostics

See [timed and bounded local runs](test-diagnostics.md) for selecting tests,
locating slow BLAS phases and retaining output from timed-out commands.

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

`tests/view-bench.lisp` times `dot`, `axpy!`, `convert!` and a sum kernel with one
[vector view](vector-views.md) operand against Lisp vectors on every backend:

```sh
sbcl --script tests/view-bench.lisp
```

`tests/overhead-bench.lisp` times 4-, 32- and 1,024-element `add!`, `dot`, `min!`
and `compare!` calls against scalar loops, the inline path and every available
backend:

```sh
sbcl --script tests/overhead-bench.lisp
```

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

## Row-batch benchmark

`tests/kernel-rows-bench.lisp` compares one batched dot kernel with separate
row calls for both float types, 1/32/1,024 rows, and widths 4/32/256/1,024.
It checks results before warmed, calibrated median timings and reports
microseconds per batch, per row, and speedup. Outputs are reused and compilation
is excluded. See [row-batch results](kernel-performance.md#row-batching).

```sh
sbcl --script tests/kernel-rows-bench.lisp
ccl --no-init --batch --eval '(load (compile-file "tests/kernel-rows-bench.lisp" :output-file "/tmp/kernel-rows-bench.fasl"))' --eval '(quit)'
ecl --eval '(load (compile-file "tests/kernel-rows-bench.lisp" :output-file "/tmp/kernel-rows-bench.fas"))' --eval '(quit)'
```

Batch regression tests cover row layouts, scalar equivalence, pointer/copy
access, foreign views, overlap errors, one-call execution, symbol fallback,
spilling, redefinition and concurrent calls. Native tests check one scratch
allocation per batch, allocation failure, overflow, domain errors and cleanup.

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

[^lifetime]: Definitions, calls, weak observations, and collections run in
    short-lived workers. The tests join workers and wait for termination before
    checking discarded owners. CCL signals its join semaphore before completing
    thread cleanup. Its tests use native weak populations for observations and
    explicitly drain the queue of already-unreachable finalizers after GC.
    Before collection, a CCL barrier inserts and removes its own metadata key
    in the weak documentation and function-name tables, replacing cached keys
    without removing any kernel entry or changing retained references.

[^timing]: Calibration uses `get-internal-real-time`. Each measured batch retains
    its final result outside the timed interval so reduction results remain
    observable. Existing diagnostic sections use the same calibrated timing.

[^compilation]: Compiled benchmark artifacts retain the source-system location
    and load from `/tmp`. The ECL section measures compiled and `eval` definitions
    for elementwise multiply-add and sum kernels in both precisions. Five trials
    use fresh caches and definitions to report median first-signature calls
    (including compilation) and first calls of another definition sharing the
    helper. Shared definitions use a different expression and result checks run
    outside timing. Cache checks require one compiled helper per signature.
    First-call results include the clock resolution; a zero means the call is
    below that resolution. Warmed timings use 32/1,024/65,536 elements.

[^profile]: Direct C timings use calibrated CPU-time batches; Lisp timings use
    the shared benchmark timer. Both report the median of three batches after
    calibration. Baseline symbols have distinct names because some Lisp
    implementations resolve foreign symbols globally. The Lisp baseline uses
    a cached-pointer wrapper; direct C columns compare equivalent call paths.

## Slice benchmark

`tests/slice-bench.lisp` compares native copy and pointer access for addition,
sum, dot, elementwise kernels, and sum kernels in both precisions. It measures
32-element slices in 1,024- and 65,536-element parents, a 1,024-element slice,
and a whole-vector control. Result checks precede calibrated median timings.

```sh
sbcl --script tests/slice-bench.lisp
ccl --no-init --batch --eval '(load (compile-file "tests/slice-bench.lisp" :output-file "/tmp/slice-bench.fasl"))' --eval '(quit)'
ecl --norc --eval '(load (compile-file "tests/slice-bench.lisp" :output-file "/tmp/slice-bench.fas"))' --eval '(quit)'
```

## BLAS profiling

`tests/blas-profile.c` times `gemm`, `syrk`, `trsm`, row- and column-major
`trsv`, and `gemv` at 64, 256, and 512 in both precisions. On x86-64 it reports
the SSE2 and AVX+FMA kernels separately.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_BLAS_PROFILE=ON \
      "-DBLAS_PROFILE_DEFINITIONS=TS_BLAS_DEPTH=384;TS_BLAS_AVX_F64_ROWS=3;TS_BLAS_AVX_F64_COLUMNS=4"
cmake --build build --target native_blas_profile
build/native_blas_profile
```

`BLAS_PROFILE_DEFINITIONS` overrides `TS_BLAS_DEPTH`, `TS_BLAS_ROW_BLOCK`,
`TS_BLAS_COLUMN_BLOCK`, `TS_BLAS_TRIANGLE_BLOCK` (`trmm` only), and the AVX tile shapes
`TS_BLAS_AVX_{F32,F64}_{ROWS,COLUMNS}`.[^tiles] The `blas` job of the manual
**FMA profile** workflow sweeps these on x86-64 Linux; see [CI](ci.md#blas-measurements).

[^tiles]: A tile is `ROWS` vectors by `COLUMNS` scalars; `ROWS × COLUMNS`
    accumulators must fit the 16 ymm registers with a few to spare.

## FMA profiling

`tests/fma-bench.lisp` compares exact portable FMA, scalar C helpers, in-process ARM64 FMA, and automatic
selection for elementwise and sum kernels, in both precisions at 32, 1,024,
and 65,536 elements. Correctness checks precede warmed calibrated medians.
`Hardware FMA` reports guarded hardware availability; ARM64 native kernels use NEON.
The profile also reports ARM64 adapter availability.

```sh
sbcl --script tests/fma-bench.lisp
ccl --no-init --batch --eval '(load (compile-file "tests/fma-bench.lisp" :output-file "/tmp/fma-bench.fasl"))' --eval '(quit)'
ecl --norc --eval '(load (compile-file "tests/fma-bench.lisp" :output-file "/tmp/fma-bench.fas"))' --eval '(quit)'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_FMA_PROFILE=ON
cmake --build build --config Release
build/native_fma_software_profile
build/native_fma_profile
```

The C executables measure identical native bytecode with software and automatic
FMA selection. They use calibrated median CPU-time batches and verify every
output outside timing. On Windows, use the `.exe` suffix.
The manual **FMA profile** GitHub workflow runs native and Lisp checks, then two
measurement trials on x86-64 Linux for SBCL, CCL, and ECL. The runner exits
with a failure status on Lisp errors. Benchmarks remain
outside the normal test suite.

For an isolated ARM64 gate comparison, set `TRIVIAL_SIMD_FMA_GATE_ONLY=1`.
Set `TRIVIAL_SIMD_FMA_BASELINE` to a baseline `kernel-math.lisp` source file to
compile and load its scalar helper before profiling. Baseline `:auto` disables
the new adapter; `:in-process` uses the candidate loop. Run five fresh processes
for each Lisp. See [ARM64 FMA measurements](arm64-fma.md#performance-gate).

[^x86-copy]: The Lisp loader reads `build/libtrivial_simd.dylib`, so the x86-64
    library cannot share the ARM64 `build/`. The SBCL tarball keeps its contrib
    fasls under `obj/sbcl-home`, which the script sets as `SBCL_HOME`.

[^sbcl-heap]: Compiling the suite's expanded kernel definitions in the same
    image as the library can exceed SBCL's default 1 GiB heap. SBCL CI jobs
    and both local architecture scripts reserve 4 GiB. For direct runs, pass
    `--dynamic-space-size 4096` before `--script` or other Lisp options.
