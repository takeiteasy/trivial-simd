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
GitHub Actions and sourcehut builds run it alongside the Lisp suite.

## Benchmark

```sh
sbcl --script tests/bench.lisp
ccl --no-init --batch --eval '(load (compile-file "tests/bench.lisp" :output-file "/tmp/trivial-simd-bench.fasl"))' --eval '(quit)'
ecl --eval '(load (compile-file "tests/bench.lisp" :output-file "/tmp/trivial-simd-bench.fas"))' --eval '(quit)'
```

Run this script explicitly when you want measurements. The ASDF test system
and GitHub Actions do not invoke it.

The benchmark reports microseconds per vector-add call for 32, 1,024, and
65,536 elements of both float types on the available Lisp, native C, and SBCL
backends. It also compares separate-operation `a*b+c`, true FMA, and `sum(a*b)`
kernels, and reports `NATIVE-COPY` for the copying fallback.

Each timing is the median of three warmed batches. Compile the benchmark on ECL
so the measurement loop and ordinary kernel definitions use compiled code. The
ECL section separately measures an `eval`-defined kernel and reports its first
native call, including helper compilation, before warmed timings. Compiled
benchmark artifacts retain the source-system location and can load from `/tmp`.
Results depend on the Lisp implementation, compiler, CPU, and array access cost.
