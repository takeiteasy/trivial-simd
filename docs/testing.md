# Testing

The FiveAM suite checks the Lisp reference and every locally available SIMD
backend for both float types, empty vectors, SIMD tails, aliasing, and errors.
Kernel tests also cover register spilling, scratch-slot reuse, and encoding
boundaries, square-root domain errors, signed zeros, single-rounding FMA, scalar-returning sum kernels, and reductions
with slices and spilling. ECL also exercises an interpreted spilling-kernel definition.

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

The native harness runs SIMD and scalar C checks for both float types, scratch indexes through 65,535,
partial blocks, new math operators, scalar reduction output, domain-error
cleanup, allocation failure,
size overflow, and cleanup. GitHub Actions
and sourcehut builds run it alongside the Lisp suite.

## Benchmark

```sh
sbcl --script tests/bench.lisp
```

Run this script explicitly when you want measurements. The ASDF test system
and GitHub Actions do not invoke it.

The benchmark reports microseconds per vector-add call for 32, 1,024, and
65,536 `single-float` elements on the available Lisp, native C, and SBCL
backends, plus separate-operation `a*b+c`, true FMA, and `sum(a*b)` kernel comparisons and `NATIVE-COPY` for the copying fallback. Results depend on the Lisp implementation, compiler, CPU, and array
access cost.
