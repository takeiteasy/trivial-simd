# Testing

The FiveAM suite checks the Lisp reference and every locally available SIMD
backend for both float types, empty vectors, SIMD tails, aliasing, and errors.

```sh
sbcl --script tests/run.lisp
ccl --no-init --load tests/run.lisp --eval '(quit)'
ecl --load tests/run.lisp --eval '(quit)'
```

Set `TRIVIAL_SIMD_BACKEND=lisp` or `native` to require that backend. GitHub
Actions runs the suite on the combinations in [backend coverage](backends.md#tested-combinations),
including ARM64 Linux, macOS, and Windows. ARM64 jobs require the native backend
and verify that the Lisp process runs on ARM64.

For a local ARM64 native check, build the library first, then run:

```sh
TRIVIAL_SIMD_BACKEND=native sbcl --script tests/run.lisp
```

## Benchmark

```sh
sbcl --script tests/bench.lisp
```

Run this script explicitly when you want measurements. The ASDF test system
and GitHub Actions do not invoke it.

The benchmark reports microseconds per vector-add call for 32, 1,024, and
65,536 `single-float` elements on the available Lisp, native C, and SBCL
backends. Results depend on the Lisp implementation, compiler, CPU, and array
access cost.
