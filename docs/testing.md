# Testing

The FiveAM suite checks the Lisp reference and every locally available SIMD
backend for both float types, empty vectors, SIMD tails, aliasing, and errors.

```sh
sbcl --script tests/run.lisp
ccl --no-init --load tests/run.lisp --eval '(quit)'
ecl --load tests/run.lisp --eval '(quit)'
```

Set `TRIVIAL_SIMD_BACKEND=lisp` or `native` to require that backend. GitHub
Actions runs the suite on available SBCL, ECL, and CCL combinations across
macOS, Linux, and Windows.

## Benchmark

```sh
sbcl --script bench/run.lisp
```

The benchmark reports microseconds per vector-add call for 32, 1,024, and
65,536 `single-float` elements on the Lisp and native C backends. Results
depend on the Lisp implementation, compiler, CPU, and copying cost.
