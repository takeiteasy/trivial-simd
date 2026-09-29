# ARM64 scalar FMA

ARM64 Lisp kernels use in-process scalar FMA in both float precisions on supported
SBCL, CCL, and ECL versions. The native backend uses NEON. The native library is
optional for Lisp FMA; unsupported compiler versions retain the scalar C helper
when available and exact integer arithmetic otherwise.

## Compiler support

| Lisp | Guarded versions | Scalar implementation |
|---|---|---|
| SBCL | 2.6.8, 2.6.9 | Compiler VOP emits `FMADD` |
| CCL | 1.13 ARM64 | LAP primitive emits `FMADD` |
| ECL | 26.5.5 AArch64 | C inline builtin emits scalar FMA |

Kernels select the scalar loop once per invocation, outside the element loop.
An `eval`-defined ECL kernel calls the compiled scalar helper. The exported
`trivial-simd:fma` function uses the same guarded acceleration. Finite operands
with finite results follow [the numerical contract](kernels.md#numerical-behavior),
including subnormals, ties, signed zero, and cancellation of an overflowing
intermediate product.[^adapter]

## Performance gate

Apple M1 measurements use SBCL 2.6.8, CCL 1.13, and ECL 26.5.5. Five fresh
processes per Lisp compare the in-process loop against the scalar implementation
at commit `65cbdc3`, with the native helper available. Each precision combines
six cases: elementwise and sum kernels at 32, 1,024, and 65,536 elements.
The gate requires at least a 10% geometric mean reduction in time in every run
and no repeatable regression above 5% in an individual case.

| Lisp | Single-float reduction across five runs | Double-float reduction across five runs |
|---|---:|---:|
| SBCL | 96.67–96.72% | 96.85–96.90% |
| CCL | 90.37–90.51% | 76.20–78.02% |
| ECL | 88.43–88.72% | 88.43–88.67% |

Every run passes; every individual case is faster. The table below gives median
microseconds per call across those five processes. Each cell is **elementwise /
sum**. The portable, C helper, and NEON columns come from five additional fresh
full-profile processes per Lisp; run [FMA profiling](testing.md#fma-profiling)
for a local comparison.[^timing]

| Lisp | Type | Elements | Exact portable | Scalar C helper | Baseline auto | In-process | Native NEON |
|---|---|---:|---:|---:|---:|---:|---:|
| SBCL | single | 32 | 6.072 / 5.985 | 2.662 / 2.626 | 2.728 / 2.660 | 0.142 / 0.099 | 0.153 / 0.135 |
| SBCL | single | 1,024 | 192.645 / 189.246 | 83.513 / 84.146 | 83.163 / 83.583 | 3.131 / 1.871 | 0.366 / 0.482 |
| SBCL | single | 65,536 | 12,381.750 / 12,123.250 | 5,335.750 / 5,361.563 | 5,468.500 / 5,390.688 | 197.207 / 115.908 | 18.058 / 24.401 |
| SBCL | double | 32 | 7.994 / 7.772 | 2.802 / 2.746 | 2.838 / 2.798 | 0.142 / 0.101 | 0.160 / 0.141 |
| SBCL | double | 1,024 | 249.895 / 246.664 | 88.005 / 86.688 | 88.750 / 86.986 | 3.131 / 1.870 | 0.572 / 0.899 |
| SBCL | double | 65,536 | 15,992.500 / 16,061.750 | 5,711.563 / 5,525.500 | 5,744.688 / 5,583.688 | 197.246 / 115.846 | 34.411 / 53.183 |
| CCL | single | 32 | 183.721 / 183.424 | 3.688 / 4.056 | 3.613 / 3.990 | 0.416 / 0.545 | 0.460 / 0.356 |
| CCL | single | 1,024 | 5,851.187 / 5,923.437 | 113.514 / 125.545 | 107.930 / 122.879 | 7.202 / 12.922 | 0.672 / 0.682 |
| CCL | single | 65,536 | 374,460.000 / 376,150.000 | 7,173.375 / 8,036.875 | 6,947.750 / 7,781.500 | 446.023 / 825.547 | 19.518 / 23.692 |
| CCL | double | 32 | 215.926 / 227.207 | 4.185 / 4.842 | 4.064 / 4.879 | 0.907 / 1.370 | 0.461 / 0.364 |
| CCL | double | 1,024 | 6,896.000 / 6,940.375 | 126.932 / 149.811 | 123.375 / 150.105 | 23.542 / 39.295 | 0.912 / 1.069 |
| CCL | double | 65,536 | 441,033.000 / 443,178.000 | 7,960.500 / 9,303.375 | 7,629.125 / 9,172.125 | 1,274.062 / 2,113.812 | 38.421 / 50.778 |
| ECL | single | 32 | 42.302 / 42.736 | 21.489 / 21.797 | 22.182 / 22.953 | 2.504 / 3.399 | 1.164 / 1.098 |
| ECL | single | 1,024 | 1,332.172 / 1,350.719 | 666.625 / 691.906 | 656.531 / 728.672 | 59.717 / 89.864 | 1.375 / 1.433 |
| ECL | single | 65,536 | 85,014.000 / 86,157.000 | 42,472.000 / 44,177.000 | 41,995.500 / 46,083.500 | 3,713.125 / 5,954.313 | 19.575 / 24.293 |
| ECL | double | 32 | 45.334 / 47.015 | 21.827 / 22.472 | 22.722 / 23.137 | 2.727 / 3.535 | 1.318 / 1.229 |
| ECL | double | 1,024 | 1,457.969 / 1,449.797 | 674.555 / 721.734 | 668.445 / 730.227 | 58.931 / 89.540 | 1.761 / 1.975 |
| ECL | double | 65,536 | 92,015.000 / 93,281.000 | 42,744.500 / 44,828.500 | 42,783.500 / 45,545.500 | 3,732.375 / 5,988.438 | 35.504 / 51.606 |

## Limitations

- Compiler internals are version guarded. Unlisted versions use the fallback.
- CCL double-float results and ECL scalar calls still box floats. See the
  [boxing improvement ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/64).
- This path uses scalar instructions. Lisp ARM64 vector generation is tracked by
  the [ARM64 SIMD ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/31).
- NaNs, infinities, non-default rounding modes, and floating-point traps share
  [the kernel limitations](kernels.md#limitations).

[^adapter]: Load-time smoke checks disable selection if the guarded primitives
    do not produce the expected result. SBCL keeps operands in float registers;
    CCL uses its float calling representation; ECL compiles C builtins as part
    of the ASDF system. Internal `*fma-mode*` bindings select `:portable`,
    `:native`, or `:in-process` for tests and benchmarks; `:auto` selects the
    qualified path. A missing adapter falls back rather than changing results.

[^timing]: Calibrated batches report warmed medians, with checks outside the timed
    interval. Cross-process medians and per-run geometric means answer different
    questions; the gate uses each run independently. Sum timings retain each
    backend's documented accumulation order. Native call setup dominates small
    inputs; NEON is faster for large inputs.
