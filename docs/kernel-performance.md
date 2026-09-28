# Kernel performance

Kernels avoid intermediate vectors. Typed Lisp kernels are much faster than
separate Lisp bulk operations in these measurements; native kernels perform
similarly to native bulk operations for simple expressions. Native reductions
may cost more at small sizes.

Microseconds per call on Apple M1, SBCL 2.6.8; `single-float` unless a type is listed.
Run [the benchmark](testing.md#benchmark) for your workload.
See [spill profiling](kernel-spilling.md) for register-heavy kernels and scratch
storage measurements.

## Multiply-add

`(+ (* a b) c)` uses separate multiplication and addition. `two ops` is
`multiply!` followed by `add!`.

| Elements | Lisp kernel | Lisp two ops | Native kernel | Native two ops |
|---|---|---|---|---|
| 32 | 0.13 | 1.37 | 0.14 | 0.19 |
| 1,024 | 2.84 | 39.57 | 0.36 | 0.41 |
| 65,536 | 178.85 | 2,564.35 | 16.55 | 16.50 |

## True FMA

`(trivial-simd:fma a b c)` guarantees one rounding step. Its exact Lisp fallback
costs substantially more than ordinary multiplication and addition; see
[kernel limitations](kernels.md#limitations). Native ARM64 uses NEON FMA.

| Elements | Lisp FMA kernel | Native FMA kernel |
|---|---|---|
| 32 | 3.96 | 0.13 |
| 1,024 | 121.74 | 0.35 |
| 65,536 | 7,805.40 | 18.55 |

## Sum of products

`(trivial-simd:sum (* a b))` returns a scalar without an intermediate vector.
`multiply+sum` uses a reusable output vector followed by `sum`. Bulk `dot`
performs the same calculation with a specialised implementation.

| Type | Elements | Lisp sum kernel | Lisp multiply+sum | Native sum kernel | Native multiply+sum | Native dot |
|---|---:|---:|---:|---:|---:|---:|
| single-float | 32 | 0.06 | 1.05 | 0.11 | 0.15 | 0.07 |
| single-float | 1,024 | 1.21 | 31.21 | 0.27 | 0.49 | 0.30 |
| single-float | 65,536 | 75.03 | 1952.59 | 10.72 | 23.75 | 15.44 |
| double-float | 32 | 0.07 | 1.23 | 0.12 | 0.15 | 0.07 |
| double-float | 1,024 | 1.22 | 39.02 | 0.46 | 0.82 | 0.53 |
| double-float | 65,536 | 75.72 | 2790.75 | 21.67 | 47.22 | 32.50 |

Native sums evaluate final instructions directly into lane accumulators. Each
256-element block retains its lane, tail, and block addition order, without an
output-block store/read. Short calls still include pointer and scalar-output
setup; see [kernel limitations](kernels.md#limitations).

## Native copy mode

Copy mode uses typed loops and separate input buffers. These measurements are
microseconds per 1,024-element `add!` call on Apple M1, using three repeated runs.
SBCL is 2.6.8, CCL is 1.13, and ECL is 26.5.5.

| Implementation | Single-float | Double-float |
|---|---:|---:|
| SBCL | 3.03 | 3.09 |
| CCL | 13.52 | 13.61 |
| ECL | 642 | 685 |

Copying still depends on implementation-specific foreign memory access. Small
slices also copy whole input vectors; see [array access](backends.md#array-access).

## ECL native calls

ECL 26.5.5 on Apple M1, `single-float` multiply-add kernels. Values are median
microseconds per warmed call across three batches. The fallback column uses
interpreted call setup when helper compilation is unavailable.

| Elements | Compiled definition | Eval definition | Eval with fallback | Two bulk calls |
|---|---:|---:|---:|---:|
| 32 | 1.18 | 1.76 | 5.45 | 1.47 |
| 1,024 | 1.31 | 2.08 | 5.61 | 1.88 |
| 65,536 | 17.87 | 18.36 | 22.38 | 18.33 |

The first native call for the eval definition takes about 320 ms, including
helper compilation. The helper is reused for later calls. See
[kernel limitations](kernels.md#limitations) for cold-start work.
