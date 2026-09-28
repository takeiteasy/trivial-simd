# Kernel performance

Kernels avoid intermediate vectors. Typed Lisp kernels are much faster than
separate Lisp bulk operations in these measurements; native kernels perform
similarly to native bulk operations for simple expressions. Native reductions
may cost more at small sizes.

Microseconds per call on Apple M1, SBCL 2.6.8, `single-float` vectors.
Run [the benchmark](testing.md#benchmark) for your workload.

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

| Elements | Lisp sum kernel | Lisp multiply+sum | Native sum kernel | Native multiply+sum | Native dot |
|---|---|---|---|---|---|
| 32 | 0.06 | 1.07 | 0.31 | 0.14 | 0.07 |
| 1,024 | 1.25 | 30.98 | 0.60 | 0.50 | 0.30 |
| 65,536 | 77.70 | 2,004.30 | 21.00 | 24.55 | 16.35 |
