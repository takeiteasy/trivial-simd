# Kernel performance

Kernels avoid intermediate vectors. Typed Lisp kernels are much faster than
separate Lisp bulk operations in these measurements; native kernels perform
similarly to native bulk operations for simple expressions. Native reductions
may cost more at small sizes.

Microseconds per call on Apple M1, SBCL 2.6.8; `single-float` unless a type is listed.
Run [the benchmark](testing.md#benchmark) for your workload.
See [benchmark results](benchmarks.md) for typed scalar comparisons and the
native reduction baseline.
See [spill profiling](kernel-spilling.md) for register-heavy kernels and scratch
storage measurements.

## Multiply-add

`(+ (* a b) c)` uses separate multiplication and addition. `two ops` is
`multiply!` followed by `add!`.

| Elements | Lisp kernel | Lisp two ops | Native kernel | Native two ops |
|---|---|---|---|---|
| 32 | 0.13 | 1.39 | 0.17 | 0.38 |
| 1,024 | 4.53 | 69.07 | 0.63 | 0.41 |
| 65,536 | 172.57 | 2,476.16 | 16.39 | 16.05 |

## True FMA

`(trivial-simd:fma a b c)` guarantees one rounding step. SBCL x86 uses guarded
scalar and packed FMA instructions. Scalar Lisp kernels use bundled C helpers
when available; the exact integer implementation remains the fallback.
Native ARM64 uses NEON FMA; native x86 selects hardware FMA at runtime.
See [kernel limitations](kernels.md#limitations) for remaining scalar-call costs.

Representative warmed elementwise Lisp kernel timings on Apple M1, in
microseconds per call. SBCL is 2.6.8, CCL is 1.13, and ECL is 26.5.5.
Both complete runs use calibrated medians; the C helper wins at small and large
sizes in both precisions. See [FMA profiling](testing.md#fma-profiling).

| Lisp | Type | Elements | Exact fallback | C helper |
|---|---|---:|---:|---:|
| SBCL | single-float | 32 | 6.45 | 2.80 |
| SBCL | single-float | 65,536 | 13,123.50 | 5,768.75 |
| SBCL | double-float | 32 | 8.52 | 3.04 |
| SBCL | double-float | 65,536 | 17,924.50 | 6,039.94 |
| CCL | single-float | 32 | 194.22 | 3.70 |
| CCL | single-float | 65,536 | 399,568.00 | 7,164.88 |
| CCL | double-float | 32 | 219.53 | 4.15 |
| CCL | double-float | 65,536 | 456,420.00 | 7,894.38 |
| ECL | single-float | 32 | 43.66 | 21.67 |
| ECL | single-float | 65,536 | 90,330.00 | 43,076.50 |
| ECL | double-float | 32 | 45.01 | 21.35 |
| ECL | double-float | 65,536 | 87,696.00 | 40,339.00 |

Native x86-64 Linux profiling compares identical FMA bytecode with per-lane
`fma`/`fmaf` and runtime-selected packed hardware FMA. Both repeated trials
show a gain at every measured size and precision. Representative C execution
microseconds per call from the SBCL profiling job are:

| Type | Elements | Per-lane software path | Packed hardware path |
|---|---:|---:|---:|
| single-float | 32 | 0.101 | 0.035 |
| single-float | 65,536 | 176.484 | 44.447 |
| double-float | 32 | 0.114 | 0.068 |
| double-float | 65,536 | 190.000 | 95.627 |

These timings come from GitHub Linux x86-64 profiling runners; they are not
comparable to the M1 table above.[^x86-fma]

SBCL 2.6.9 on an AMD EPYC 9V74 GitHub runner measures the shared SIMD kernel
with exact lane FMA and automatic packed FMA:

| Type | Elements | Exact lane FMA | Packed hardware FMA |
|---|---:|---:|---:|
| single-float | 32 | 9.277 | 0.149 |
| single-float | 65,536 | 18,749.750 | 140.623 |
| double-float | 32 | 11.596 | 0.221 |
| double-float | 65,536 | 23,749.750 | 308.586 |

Two trials show the gain for elementwise and sum kernels at all three sizes.


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

Copy mode allocates and transfers only the requested slice. Inputs have separate
buffers, and copy-back updates only the destination range. The cost depends on
slice length rather than parent-vector length. See the
[slice benchmark](testing.md#slice-benchmark) for both precisions and all native
operation families.

For a 32-element `add!` slice in a 65,536-element parent, measured copy-mode
microseconds per call are:

| Lisp | Single-float whole buffers | Single-float compact buffers | Double-float whole buffers | Double-float compact buffers |
|---|---:|---:|---:|---:|
| SBCL | 92.14 | 0.62 | 91.48 | 0.63 |
| CCL | 572.47 | 2.38 | 606.65 | 2.37 |
| ECL | 28,389.00 | 22.30 | 28,820.50 | 22.80 |

Whole-vector controls remain comparable or faster: 65,536-element single-float
copy additions take 164.63 µs on SBCL, 414.75 µs on CCL, and 37,099.50 µs on ECL,
versus 161.14, 771.58, and 38,067.00 µs with whole buffers. Measurements use
Apple M1 and the Lisp versions above. The baseline is commit `5bbddc2`.

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

[^x86-fma]: The C profile disables hardware dispatch in its software executable;
    the platform math library may itself use hardware FMA. The first C table
    uses a separate GitHub profiling job from the SBCL SIMD table.
