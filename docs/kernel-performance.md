# Kernel performance

Kernels avoid intermediate vectors. Typed Lisp kernels are much faster than
separate Lisp bulk operations in these measurements; native kernels perform
similarly to native bulk operations for simple expressions. Native reductions
may cost more at small sizes.

Multiply-add and sum-of-products tables are measured on 2026-09-29 at
implementation commit `8dd1cdb`, on Apple M1 with SBCL 2.6.8. Times are
microseconds per call; `single-float` unless a type is listed.[^snapshot]
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
| 32 | 0.135 | 1.334 | 0.151 | 0.199 |
| 1,024 | 3.047 | 38.524 | 0.391 | 0.397 |
| 65,536 | 192.609 | 2476.125 | 16.062 | 15.958 |

## True FMA

`(trivial-simd:fma a b c)` guarantees one rounding step. ARM64 Lisp kernels use
in-process scalar FMA on guarded SBCL, CCL, and ECL versions. See
[ARM64 FMA](arm64-fma.md) for compiler support, the five-process performance gate,
and complete elementwise and sum measurements. Unsupported versions use optional
scalar C helpers or exact integer arithmetic. SBCL x86 uses guarded scalar and
packed FMA; native ARM64 uses NEON and native x86 selects hardware FMA at runtime.

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
| single-float | 32 | 0.065 | 0.776 | 0.100 | 0.146 | 0.069 |
| single-float | 1,024 | 1.358 | 21.190 | 0.262 | 0.483 | 0.298 |
| single-float | 65,536 | 84.656 | 1419.922 | 10.719 | 23.753 | 15.436 |
| double-float | 32 | 0.070 | 0.952 | 0.103 | 0.147 | 0.069 |
| double-float | 1,024 | 1.365 | 28.540 | 0.445 | 0.815 | 0.531 |
| double-float | 65,536 | 84.796 | 1926.000 | 21.598 | 50.220 | 30.774 |

Native sums evaluate final instructions directly into lane accumulators. Each
256-element block retains its lane, tail, and block addition order, without an
output-block store/read. Short calls still include pointer and scalar-output
setup; see [kernel limitations](kernels.md#limitations).

## Reducers

`(argmax (* a b))`, `(asum (* a b))` and `(nrm2 (* a b))` return a scalar without
an intermediate vector. Each is compared with `multiply!` into a reusable
vector followed by the bulk reducer. Microseconds per call at 65,536 elements:[^reducers]

| Type | Reducer | Lisp kernel | Lisp multiply+bulk | Native kernel | Native multiply+bulk |
|---|---|---:|---:|---:|---:|
| single-float | argmax | 89.0 | 1505.2 | 29.0 | 40.4 |
| single-float | asum | 91.3 | 1531.2 | 15.9 | 24.9 |
| single-float | nrm2 | 92.0 | 1587.3 | 38.6 | 30.1 |
| double-float | argmax | 86.4 | 2068.2 | 37.4 | 48.6 |
| double-float | asum | 93.4 | 2171.8 | 32.3 | 52.5 |
| double-float | nrm2 | 97.4 | 2163.6 | 44.3 | 51.4 |

Native reducers evaluate each 256-element block, then reduce it with eight
independent lanes. The single-float `nrm2` kernel converts to double before
squaring, so it trails the staged form; see
[kernel limitations](kernels.md#limitations).

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

ECL 26.5.5 on Apple M1, measured on 2026-09-29 at implementation commit
`8dd1cdb`, shares setup helpers across definitions with the same
input count and reduction mode. A new definition using an existing helper avoids
compilation; its first native call still allocates its program buffers.
See [kernel limitations](kernels.md#limitations) for cold compilation.

First-call milliseconds for 32 elements, using five-trial medians:

| Definition | Kernel | Float | Cold signature | Shared signature |
|---|---|---|---:|---:|
| Compiled | Elementwise | single | 369.568 | 0.191 |
| Compiled | Elementwise | double | 368.597 | 0.190 |
| Compiled | Sum | single | 367.389 | 0.148 |
| Compiled | Sum | double | 372.691 | 0.195 |
| Eval | Elementwise | single | 367.458 | 0.192 |
| Eval | Elementwise | double | 367.717 | 0.158 |
| Eval | Sum | single | 361.293 | 0.166 |
| Eval | Sum | double | 365.651 | 0.193 |

Warmed microseconds per call for definitions using the shared helper:

| Kernel | Float | Elements | Compiled | Eval |
|---|---|---:|---:|---:|
| Elementwise | single | 32 | 1.129 | 1.601 |
| Elementwise | single | 1,024 | 1.345 | 1.836 |
| Elementwise | single | 65,536 | 17.475 | 18.102 |
| Elementwise | double | 32 | 1.265 | 1.758 |
| Elementwise | double | 1,024 | 1.760 | 2.232 |
| Elementwise | double | 65,536 | 33.050 | 33.450 |
| Sum | single | 32 | 1.011 | 1.546 |
| Sum | single | 1,024 | 1.355 | 1.772 |
| Sum | single | 65,536 | 21.377 | 21.844 |
| Sum | double | 32 | 1.126 | 1.586 |
| Sum | double | 1,024 | 1.803 | 2.270 |
| Sum | double | 65,536 | 45.002 | 43.877 |

The elementwise expression is `(- (* a b) c)`; the sum wraps that expression
in `sum`. Each helper is first initialized by a different definition using
`(+ (* a b) c)`. Inputs are identical typed vectors and all results are checked
outside timing. Cold trials use fresh caches and definitions; warm values are
medians of three calibrated batches. The first-call clock resolution is 1 µs.[^snapshot]

[^x86-fma]: The C profile disables hardware dispatch in its software executable;
    the platform math library may itself use hardware FMA. The first C table
    uses a separate GitHub profiling job from the SBCL SIMD table.

[^snapshot]: Each Lisp has one complete `tests/bench.lisp` run at implementation
    commit `8dd1cdb`; runs are sequential. See [raw SBCL output](benchmark-runs/2026-09-29-sbcl.txt)
    and [raw ECL output](benchmark-runs/2026-09-29-ecl.txt). The separate x86 FMA,
    native copy, and performance-gate evaluations retain their own measurements.

[^reducers]: From `tests/bench.lisp` on SBCL ARM64 with the `:native` and `:lisp`
    backends, 2026-10-01: [raw output](benchmark-runs/2026-10-01-kernel-reducers-sbcl.txt).
    The Lisp multiply+sum column of the sum table comes from the same run.
