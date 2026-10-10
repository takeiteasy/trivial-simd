# Benchmark results

On the measured Apple M1, SBCL native pointer access beats typed scalar Lisp loops
for 1,024- and 65,536-element arrays. Backend calls are slower than scalar loops
at 32 elements; short float vectors [run inline](small-arrays.md) instead, and
copying inputs and outputs costs more than the scalar loop in these cases.
Native sum kernels avoid an intermediate vector. Reusable spill scratch does
not meet the 10% adoption threshold, so normal calls retain per-call allocation.

The array tables are measured on 2026-10-02 at implementation commit `1061ab8`
with SBCL 2.6.8 on macOS ARM64. The sum tables and the CCL 1.13 and ECL 26.5.5
comparison are measured on 2026-09-29 at commit `8dd1cdb`. The main tables use
SBCL, native NEON, and pointer array access.
Times are microseconds per complete warmed call; speedup is the scalar time
divided by native time.[^timing] These are machine-specific
measurements, rather than performance guarantees.

## Array operations

Addition writes `a+b`. Multiply-add writes `a*b+c` with separate multiplication
and addition. Dot returns the sum of products. Scalar loops and the library
use identical typed arrays; output buffers are reused.[^scalar]

### single-float

| Elements | Operation | Scalar | Lisp fallback | Native | Native copy | Native speedup |
|---:|---|---:|---:|---:|---:|---:|
| 32 | add | 0.029 | 0.120 | 0.086 | 0.532 | 0.34x |
| 32 | multiply-add | 0.036 | 0.164 | 0.179 | 0.375 | 0.20x |
| 32 | dot | 0.024 | 0.077 | 0.068 | 0.342 | 0.36x |
| 1,024 | add | 0.967 | 2.184 | 0.215 | 2.869 | 4.50x |
| 1,024 | multiply-add | 1.113 | 3.051 | 0.401 | 3.593 | 2.78x |
| 1,024 | dot | 0.939 | 1.499 | 0.301 | 2.024 | 3.12x |
| 65,536 | add | 61.858 | 135.479 | 8.286 | 154.354 | 7.47x |
| 65,536 | multiply-add | 67.367 | 191.727 | 16.236 | 208.926 | 4.15x |
| 65,536 | dot | 61.520 | 93.853 | 15.477 | 112.428 | 3.97x |

### double-float

| Elements | Operation | Scalar | Lisp fallback | Native | Native copy | Native speedup |
|---:|---|---:|---:|---:|---:|---:|
| 32 | add | 0.029 | 0.119 | 0.097 | 0.539 | 0.29x |
| 32 | multiply-add | 0.036 | 0.155 | 0.193 | 0.389 | 0.18x |
| 32 | dot | 0.025 | 0.074 | 0.063 | 0.348 | 0.39x |
| 1,024 | add | 0.970 | 2.179 | 0.318 | 2.971 | 3.05x |
| 1,024 | multiply-add | 1.168 | 3.068 | 0.650 | 3.881 | 1.80x |
| 1,024 | dot | 0.938 | 1.498 | 0.527 | 2.269 | 1.78x |
| 65,536 | add | 61.267 | 134.984 | 16.520 | 163.342 | 3.71x |
| 65,536 | multiply-add | 65.827 | 191.576 | 33.482 | 245.086 | 1.97x |
| 65,536 | dot | 61.435 | 93.727 | 30.779 | 128.018 | 2.00x |

## Call overhead

A call with no slice or stride keywords validates its operands and goes straight
to the backend. This holds for arithmetic, `sum`, `dot`, min/max, clamp,
comparison, selection and the unary operations. Small arrays remain dominated by
that fixed cost: a scalar loop is faster than any backend up to a few hundred
elements. Float vectors of up to 64 elements [run inline](small-arrays.md)
instead, at about the cost of a scalar loop; the Inline column shows a dash above
that length.[^overhead]

| Elements | Operation | Scalar | Inline | Lisp fallback | Native |
|---:|---|---:|---:|---:|---:|
| 4 | add | 0.011 | 0.010 | 0.056 | 0.083 |
| 4 | dot | 0.009 | 0.009 | 0.035 | 0.063 |
| 4 | min | - | 0.010 | 0.130 | 0.193 |
| 4 | compare | - | - | 0.125 | 0.182 |
| 32 | add | 0.029 | 0.025 | 0.119 | 0.088 |
| 32 | dot | 0.040 | 0.019 | 0.073 | 0.070 |
| 32 | min | - | 0.025 | 0.176 | 0.192 |
| 32 | compare | - | - | 0.162 | 0.182 |
| 1,024 | add | 0.842 | - | 2.180 | 0.212 |
| 1,024 | dot | 0.939 | - | 1.501 | 0.300 |
| 1,024 | min | - | - | 1.983 | 0.254 |
| 1,024 | compare | - | - | 1.809 | 0.237 |

Calls with `:start`, `:end`, `:stride` or per-vector keywords resolve slices
first and cost more.

## Lisp implementation comparison

Native pointer access for 1,024-element arrays on the same Apple M1. Cells show
microseconds per call, followed by speedup over that implementation's typed
scalar loop.[^implementations]

| Type | Operation | SBCL 2.6.8 | CCL 1.13 | ECL 26.5.5 |
|---|---|---:|---:|---:|
| single-float | add | 0.192 (5.04x) | 0.380 (11.86x) | 0.861 (32.09x) |
| single-float | multiply-add | 0.354 (3.14x) | 0.676 (8.09x) | 1.424 (28.18x) |
| single-float | dot | 0.297 (3.16x) | 0.429 (11.94x) | 0.820 (109.58x) |
| double-float | add | 0.297 (3.25x) | 0.490 (9.19x) | 1.063 (25.97x) |
| double-float | multiply-add | 0.607 (1.92x) | 0.913 (6.00x) | 1.831 (22.32x) |
| double-float | dot | 0.531 (1.77x) | 0.671 (7.64x) | 1.129 (79.65x) |

## Sum kernels

`(trivial-simd:sum (* a b))` returns a scalar. The two-call alternative writes
products into a reused vector, then calls `sum`; bulk `dot` uses a specialised
routine. The native kernel retains the existing block/lane/tail addition order.

| Type | Elements | Lisp kernel | Native kernel | Native multiply+sum | Native dot |
|---|---:|---:|---:|---:|---:|
| single-float | 32 | 0.065 | 0.100 | 0.146 | 0.069 |
| single-float | 1,024 | 1.358 | 0.262 | 0.483 | 0.298 |
| single-float | 65,536 | 84.656 | 10.719 | 23.753 | 15.436 |
| double-float | 32 | 0.070 | 0.103 | 0.147 | 0.069 |
| double-float | 1,024 | 1.365 | 0.445 | 0.815 | 0.531 |
| double-float | 65,536 | 84.796 | 21.598 | 50.220 | 30.774 |

See [kernel performance](kernel-performance.md) for multiply-add, true FMA, and
implementation-specific comparisons.

## Native reduction comparison

Direct C execution of identical `sum(a*a)` bytecode compares the current
terminal reduction with the materialised-output baseline.[^baseline] At 65,536
elements, the measured reduction takes 45.5% less time for single-float and
50.1% less time for double-float.

| Type | Elements | Materialised baseline | Terminal reduction | Speedup |
|---|---:|---:|---:|---:|
| single-float | 32 | 0.015 | 0.012 | 1.23x |
| single-float | 1,024 | 0.300 | 0.173 | 1.74x |
| single-float | 65,536 | 18.967 | 10.342 | 1.83x |
| double-float | 32 | 0.020 | 0.014 | 1.39x |
| double-float | 1,024 | 0.642 | 0.345 | 1.86x |
| double-float | 65,536 | 42.785 | 21.342 | 2.00x |

## Copy, fill and swap

`copy!` uses `replace`, which matches libc `memmove` on SBCL at 1,024 and
65,536 elements, so no native copy exists. `fill!` and `swap!` have native C
paths that win above a size threshold. Cells show microseconds per call
on SBCL 2.6.8, Apple M1, measured 2026-10-01 with `single-float` vectors.

| Elements | `fill` | Native fill | Typed swap | Native swap |
|---:|---:|---:|---:|---:|
| 16 | 0.063 | 0.266 | 0.210 | 0.168 |
| 256 | 0.139 | 0.286 | 0.586 | 0.192 |
| 1,024 | 0.304 | 0.309 | 1.774 | 0.277 |
| 65,536 | 16.446 | 4.200 | 100.272 | 10.144 |

Native fill breaks even near 4,096 bytes and swap near 64 bytes, for any
element type. Native call setup is the fixed cost.

## Strided access

Microseconds per call on SBCL 2.6.8, Apple M1, `single-float`, measured
2026-10-01. Strided calls gather into temporaries, run the contiguous kernel
and scatter outputs.

| Backend | Elements | Stride | `sum` | `dot` | `add!` |
|---|---:|---:|---:|---:|---:|
| native | 1,024 | 1 | 0.36 | 0.40 | 0.38 |
| native | 1,024 | 2 | 1.14 | 1.89 | 2.63 |
| native | 1,024 | -1 | 1.15 | 1.89 | 2.64 |
| native | 65,536 | 1 | 16.05 | 16.10 | 8.75 |
| native | 65,536 | 2 | 63.16 | 237.01 | 194.37 |
| lisp | 1,024 | 1 | 1.11 | 1.71 | 21.29 |
| lisp | 1,024 | 2 | 1.88 | 3.30 | 23.40 |
| lisp | 65,536 | 1 | 61.75 | 93.84 | 1258.39 |
| lisp | 65,536 | 2 | 107.15 | 185.16 | 1403.02 |

Strided calls cost 1.1x to 22x their contiguous equivalents; the native
backend pays most because the contiguous kernels are fastest. Contiguous short
calls pay about 0.02 to 0.05 us for stride handling: `sum` of 8 elements takes
0.08 us and `add!` 0.18 us on the native backend.

## Spill scratch evaluation

Five independent SBCL runs compare complete calls with per-call allocation and
an explicitly owned cache. Across the 32/1,024-element spilling cases, the
cache is 0.26% slower on the geometric mean. Individual runs range from 0.32%
faster to 0.56% slower. This falls below the 10% improvement gate.

See [spill profiling](kernel-spilling.md) for workload sizes, direct C versus
complete Lisp timings, concurrent calls, and the storage design.

## Reproducing

Run [the array benchmark](testing.md#benchmark), the call-overhead benchmark and
[the kernel profile](testing.md#kernel-profiling). Results change with the Lisp
compiler, CPU, memory access mode, and system load.

## Limitations

- These results cover one Apple M1 using SBCL, CCL, and ECL. They do not
  measure x86-64 SBCL SIMD or other CPUs.
- Strided calls gather into temporaries;
  see [#49](https://github.com/communal-software/trivial-simd/issues/49).
- Short kernel setup and copy-mode costs remain; see
  [kernel limitations](kernels.md#limitations) and
  [array access limitations](backends.md#limitations).

[^overhead]: Single-float, from `tests/overhead-bench.lisp` on SBCL 2.6.8 and
    macOS ARM64 on 2026-10-02, using the same timing method as the tables above.
    Raw output: [SBCL](benchmark-runs/2026-10-02-call-overhead-sbcl.txt).

[^timing]: Arrays and programs are prepared outside timing. Each case warms up,
    calibrates batches to at least 50 ms, and reports the median of three
    further batches. The array tables come from `tests/bench.lisp` at
    implementation commit `1061ab8` on 2026-10-02, which calls the functions
    through `funcall` so the backends are measured; the sum tables and the CCL
    and ECL comparison come from commit `8dd1cdb` on 2026-09-29. Results are
    checked before timing. Values are rounded for display. Raw output:
    [SBCL](benchmark-runs/2026-10-02-sbcl.txt) (arrays),
    [SBCL](benchmark-runs/2026-09-29-sbcl.txt),
    [CCL](benchmark-runs/2026-09-29-ccl.txt),
    [ECL](benchmark-runs/2026-09-29-ecl.txt).

[^scalar]: Scalar functions are compiled typed Lisp loops without explicit
    SIMD. The compiler may still emit vector instructions. `Native copy`
    includes input and output copying; `Native` uses direct array pointers.

[^baseline]: The baseline native source is commit `6289ec4`, compiled with the
    same optimisation settings. Separate foreign symbols ensure the libraries
    remain distinct. Direct C measurements exclude Lisp setup and use calibrated
    CPU-time batches. The figures come from one complete kernel-profile run.

[^implementations]: CCL and ECL benchmarks are compiled before loading, using
    the commands in the testing guide. Each implementation has one complete
    benchmark run; the runs are sequential to avoid competing measurements.
