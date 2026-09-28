# Benchmark results

On the measured Apple M1, native pointer access beats typed scalar Lisp loops
for 1,024- and 65,536-element arrays. Scalar loops are faster at 32 elements;
copying inputs and outputs costs more than the scalar loop in these cases.
Native sum kernels avoid an intermediate vector. Reusable spill scratch does
not meet the 10% adoption threshold, so normal calls retain per-call allocation.

Measured on 2026-09-28 with SBCL 2.6.8, macOS ARM64, native NEON and pointer
array access. Times are microseconds per complete warmed call; speedup is the
scalar time divided by native time.[^timing] These are machine-specific
measurements, rather than performance guarantees.

## Array operations

Addition writes `a+b`. Multiply-add writes `a*b+c` with separate multiplication
and addition. Dot returns the sum of products. Scalar loops and the library
use identical typed arrays; output buffers are reused.[^scalar]

### single-float

| Elements | Operation | Scalar | Lisp fallback | Native | Native copy | Native speedup |
|---:|---|---:|---:|---:|---:|---:|
| 32 | add | 0.030 | 0.725 | 0.092 | 0.590 | 0.33x |
| 32 | multiply-add | 0.037 | 0.133 | 0.137 | 0.386 | 0.27x |
| 32 | dot | 0.025 | 0.790 | 0.068 | 0.367 | 0.37x |
| 1,024 | add | 1.000 | 20.102 | 0.208 | 2.921 | 4.80x |
| 1,024 | multiply-add | 1.150 | 2.880 | 0.372 | 3.451 | 3.09x |
| 1,024 | dot | 0.971 | 21.800 | 0.306 | 1.842 | 3.17x |
| 65,536 | add | 62.896 | 1289.813 | 8.633 | 155.162 | 7.29x |
| 65,536 | multiply-add | 69.735 | 178.365 | 16.646 | 207.527 | 4.19x |
| 65,536 | dot | 63.588 | 1188.406 | 15.974 | 104.033 | 3.98x |

### double-float

| Elements | Operation | Scalar | Lisp fallback | Native | Native copy | Native speedup |
|---:|---|---:|---:|---:|---:|---:|
| 32 | add | 0.030 | 0.813 | 0.091 | 0.597 | 0.33x |
| 32 | multiply-add | 0.040 | 0.129 | 0.142 | 0.399 | 0.28x |
| 32 | dot | 0.026 | 0.808 | 0.069 | 0.373 | 0.37x |
| 1,024 | add | 1.001 | 25.079 | 0.317 | 3.019 | 3.16x |
| 1,024 | multiply-add | 1.247 | 2.835 | 0.624 | 3.711 | 2.00x |
| 1,024 | dot | 0.972 | 25.536 | 0.549 | 2.103 | 1.77x |
| 65,536 | add | 63.132 | 1654.094 | 13.051 | 162.793 | 4.84x |
| 65,536 | multiply-add | 77.707 | 178.174 | 32.658 | 282.605 | 2.38x |
| 65,536 | dot | 63.563 | 1695.531 | 31.865 | 119.480 | 1.99x |

## Lisp implementation comparison

Native pointer access for 1,024-element arrays on the same Apple M1. Cells show
microseconds per call, followed by speedup over that implementation's typed
scalar loop.[^implementations]

| Type | Operation | SBCL 2.6.8 | CCL 1.13 | ECL 26.5.5 |
|---|---|---:|---:|---:|
| single-float | add | 0.208 (4.80x) | 0.386 (11.68x) | 0.867 (31.61x) |
| single-float | multiply-add | 0.372 (3.09x) | 0.671 (8.22x) | 1.410 (28.75x) |
| single-float | dot | 0.306 (3.17x) | 0.443 (11.56x) | 0.779 (114.41x) |
| double-float | add | 0.317 (3.16x) | 0.494 (9.12x) | 1.042 (26.38x) |
| double-float | multiply-add | 0.624 (2.00x) | 0.911 (6.00x) | 1.782 (23.16x) |
| double-float | dot | 0.549 (1.77x) | 0.672 (7.92x) | 1.068 (83.04x) |

## Sum kernels

`(trivial-simd:sum (* a b))` returns a scalar. The two-call alternative writes
products into a reused vector, then calls `sum`; bulk `dot` uses a specialised
routine. The native kernel retains the existing block/lane/tail addition order.

| Type | Elements | Lisp kernel | Native kernel | Native multiply+sum | Native dot |
|---|---:|---:|---:|---:|---:|
| single-float | 32 | 0.064 | 0.106 | 0.148 | 0.066 |
| single-float | 1,024 | 1.206 | 0.269 | 0.486 | 0.301 |
| single-float | 65,536 | 75.027 | 10.722 | 23.751 | 15.436 |
| double-float | 32 | 0.066 | 0.121 | 0.150 | 0.069 |
| double-float | 1,024 | 1.219 | 0.461 | 0.819 | 0.532 |
| double-float | 65,536 | 75.720 | 21.668 | 47.220 | 32.499 |

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

## Spill scratch evaluation

Five independent SBCL runs compare complete calls with per-call allocation and
an explicitly owned cache. Across the 32/1,024-element spilling cases, the
cache is 0.26% slower on the geometric mean. Individual runs range from 0.32%
faster to 0.56% slower. This falls below the 10% improvement gate.

See [spill profiling](kernel-spilling.md) for workload sizes, direct C versus
complete Lisp timings, concurrent calls, and the storage design.

## Reproducing

Run [the array benchmark](testing.md#benchmark) and
[the kernel profile](testing.md#kernel-profiling). Results change with the Lisp
compiler, CPU, memory access mode, and system load.

## Limitations

- These results cover one Apple M1 using SBCL and native pointer access. They
  do not measure x86-64 SBCL SIMD or other CPUs.
- Short kernel setup and copy-mode costs remain; see
  [kernel limitations](kernels.md#limitations) and
  [array access limitations](backends.md#limitations).

[^timing]: Arrays and programs are prepared outside timing. Each case warms up,
    calibrates batches to at least 50 ms, and reports the median of three
    further batches. The array tables come from one complete run of
    `tests/bench.lisp` at implementation commit `fcfab2b`; results are checked
    before timing. Values are rounded for display.

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
