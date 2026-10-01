# Benchmark results

On the measured Apple M1, SBCL native pointer access beats typed scalar Lisp loops
for 1,024- and 65,536-element arrays. Scalar loops are faster at 32 elements;
copying inputs and outputs costs more than the scalar loop in these cases.
Native sum kernels avoid an intermediate vector. Reusable spill scratch does
not meet the 10% adoption threshold, so normal calls retain per-call allocation.

Array and sum tables are measured on 2026-09-29 at implementation commit
`8dd1cdb`, with SBCL 2.6.8, CCL 1.13, and ECL 26.5.5 on macOS ARM64. The
main array and sum tables use SBCL, native NEON, and pointer array access.
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
| 32 | add | 0.029 | 0.699 | 0.082 | 0.520 | 0.36x |
| 32 | multiply-add | 0.035 | 0.135 | 0.132 | 0.338 | 0.27x |
| 32 | dot | 0.024 | 0.093 | 0.067 | 0.346 | 0.36x |
| 1,024 | add | 0.966 | 19.462 | 0.192 | 2.856 | 5.04x |
| 1,024 | multiply-add | 1.111 | 3.052 | 0.354 | 3.532 | 3.14x |
| 1,024 | dot | 0.938 | 1.573 | 0.297 | 2.038 | 3.16x |
| 65,536 | add | 61.196 | 1242.797 | 8.286 | 152.486 | 7.39x |
| 65,536 | multiply-add | 67.269 | 192.742 | 16.069 | 207.656 | 4.19x |
| 65,536 | dot | 61.423 | 100.529 | 15.440 | 110.412 | 3.98x |

### double-float

| Elements | Operation | Scalar | Lisp fallback | Native | Native copy | Native speedup |
|---:|---|---:|---:|---:|---:|---:|
| 32 | add | 0.028 | 0.823 | 0.084 | 0.522 | 0.34x |
| 32 | multiply-add | 0.035 | 0.136 | 0.137 | 0.343 | 0.26x |
| 32 | dot | 0.025 | 0.093 | 0.068 | 0.347 | 0.37x |
| 1,024 | add | 0.967 | 28.096 | 0.297 | 2.918 | 3.25x |
| 1,024 | multiply-add | 1.163 | 3.061 | 0.607 | 3.749 | 1.92x |
| 1,024 | dot | 0.938 | 1.546 | 0.531 | 2.248 | 1.77x |
| 65,536 | add | 61.071 | 1826.875 | 16.417 | 159.283 | 3.72x |
| 65,536 | multiply-add | 65.713 | 192.531 | 31.726 | 314.398 | 2.07x |
| 65,536 | dot | 61.415 | 97.091 | 30.808 | 139.016 | 1.99x |

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

- These results cover one Apple M1 using SBCL, CCL, and ECL. They do not
  measure x86-64 SBCL SIMD or other CPUs.
- Short kernel setup and copy-mode costs remain; see
  [kernel limitations](kernels.md#limitations) and
  [array access limitations](backends.md#limitations).

[^timing]: Arrays and programs are prepared outside timing. Each case warms up,
    calibrates batches to at least 50 ms, and reports the median of three
    further batches. Array and sum tables come from `tests/bench.lisp` at
    implementation commit `8dd1cdb` on 2026-09-29; results are checked before
    timing. Values are rounded for display. Raw output:
    [SBCL](benchmark-runs/2026-09-29-sbcl.txt),
    [CCL](benchmark-runs/2026-09-29-ccl.txt),
    [ECL](benchmark-runs/2026-09-29-ecl.txt). The Lisp fallback `dot` cells come from
    the 2026-10-01 [run](benchmark-runs/2026-10-01-kernel-reducers-sbcl.txt), after the
    typed Lisp loops.

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
