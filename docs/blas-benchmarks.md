# BLAS benchmarks

Level 2 and 3 routines run typed Lisp kernels that are 4× to 6,600× faster
than the earlier checked-access loops. On SBCL, Level 2 is 2× to 10× slower
than system CBLAS and Level 3 is 3× to 200× slower, widening with matrix
size.[^scope] CBLAS on the M1 uses the matrix coprocessor for Level 3.

Times are microseconds per warmed call on an Apple M1 (macOS ARM64) against
Accelerate CBLAS, measured on 2026-09-30. They are machine-specific
measurements, rather than performance guarantees.[^method]

## Run it

```sh
BLAS_BENCH_SIZES="16 64 256" sbcl --script tests/blas-bench.lisp
```

The script prints one row per routine, layout, and size. `ccl` and `ecl`
accept the same script. Recorded runs are in
[`benchmark-runs/`](benchmark-runs/).

## 64 × 64 results

Before is the earlier implementation; speedup is before divided by after for
row-major views.

### SBCL 2.6.8

| Routine | Before | After row-major | After column-major | CBLAS | Speedup |
|---|---:|---:|---:|---:|---:|
| `dgemm` | 22,284 | 285 | 244 | 2.39 | 78× |
| `dsyrk` | 11,968 | 132 | 129 | 3.05 | 91× |
| `dtrsm` | 7,348 | 131 | 131 | 4.51 | 56× |
| `dgemv` | 271 | 2.50 | 4.54 | 0.43 | 108× |
| `dgbmv` | 295 | 0.72 | 0.79 | 0.20 | 410× |
| `dtrsv` | 92.3 | 2.67 | 2.38 | 0.82 | 35× |
| `dger` | 374 | 3.46 | 3.16 | 0.59 | 108× |

### CCL 1.13

| Routine | Before | After row-major | After column-major | CBLAS | Speedup |
|---|---:|---:|---:|---:|---:|
| `dgemm` | 2,501,579 | 381 | 397 | 1.97 | 6,569× |
| `dsyrk` | 1,269,536 | 626 | 624 | 3.05 | 2,027× |
| `dtrsm` | 631,006 | 579 | 581 | 4.48 | 1,089× |
| `dgemv` | 20,011 | 9.77 | 9.14 | 0.36 | 2,049× |
| `dgbmv` | 19,979 | 3.85 | 3.46 | 0.14 | 5,184× |
| `dtrsv` | 9,806 | 11.9 | 6.97 | 0.79 | 826× |
| `dger` | 552 | 6.74 | 6.75 | 0.54 | 82× |

### ECL 26.5.5

| Routine | Before | After row-major | After column-major | CBLAS | Speedup |
|---|---:|---:|---:|---:|---:|
| `dgemm` | 191,437 | 13,512 | 13,539 | 12.7 | 14× |
| `dsyrk` | 98,661 | 5,330 | 5,345 | 13.0 | 19× |
| `dtrsm` | 58,059 | 10,552 | 10,366 | 16.8 | 6× |
| `dgemv` | 1,696 | 218 | 241 | 9.80 | 8× |
| `dgbmv` | 1,596 | 29.7 | 32.2 | 9.47 | 54× |
| `dtrsv` | 904 | 156 | 194 | 12.1 | 6× |
| `dger` | 2,060 | 203 | 206 | 9.48 | 10× |

## Other sizes (SBCL)

| Routine | 16 × 16 | 16 × 16 CBLAS | 256 × 256 | 256 × 256 CBLAS |
|---|---:|---:|---:|---:|
| `dgemm` | 3.89 | 0.36 | 16,303 | 99.5 |
| `dsyrk` | 2.32 | 0.67 | 7,537 | 65.8 |
| `dtrsm` | 2.13 | 0.43 | 7,467 | 116 |
| `dgemv` | 0.45 | 0.14 | 34.3 | 3.89 |
| `dgbmv` | 0.40 | 0.13 | 2.02 | 0.55 |
| `dtrsv` | 0.44 | 0.24 | 33.5 | 7.30 |
| `dger` | 0.43 | 0.13 | 50.4 | 12.6 |

Row-major times. Column-major times follow the same pattern; `dgemv` differs
most (up to 80%) because the dot and update loops swap roles. All layouts are
in the recorded runs.

## Limitations

Kernels are scalar Lisp loops without cache blocking or SIMD. ECL boxes
double-floats unless each array read is bound to a typed variable, which keeps
its times far above SBCL and CCL. Native kernels are tracked by
[#79](https://todo.sr.ht/~takeiteasy/trivial-simd/79).

[^scope]: The benchmark covers `dgemm`, `dsyrk`, `dtrsm`, `dgemv`, `dgbmv`,
    `dtrsv`, and `dger`. Other precisions and families share the same
    kernels per precision.
[^method]: Each time is the median of three batches of at least 50 ms.
    In-place routines (`dtrsm`, `dtrsv`) restore their operand before each
    call in both columns. CBLAS calls run with floating-point traps masked.
    ECL's CBLAS times have a floor of about 9 µs from the foreign call.
