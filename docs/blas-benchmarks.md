# BLAS benchmarks

Real Level 2 and 3 routines run native SIMD kernels above a size threshold. On
SBCL at 256 × 256 they are 3× to 21× faster than the typed Lisp kernels and 1.2×
to 8× slower than system CBLAS, which uses the M1 matrix coprocessor for
Level 3.[^scope] On ECL, native `dgemm` at 64 × 64 takes 34.6 µs against 14.4 ms.

Times are microseconds per warmed call on an Apple M1 (macOS ARM64) against
Accelerate CBLAS, measured on 2026-10-01. They are machine-specific
measurements, rather than performance guarantees.[^method]

## Run it

```sh
BLAS_BENCH_SIZES="16 64 256" sbcl --script tests/blas-bench.lisp
TRIVIAL_SIMD_BACKEND=lisp BLAS_BENCH_SIZES="16 64 256" sbcl --script tests/blas-bench.lisp
```

The script prints one row per routine, layout, and size. The second command
selects the typed Lisp kernels. `BLAS_BENCH_THRESHOLD=1` sends every size to the
native kernels, to measure below the dispatch threshold. `ccl` and `ecl` accept the same script.
Recorded runs are in [`benchmark-runs/`](benchmark-runs/): `2026-10-01-blas-native-*` and `2026-10-01-blas-lisp-*`.

## 64 × 64 results

Lisp is the typed Lisp kernels; native is the default configuration. Row-major
views.

### SBCL 2.6.8

| Routine | Lisp | Native | CBLAS | Speedup |
|---|---:|---:|---:|---:|
| `dgemm` | 285 | 18.8 | 2.43 | 15× |
| `dsyrk` | 142 | 12.2 | 3.04 | 12× |
| `dtrsm` | 127 | 26.8 | 4.51 | 4.7× |
| `dgemv` | 2.86 | 0.87 | 0.43 | 3.3× |
| `dtrsv` | 3.04 | 1.48 | 0.82 | 2.1× |
| `dger` | 3.91 | 1.03 | 0.59 | 3.8× |
| `dgbmv` | 0.76 | 0.70 | 0.20 | Lisp |

### CCL 1.13

| Routine | Lisp | Native | CBLAS | Speedup |
|---|---:|---:|---:|---:|
| `dgemm` | 408 | 19.9 | 2.68 | 20× |
| `dsyrk` | 701 | 13.1 | 3.28 | 54× |
| `dtrsm` | 644 | 29.8 | 4.74 | 22× |
| `dgemv` | 11.4 | 1.91 | 0.64 | 6.0× |
| `dtrsv` | 12.9 | 2.25 | 1.09 | 5.7× |
| `dger` | 8.59 | 2.21 | 0.82 | 3.9× |
| `dgbmv` | 4.97 | 3.85 | 0.41 | Lisp |

### ECL 26.5.5

| Routine | Lisp | Native | CBLAS | Speedup |
|---|---:|---:|---:|---:|
| `dgemm` | 14,368 | 34.6 | 18.6 | 416× |
| `dsyrk` | 5,820 | 20.8 | 20.8 | 279× |
| `dtrsm` | 10,840 | 28.2 | 23.8 | 384× |
| `dgemv` | 219 | 3.43 | 13.1 | 64× |
| `dtrsv` | 166 | 3.40 | 20.7 | 49× |
| `dger` | 218 | 3.47 | 12.2 | 63× |
| `dgbmv` | 31.3 | 47.0 | 15.2 | Lisp |

## Other sizes (SBCL)

| Routine | 16 × 16 | 16 × 16 CBLAS | 256 × 256 | 256 × 256 CBLAS |
|---|---:|---:|---:|---:|
| `dgemm` | 0.92 | 0.41 | 840 | 100 |
| `dsyrk` | 0.79 | 0.67 | 468 | 65.8 |
| `dtrsm` | 1.09 | 0.43 | 630 | 116 |
| `dgemv` | 0.47 | 0.13 | 10.6 | 4.04 |
| `dtrsv` | 0.44 | 0.22 | 10.1 | 7.15 |
| `dger` | 0.43 | 0.12 | 16.8 | 14.4 |
| `dgbmv` | 0.38 | 0.12 | 2.07 | 0.57 |

Row-major times; column-major follows the same pattern.[^trsv] `dtrsm` takes 0.75× the time of `dgemm` at 256 × 256 on M1.[^trsm] Level 2 at 16 × 16 and `dgbmv` run the Lisp kernels. All
layouts are in the recorded runs.

## x86-64 results

Microseconds per warmed call on SBCL 2.2.9 with an AMD EPYC 9V45 (GitHub
`ubuntu-latest`, AVX+FMA path), row-major, against the system CBLAS.[^x86]

| Routine | 64 × 64 native | 64 × 64 Lisp | 256 × 256 native | 256 × 256 Lisp | 256 × 256 CBLAS |
|---|---:|---:|---:|---:|---:|
| `dgemm` | 11.2 | 129 | 555 | 8,500 | 168 |
| `dsyrk` | 10.7 | 139 | 453 | 8,250 | 135 |
| `dtrsm` | 9.4 | 74.2 | 445 | 4,438 | 195 |
| `dgemv` | 0.55 | 3.42 | 5.31 | 28.3 | 3.48 |
| `dtrsv` | 1.05 | 1.65 | 5.98 | 18.1 | 6.84 |
| `dger` | 0.58 | 1.47 | 7.69 | 21.7 | 3.60 |

The default 8 × 6 (`f64`) and 16 × 6 (`f32`) tiles beat 12 × 4, 8 × 4 and 4 × 8
by 5% to 30%. Depth, row and column blocks stay within 5% of each other, so
the M1 values (256, 128, 1024) apply.[^sweep]

`dtrsm` takes 0.77× the time of `dgemm` at 256 × 256 for `f64` on AVX+FMA and
1.05× for `f32`; at 64 × 64 the ratios are 1.19× and 1.78×.[^x86trsm]

## Thresholds

Native is used when the work is at least the threshold; Lisp is faster below it
on SBCL and CCL, and native wins at every size on ECL.

| Implementation | Level 3 (`m·n·k`) | Level 2 (elements) |
|---|---:|---:|
| SBCL, CCL | 1,000 | 1,000 |
| ECL | 1 | 1 |

On x86-64 native is at least as fast as Lisp from 8 × 8, so the same
thresholds apply.

## Limitations

Complex routines and band, packed, and symmetric Level 2 routines run scalar
Lisp loops, tracked by [#80](https://todo.sr.ht/~takeiteasy/trivial-simd/80) and
[#81](https://todo.sr.ht/~takeiteasy/trivial-simd/81). The Lisp kernels box
double-floats on ECL unless each array read is bound to a typed variable.
On x86-64, native `trsm` is slow relative to `gemm` for `f32` and at 64 × 64,
tracked by [#89](https://todo.sr.ht/~takeiteasy/trivial-simd/89).

[^scope]: The benchmark covers `dgemm`, `dsyrk`, `dtrsm`, `dgemv`, `dgbmv`,
    `dtrsv`, and `dger`. Other real routines in those families and `sgemm`
    share the same native code per precision.
[^trsv]: Column-major `dtrsv` at 256 × 256 takes 9.0 µs natively against 31.9 µs
    in Lisp; see
    [`benchmark-runs/2026-10-01-blas-native-sbcl-trsv.txt`](benchmark-runs/2026-10-01-blas-native-sbcl-trsv.txt).
[^trsm]: Fused solve kernel, 630 µs against 840 µs for `dgemm`; column-major takes 906 µs. See
    [`benchmark-runs/2026-10-01-blas-native-sbcl-fused-trsm.txt`](benchmark-runs/2026-10-01-blas-native-sbcl-fused-trsm.txt).
[^x86]: Logs:
    [`benchmark-runs/2026-10-01-blas-x86-sbcl.txt`](benchmark-runs/2026-10-01-blas-x86-sbcl.txt).
[^x86trsm]: AMD EPYC 9V74, default tiles, in
    [`benchmark-runs/2026-10-01-blas-x86-fused-trsm.txt`](benchmark-runs/2026-10-01-blas-x86-fused-trsm.txt).
[^sweep]: Twelve tile and blocking configurations, two runs each, in
    [`benchmark-runs/2026-10-01-blas-x86-sweep.txt`](benchmark-runs/2026-10-01-blas-x86-sweep.txt).
    On the same CPU the AVX+FMA kernels run `dgemm` at 64 × 64 in 10.8 µs
    against 24.3 µs for SSE2, and at 512 × 512 at 117 GFLOP/s.
[^method]: Each time is the median of three batches of at least 50 ms.
    In-place routines (`dtrsm`, `dtrsv`) restore their operand before each
    call in both columns. CBLAS calls run with floating-point traps masked.
    ECL's CBLAS times have a floor of about 9 µs from the foreign call.
