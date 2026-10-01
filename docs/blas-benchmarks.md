# BLAS benchmarks

Real Level 2 and 3 routines run native SIMD kernels above a size threshold. On
SBCL at 256 × 256 they are 3× to 20× faster than the typed Lisp kernels and 1.2×
to 9× slower than system CBLAS, which uses the M1 matrix coprocessor for
Level 3.[^scope] On ECL, native `dgemm` at 64 × 64 takes 17.8 µs against 12.3 ms.

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
| `dgemm` | 271 | 16.3 | 2.48 | 17× |
| `dsyrk` | 129 | 15.1 | 3.16 | 8.6× |
| `dtrsm` | 129 | 15.5 | 4.80 | 8.3× |
| `dgemv` | 2.53 | 0.91 | 0.44 | 2.8× |
| `dtrsv` | 2.69 | 1.49 | 0.86 | 1.8× |
| `dger` | 3.50 | 1.06 | 0.61 | 3.3× |
| `dgbmv` | 0.72 | 0.75 | 0.21 | Lisp |

### CCL 1.13

| Routine | Lisp | Native | CBLAS | Speedup |
|---|---:|---:|---:|---:|
| `dgemm` | 384 | 16.6 | 2.55 | 23× |
| `dsyrk` | 624 | 15.9 | 3.29 | 39× |
| `dtrsm` | 579 | 25.9 | 4.70 | 22× |
| `dgemv` | 9.84 | 1.94 | 0.66 | 5.1× |
| `dtrsv` | 11.9 | 2.44 | 1.03 | 4.9× |
| `dger` | 6.77 | 2.22 | 0.83 | 3.0× |
| `dgbmv` | 3.88 | 3.90 | 0.43 | Lisp |

### ECL 26.5.5

| Routine | Lisp | Native | CBLAS | Speedup |
|---|---:|---:|---:|---:|
| `dgemm` | 12,303 | 17.8 | 12.9 | 690× |
| `dsyrk` | 5,268 | 16.8 | 13.1 | 310× |
| `dtrsm` | 9,642 | 18.1 | 17.0 | 530× |
| `dgemv` | 208 | 2.09 | 9.82 | 99× |
| `dtrsv` | 147 | 2.33 | 12.0 | 63× |
| `dger` | 187 | 2.22 | 9.62 | 84× |
| `dgbmv` | 27.7 | 30.2 | 9.91 | Lisp |

## Other sizes (SBCL)

| Routine | 16 × 16 | 16 × 16 CBLAS | 256 × 256 | 256 × 256 CBLAS |
|---|---:|---:|---:|---:|
| `dgemm` | 0.79 | 0.43 | 827 | 107 |
| `dsyrk` | 0.87 | 0.70 | 649 | 70.6 |
| `dtrsm` | 0.99 | 0.45 | 664 | 128 |
| `dgemv` | 0.49 | 0.14 | 11.0 | 4.05 |
| `dtrsv` | 0.45 | 0.23 | 10.1 | 7.12 |
| `dger` | 0.44 | 0.13 | 16.5 | 14.1 |
| `dgbmv` | 0.40 | 0.13 | 2.12 | 0.57 |

Row-major times; column-major follows the same pattern.[^trsv] Level 2 at 16 × 16 and `dgbmv` run the Lisp kernels. All
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

[^scope]: The benchmark covers `dgemm`, `dsyrk`, `dtrsm`, `dgemv`, `dgbmv`,
    `dtrsv`, and `dger`. Other real routines in those families and `sgemm`
    share the same native code per precision.
[^trsv]: Column-major `dtrsv` at 256 × 256 takes 9.0 µs natively against 31.9 µs
    in Lisp; see
    [`benchmark-runs/2026-10-01-blas-native-sbcl-trsv.txt`](benchmark-runs/2026-10-01-blas-native-sbcl-trsv.txt).
[^x86]: Logs:
    [`benchmark-runs/2026-10-01-blas-x86-sbcl.txt`](benchmark-runs/2026-10-01-blas-x86-sbcl.txt).
[^sweep]: Twelve tile and blocking configurations, two runs each, in
    [`benchmark-runs/2026-10-01-blas-x86-sweep.txt`](benchmark-runs/2026-10-01-blas-x86-sweep.txt).
    On the same CPU the AVX+FMA kernels run `dgemm` at 64 × 64 in 10.8 µs
    against 24.3 µs for SSE2, and at 512 × 512 at 117 GFLOP/s.
[^method]: Each time is the median of three batches of at least 50 ms.
    In-place routines (`dtrsm`, `dtrsv`) restore their operand before each
    call in both columns. CBLAS calls run with floating-point traps masked.
    ECL's CBLAS times have a floor of about 9 µs from the foreign call.
