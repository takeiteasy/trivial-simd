# Inference-stage kernel performance

In the local M1 run, native execution is faster for 4,096-element stages, while
typed Lisp is faster for 32-element softmax. Native setup and repeated row calls
can dominate short workloads.

These measurements cover complete softmax, SiLU, and RoPE table preparation.
They do not measure a model executor or establish a model-level speedup.

## ARM64 results

Apple M1, SBCL 2.6.8, release native build, 2026-10-05. Times are microseconds per
complete stage; native uses direct pointer access.[^method]

| Precision | Stage, 4,096 elements | Typed Lisp | Lisp kernel | Native kernel |
|---|---|---:|---:|---:|
| f32 | Softmax | 39.62 | 88.41 | 18.29 |
| f32 | SiLU | 21.55 | 21.69 | 8.11 |
| f32 | RoPE: both tables | 118.08 | 37.40 | 17.99 |
| f64 | Softmax | 38.53 | 86.89 | 28.91 |
| f64 | SiLU | 20.50 | 20.84 | 13.31 |
| f64 | RoPE: both tables | 117.63 | 35.81 | 22.04 |

The typed softmax reference stores exponentials in the destination before
normalization. The kernel uses three dependency passes and evaluates each
exponential twice. SiLU uses one kernel pass; RoPE measures separate sine and
cosine output kernels.

## Short vectors and rows

Softmax times include all rows and every pass:

| Precision | Rows × width | Typed Lisp | Lisp kernel | Native pointer | Native copy |
|---|---|---:|---:|---:|---:|
| f32 | 1 × 32 | 0.35 | 1.60 | 2.61 | 2.79 |
| f32 | 32 × 32 | 11.08 | 34.07 | 77.52 | 70.87 |
| f32 | 32 × 512 | 159.75 | 359.62 | 135.86 | 175.42 |
| f64 | 1 × 32 | 0.34 | 1.60 | 2.79 | 2.97 |
| f64 | 32 × 32 | 10.78 | 33.92 | 82.84 | 77.35 |
| f64 | 32 × 512 | 156.06 | 355.13 | 180.80 | 220.79 |

Every multi-pass row sets up its own calls. Larger vectors amortize setup;
copy access adds transfers. The benchmark's Lisp allocation counters include
call setup and private scalar bindings, but exclude C constant, spill, and copy
buffers. Counters have implementation-dependent granularity and are approximate.

## x86-64 SBCL fallback

SBCL 2.6.8 under Rosetta on the same M1, using the x86-64 native build. Times
are microseconds per complete 4,096-element stage. Transcendental passes on
`:sbcl` use the typed scalar path; these timings do not imply packed math.

| Precision | Stage | Typed Lisp | Lisp kernel | Native kernel | SBCL kernel |
|---|---|---:|---:|---:|---:|
| f32 | Softmax | 42.41 | 149.37 | 36.71 | 147.22 |
| f32 | SiLU | 36.47 | 45.69 | 16.97 | 45.75 |
| f32 | RoPE: both tables | 71.72 | 85.61 | 45.46 | 85.65 |
| f64 | Softmax | 38.42 | 128.28 | 43.00 | 133.03 |
| f64 | SiLU | 33.04 | 41.16 | 20.45 | 41.70 |
| f64 | RoPE: both tables | 63.84 | 79.13 | 48.32 | 79.14 |

## Reproduce

```sh
sbcl --dynamic-space-size 4096 --script tests/kernel-stages-bench.lisp
```

The runner checks outputs against typed reference loops before timing. It covers
f32/f64 lengths 32, 128, 512, 4,096, and 32,768, plus 32-row softmax batches.
The checkout's source registry takes precedence over Quicklisp local projects.

- [ARM64 raw results](benchmark-runs/2026-10-05-kernel-stages-native.txt)
- [x86-64 Rosetta raw results](benchmark-runs/2026-10-05-kernel-stages-sbcl-rosetta.txt)
- [Runnable stage examples](../examples/inference-stages.lisp)
- [Numerical contract](kernel-transcendentals.md)

## Limitations

- Rosetta measurements characterize translated x86-64 execution on the M1;
  they do not predict performance on physical x86-64 hardware.
- Transcendental operators use scalar system math. SBCL passes containing them
  use typed scalar loops; native arithmetic around them retains VM SIMD paths.
- Pass setup, per-row native calls, quadratic structural deduplication, and
  vector-intermediate storage evaluation are tracked in
  [#137](https://todo.sr.ht/~takeiteasy/trivial-simd/137).
- Exceptional-value consistency remains tracked in
  [#53](https://todo.sr.ht/~takeiteasy/trivial-simd/53).

[^method]: `benchmark-time` calibrates each batch to at least 50 ms and reports
    the median of three timed batches after warmup. Inputs repeat moderate
    values in [-2, 2]. Buffers are reused. Typed loops compile with speed 3 and
    safety 1. Allocation sampling runs between 100 and 10,000 additional calls,
    depending on measured stage time. RoPE includes writing both tables.
