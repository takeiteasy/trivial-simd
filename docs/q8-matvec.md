# Q8_0 matvec spike

**Recommendation: use a small direct-block NEON C matvec for inference integration.**
With packed ARM64 loaders, the row-batched mixed-input kernel reaches
**16.56–20.45%** of direct C throughput, taking **4.89–6.04× longer**. This
misses the agreed **80% throughput** target for inference integration. The
loader changes improve kernel latency by **4.01–4.63×** against the fresh
scalar-loader baseline. This branch retains the prototype and measurements;
production integration is separate.

## Matvec results

Apple M1, macOS 15.7.5, SBCL 2.6.8, Apple Clang 17.0.0, source commit
`e6f17de`. Values are median microseconds per complete matvec across five fresh
processes. Every method computes the same quantized weights times an f32 vector.

| Storage | Rows × columns | Kernel | NEON Q8_0 | Native repo sgemv | Accelerate sgemv | Kernel/C throughput |
|---|---:|---:|---:|---:|---:|---:|
| Lisp arrays | 1024 × 1024 | 678.80 | 130.69 | 105.05 | 64.55 | 19.25% |
| Lisp arrays | 3072 × 1024 | 1898.09 | 388.09 | 405.67 | 426.05 | 20.45% |
| Foreign views | 1024 × 1024 | 729.27 | 120.78 | 111.35 | 37.41 | 16.56% |
| Foreign views | 3072 × 1024 | 2023.69 | 368.75 | 377.28 | 379.62 | 18.22% |

Effective throughput counts two operations per weight: 2.88–3.31 GFLOP/s for
the kernel and 16.0–17.4 GFLOP/s for NEON Q8_0. It excludes decoding operations.
The f32 methods use a pre-dequantized matrix and therefore require more storage.
Accelerate varies with storage and process; its internal thread count is
unobserved. See [limitations](#limitations).

The [packed-loader results](benchmark-runs/2026-10-05-q8-matvec-loaders-sbcl.txt)
include process medians, min/max ranges, numerical errors, repacking times and
build flags. Kernel process medians range from 626–793 µs for the smaller shape
and 1,731–2,559 µs for the larger shape. Every comparison remains below the
80% throughput target; system load and core placement are unobserved.

## Loader comparison

Both builds use the same fixtures, bytecode, C comparator, compiler flags and
five-process protocol. The [fresh scalar-loader baseline](benchmark-runs/2026-10-05-q8-matvec-baseline-sbcl.txt)
reaches 3.70–3.73% of C throughput, reproducing the original roughly 27× gap.[^baseline]

| Storage | Rows × columns | Scalar-loader kernel µs | Packed-loader kernel µs | Speedup |
|---|---:|---:|---:|---:|
| Lisp arrays | 1024 × 1024 | 2926.75 | 678.80 | 4.31× |
| Lisp arrays | 3072 × 1024 | 8781.25 | 1898.09 | 4.63× |
| Foreign views | 1024 × 1024 | 2923.34 | 729.27 | 4.01× |
| Foreign views | 3072 × 1024 | 8761.75 | 2023.69 | 4.33× |

The [separate loader profile](kernel-input-performance.md) measures preparation
at 237.56 µs versus 2,372.34 µs for an int8/f32 batch of 1,024 × 1,024 elements.
VM-only arithmetic on expanded float inputs takes 339.59 µs. Packed preparation
helps substantially; generic VM arithmetic and materialized load buffers remain
costs. The component profile uses CPU time and the spike uses wall time, so their
timings are not an additive decomposition.

## Inference integration boundary

The recommendation confines specialised C to quantized numerical operations.
The model and executor remain Lisp, with a dtype-dispatched call such as
`(matvec weights activation output)` selecting the numerical implementation.

| Layer | Recommended responsibility |
|---|---|
| Lisp | GGUF parsing, tokenizer, model layers, attention orchestration, KV cache, scratch management, sampling, generation and thread scheduling |
| Specialised C | Packed Q8_0 matrix-vector arithmetic over validated buffers |
| BLAS | Float matrix-vector and matrix-matrix arithmetic |
| trivial-simd | Elementwise arithmetic, reductions, conversions, copies and foreign-memory views |

Concrete trivial-simd uses include RMSNorm sum-of-squares and scaling, residual
addition, feed-forward gating, and attention dot products and reductions.
Quantized matvec bypasses `define-kernel`; the other vector operations remain
part of the Lisp model implementation. The first demo uses flat arrays; tensor
abstractions are extracted after the demo works.

These measurements establish a Q8_0 matvec boundary. They do not establish a
whole-model Lisp/C runtime split or a source-code percentage. Additional native
operations need separate profiling evidence. See [limitations](#limitations)
for the remaining operator and packaging questions.

## Layout and kernel

A packed Q8_0 block stores one little-endian f16 scale and 32 signed bytes,
occupying 34 bytes.[^layout] Rows contain whole blocks. The kernel uses repacked
int8 weights and f32 scales, with no expanded scale vector:

```lisp
(trivial-simd:define-kernel quantized-dot
    (x (q :type :s8) (scale :repeat 32))
  (trivial-simd:sum (* x (* q scale))))

(quantized-dot x q scales :rows rows :row-length width
               :x-row-stride 0 :q-row-stride width
               :scale-row-stride (/ width 32)
               :destination out)
```

The native kernel makes one foreign call per nonempty batch. Its declared-input
path uses packed ARM64 integer conversion and repeated-scale broadcasts into
bounded buffers before executing the unchanged float VM. The direct C loop
reads packed blocks with NEON and explicit FMA.[^neon] These paths have different rounding orders.

## Repacking and storage

Repacking extracts bytes and widens f16 scales using `convert!`. Its timed path
validates and fills reused arrays; allocation is excluded. The medians below
combine ten samples per shape, two in each process.[^repack]

| Rows × columns | Repack milliseconds | Packed weights | Repacked weights | f32 weights |
|---|---:|---:|---:|---:|
| 1024 × 1024 | 18.22 | 1.0625 MiB | 1.125 MiB | 4 MiB |
| 3072 × 1024 | 52.58 | 3.1875 MiB | 3.375 MiB | 12 MiB |

Repacked storage uses 36 bytes per block, about 5.9% more than the packed
format and 71.9% less than f32 weights. The direct C loop uses the packed
representation without repacking. Matvec timings exclude all repacking and
dequantization.

## Validation and reproduction

The harness checks decoded values against an independent f16 decoder and
double-precision dot reference. It covers zero rows and widths, widths
32/64/256/288, signed-byte extremes, zero and varying scales, cancellation,
output sentinels, malformed payloads, and rejected C widths. Every method
overwrites freshly reset outputs. Native and Lisp kernel paths pass for arrays
and foreign views on SBCL, CCL and ECL.

The per-row error bound is `1e-4 + 1e-4 * sum(abs(weight * x))`.
Across the large cases, maximum absolute error is 2.82e-4 for the kernel,
3.43e-4 for NEON, and 2.73e-3 for the scalar C reference. All errors stay
within the bound. The existing full SBCL/CCL/ECL suites and all five native
CTest checks pass.

Use the [benchmark commands](testing.md#q8_0-matvec-spike). Timing reuses outputs,
includes normal dispatch, pointer setup and per-call scratch costs, and excludes
compilation and first-use setup. Each method uses the existing 50 ms calibration
and three-batch median; five fresh processes rotate the method order. Each
process prints and checks its resolved checkout and native-library path.

## Limitations

- This is the isolated [spike](https://todo.sr.ht/~takeiteasy/trivial-simd/106),
  retained on `spike/128-q8-matvec` for the
  [inference project](https://todo.sr.ht/~takeiteasy/trivial-simd/107).
  It exports no supported Q8_0 API and includes no GGUF reader, model, Q4_0
  implementation or production integration.
- `define-kernel` has no `exp`, `sin` or `cos` operators. Complete softmax,
  SiLU and rotary-position calculations therefore need code beyond the current
  DSL. The recommended demo implements those pieces in typed Lisp and profiles
  them before adding native primitives.
- BLAS packaging remains an integration decision: the inference ticket names
  CLBLAS, while this repository supplies `trivial-simd/blas`.
- Measurements use deterministic synthetic weights, one repeatedly accessed
  matrix and one calling thread on Apple M1. They do not measure model-wide
  streaming, cold caches, quantization quality or multi-threaded inference.
- `VECLIB_MAXIMUM_THREADS=1` requests single-threaded Accelerate execution;
  internal workers and CPU-core placement are not observed.
- [Loader optimisation](https://todo.sr.ht/~takeiteasy/trivial-simd/128) preserves
  bounded preparation buffers and generic float VM execution. Fusing input
  loading with arithmetic is outside this experiment.
- The [checkout-selection fix](https://todo.sr.ht/~takeiteasy/trivial-simd/130)
  makes separate-worktree runs load their own native library. An initial rerun
  that loaded the original checkout is discarded; retained baseline and packed
  runs both use the corrected harness.
- The C prototype and reproducible driver target ARM64 macOS with Accelerate
  and the documented CMake Makefiles build. Only whole 32-element blocks and
  finite scales are covered; direct C entry points assume validated buffer sizes.

[^layout]: [GGML Q8_0 declaration](https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h).
    The fixture matches its f16-scale-plus-int8 layout; no GGML dependency is required.

[^repack]: The spike uses an ordinary scalar Lisp extraction loop. These are
    prototype load-time costs, not an optimized GGUF loader measurement. The
    table counts weight storage only, excluding the activation/output vectors,
    temporary f16 scale array and benchmark reference buffers.

[^neon]: The loop sign-extends int8 values, converts them to f32, multiplies by
    the block scale and uses four independent f32 lane accumulators. The C build
    disables implicit contraction; only the explicit NEON FMA operations fuse.

[^baseline]: The scalar-loader implementation is `759ff09` plus the harness fix
    `e6f17de`, measured in detached checkout `dac2c16`. The original measurement
    remains in [its raw log](benchmark-runs/2026-10-05-q8-matvec-sbcl.txt).
