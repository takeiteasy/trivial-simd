# Q8_0 matvec spike

**Recommendation: use a small direct-block NEON C matvec for inference integration.**
The row-batched mixed-input kernel reaches **3.70–3.72%** of the C loop's
throughput on both measured shapes, taking about **27× longer**. Arrays and
foreign views give similar kernel and C timings. This branch retains the
prototype and measurements; production integration is separate.

## Matvec results

Apple M1, macOS 15.7.5, SBCL 2.6.8, Apple Clang 17.0.0, source commit
`7ba22e5`. Values are median microseconds per complete matvec across five fresh
processes. Every method computes the same quantized weights times an f32 vector.

| Storage | Rows × columns | Kernel | NEON Q8_0 | Native repo sgemv | Accelerate sgemv | Kernel/C throughput |
|---|---:|---:|---:|---:|---:|---:|
| Lisp arrays | 1024 × 1024 | 2722.25 | 100.82 | 87.56 | 35.53 | 3.70% |
| Lisp arrays | 3072 × 1024 | 8159.38 | 303.06 | 297.24 | 198.04 | 3.71% |
| Foreign views | 1024 × 1024 | 2721.72 | 100.96 | 87.38 | 25.75 | 3.71% |
| Foreign views | 3072 × 1024 | 8161.13 | 303.58 | 294.23 | 174.75 | 3.72% |

Effective throughput counts two operations per weight: about 0.77 GFLOP/s for
the kernel and 20.7–20.8 GFLOP/s for NEON Q8_0. It excludes decoding operations.
The f32 methods use a pre-dequantized matrix and therefore require more storage.
Accelerate varies with storage and process; its internal thread count is
unobserved. See [limitations](#limitations).

The [raw results](benchmark-runs/2026-10-05-q8-matvec-sbcl.txt) include all
process medians, min/max ranges, numerical errors, repacking times and build flags.

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
path prepares converted and repeated values in bounded buffers before executing
the float VM. The direct C loop reads packed blocks with NEON and explicit
FMA.[^neon] These paths have different rounding orders.

## Repacking and storage

Repacking extracts bytes and widens f16 scales using `convert!`. Its timed path
validates and fills reused arrays; allocation is excluded. The medians below
combine ten samples per shape, two in each process.[^repack]

| Rows × columns | Repack milliseconds | Packed weights | Repacked weights | f32 weights |
|---|---:|---:|---:|---:|
| 1024 × 1024 | 14.20 | 1.0625 MiB | 1.125 MiB | 4 MiB |
| 3072 × 1024 | 42.44 | 3.1875 MiB | 3.375 MiB | 12 MiB |

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
and three-batch median; five fresh processes rotate the method order.

## Limitations

- This is the isolated [spike](https://todo.sr.ht/~takeiteasy/trivial-simd/106),
  retained on `spike/106-q8-matvec` for the
  [inference project](https://todo.sr.ht/~takeiteasy/trivial-simd/107).
  It exports no supported Q8_0 API and includes no GGUF reader, model, Q4_0
  implementation or production integration.
- Measurements use deterministic synthetic weights, one repeatedly accessed
  matrix and one calling thread on Apple M1. They do not measure model-wide
  streaming, cold caches, quantization quality or multi-threaded inference.
- `VECLIB_MAXIMUM_THREADS=1` requests single-threaded Accelerate execution;
  internal workers and CPU-core placement are not observed.
- The current mixed-input preparation uses scalar loads. Packed-loader work is
  tracked in [the loader follow-up](https://todo.sr.ht/~takeiteasy/trivial-simd/128).
  These measurements do not establish the throughput of an optimized loader.
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
