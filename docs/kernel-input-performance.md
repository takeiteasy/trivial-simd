# Declared-input performance

Packed ARM64 loaders reduce preparation cost by about **10×** for int8/f32
inputs with f32 scales repeated over 32 elements. A complete Q8-like native sum
batch takes **571 µs**, versus **2,720 µs** for the scalar-loader baseline.
Float VM arithmetic remains a substantial part of the call.

## Native preparation and arithmetic

Apple M1, Apple Clang 17.0.0, one calling thread. Each case processes
1,024 rows of 1,024 elements, reusing the activation vector and using separate
integer and repeated-scale inputs. Values are median microseconds across five
fresh processes.[^method]

| Integer / arithmetic | Preparation baseline | Preparation packed | Complete baseline | Complete packed | VM only, packed build |
|---|---:|---:|---:|---:|---:|
| s8 / f32 | 2,372.34 | 237.56 | 2,720.16 | 571.16 | 339.59 |
| u16 / f32 | 3,094.81 | 238.60 | 3,355.88 | 565.98 | 339.51 |
| s32 / f64 | 2,393.62 | 379.50 | 3,093.38 | 1,084.50 | 695.77 |
| s64 / f64 | 2,065.22 | 457.73 | 2,801.66 | 1,150.06 | 700.11 |
| u64 / f64 | 2,494.09 | 475.37 | 3,102.78 | 1,180.05 | 696.66 |

Baseline and packed builds execute identical float bytecode and produce
bit-identical complete and VM-only outputs. The operation is `sum(x * (q * scale))`.
The [raw measurements](benchmark-runs/2026-10-05-kernel-inputs-native.txt) also
include 64-row batches of width 257 with scale phase 5, exposing tails and
partially accessed repetition blocks. Complete-call speedups are 2.44–5.93×
for the large cases and are recorded independently from preparation speedups.

Use the [profiling commands](testing.md#declared-input-profiling).
The [input documentation](kernel-inputs.md#execution) describes loader coverage.

## SBCL x86-64 evaluation

Under Rosetta on the same M1, a benchmark-only prototype converts integer lanes
with scalar Lisp loads and feeds packed SSE arithmetic. It is faster than the
typed Lisp declaration path but slower than the native loader path. Median
microseconds across five fresh SBCL 2.6.8 processes:

| Rows × width | Typed Lisp declarations | Packed-arithmetic prototype | Native descriptors |
|---|---:|---:|---:|
| 1,024 × 1,024 | 2,014.81 | 1,033.33 | 689.23 |
| 3,072 × 1,024 | 6,010.06 | 3,101.22 | 2,057.22 |

The [raw evaluation](benchmark-runs/2026-10-05-kernel-inputs-sbcl-rosetta.txt)
retains method-order rotation, validation and process ranges. The prototype
checks signed-byte extremes, empty dimensions and tails against a double-float
reference. It does not provide packed integer conversion, views, slice phases,
other arithmetic types or general expression compilation.

## Limitations

- These are warm synthetic workloads, not model-wide inference. Only the
  ARM64 measurements establish native hardware performance.
- VM-only timings use fully expanded float inputs, whereas complete calls use
  bounded preparation buffers. The timings identify costs but are not an exact
  additive decomposition or a guaranteed lower bound.
- The packed SBCL prototype is a benchmark, not a supported execution path.
  Compiler integration is tracked in
  [#129](https://todo.sr.ht/~takeiteasy/trivial-simd/129).
- Native x86-64 integer preparation uses typed source loops. Additional packed
  conversion primitives are tracked in
  [#72](https://todo.sr.ht/~takeiteasy/trivial-simd/72).

[^method]: Baseline loader source is `2f82036`; implementation and profiler are
    `dbd9a61`. Each process calibrates to at least 50 ms of CPU time and takes
    the median of three batches. Trial order alternates baseline and packed
    executables. Preparation uses reused private buffers and an indirect call
    per block to retain stores. Complete calls include allocation and descriptor
    validation; VM-only calls use pre-expanded float arrays. Setup and expansion
    stay outside timing. Compiler flags include `-O3 -ffp-contract=off`.
