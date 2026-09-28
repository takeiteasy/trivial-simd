# Kernel spill profiling

Native kernels retain per-call scratch allocation. An explicitly owned cache
does not meet the 10% end-to-end improvement threshold on the measured Apple M1
workloads. VM execution dominates these register-heavy expressions.

## Workloads

The profile uses balanced additions of one input at 32, 1,024, and 65,536
elements, for both float types and both elementwise and scalar-sum kernels.
Arrays and compiled programs are prepared before timing.

| Tree depth | Instructions | Spills | Reloads | Peak slots | Single-float bytes | Double-float bytes |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 7 | 0 | 0 | 0 | 0 | 0 |
| 9 | 513 | 1 | 1 | 1 | 1,024 | 2,048 |
| 10 | 1,029 | 3 | 3 | 2 | 2,048 | 4,096 |

Each nonempty spilling call allocates and releases one scratch buffer. Empty
and non-spilling calls allocate none. Scratch size depends on peak live slots,
not vector length; spills and reloads operate on blocks of up to 256 elements.

## Storage comparison

The benchmark-only cache owns one lazy buffer per float type for each program.
A short lock protects borrowing and returning it; execution holds no lock.
Overlapping calls allocate private temporary buffers when the cached buffer is
busy. The profile releases all retained buffers explicitly after each case.

The adoption gate uses the geometric mean of allocated/cache timings for
32/1,024-element spilling cases across five independent SBCL runs. It requires
at least 10% improvement, with no reproducible regression above 5% outside
measurement noise in other cases. Direct C timings separate allocation costs
from VM execution; complete Lisp timings include borrowing and foreign setup.

## Reproducing

See [kernel profiling commands](testing.md#kernel-profiling). Each profile run
reports complete Lisp calls, direct C execution with allocated/reused storage,
and optional baseline execution of the same bytecode. CCL and ECL use the same
workloads and cache prototype.

## Limitations

- These measurements cover balanced addition stress kernels on Apple M1;
  other expressions and allocators may have different costs. The
  [storage evaluation](https://todo.sr.ht/~takeiteasy/trivial-simd/49) records
  the measured decision.
- Bytecode scratch indexes support at most 65,536 simultaneous slots. See the
  [scratch addressing ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/50).
- Setup dominates short non-spilling reductions. See the
  [call setup ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/59).
