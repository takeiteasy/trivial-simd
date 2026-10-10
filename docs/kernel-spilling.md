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
Mask-output and count/any/all kernel calls reuse that allocation across all
blocks and release it after success or an error. Concurrent calls own separate
scratch buffers.

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

## Results

Across five independent SBCL runs, the geometric mean cache result is 0.26%
slower for the spilling 32/1,024-element cases. Individual runs range from
0.32% faster to 0.56% slower. The 10% adoption gate is not met.

Median microseconds per call across the five runs, for depth-10 stress kernels:

| Type | Elements | Mode | Complete allocated | Complete cache | C allocated | C reused |
|---|---:|---|---:|---:|---:|---:|
| single-float | 32 | elementwise | 8.000 | 8.116 | 7.989 | 8.069 |
| single-float | 32 | sum | 7.273 | 7.185 | 6.943 | 6.911 |
| single-float | 1,024 | elementwise | 119.957 | 120.605 | 120.379 | 119.686 |
| single-float | 1,024 | sum | 132.895 | 131.625 | 120.371 | 120.586 |
| double-float | 32 | elementwise | 10.031 | 10.059 | 9.883 | 9.959 |
| double-float | 32 | sum | 10.182 | 10.302 | 10.071 | 10.075 |
| double-float | 1,024 | elementwise | 248.023 | 248.176 | 247.590 | 248.090 |
| double-float | 1,024 | sum | 247.219 | 247.156 | 245.930 | 247.430 |

CCL and ECL also fall below the adoption gate. Their figures come from one
complete profile each; SBCL's gate result uses the five independent trials.

| Lisp | Cache change for spilling calls | Four workers: allocated | Four workers: cache |
|---|---:|---:|---:|
| SBCL 2.6.8 | 0.26% slower | 1.037 µs/call | 1.068 µs/call |
| CCL 1.13 | 3.58% slower | 1.021 µs/call | 1.442 µs/call |
| ECL 26.5.5 | 1.01% slower | 2.314 µs/call | 2.963 µs/call |

Four-worker overlap checks return correct results in both storage modes.
These timings include private buffers when the cache is busy.
See [benchmark results](benchmarks.md) for the terminal-reduction comparison.

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
