# Kernels

`define-kernel` compiles an elementwise expression or scalar sum into one
backend execution. It is experimental.

```lisp
(trivial-simd:define-kernel multiply-add (a b c) (+ (* a b) c))

(multiply-add out x y z)                        ; out[i] = x[i]*y[i] + z[i]
(multiply-add out x y z :start 0 :end 64)       ; same slice keywords as bulk operations
(multiply-add out x y z :a-start 8 :end 64)     ; per-argument starts: <argument>-start
```

An elementwise kernel takes `destination`, one vector per argument, and the
[slice keywords](api.md#slices), plus one `<argument>-start` keyword per argument.
It returns `destination`.

## Expressions

| Form | Meaning |
|---|---|
| argument name | Element of that vector |
| real number | Constant[^constants] |
| `(+ x y ...)`, `(* x y ...)` | N-ary, folded left |
| `(- x y ...)`, `(/ x y ...)` | N-ary, folded left |
| `(- x)`, `(/ x)` | Negation, reciprocal |
| `(sqrt x)`, `(abs x)` | Square root, absolute value |
| `(min x ...)`, `(max x ...)` | Minimum, maximum; folded left |
| `(trivial-simd:fma a b c)` | `a*b+c` with one rounding step |
| `(> a b)` and other two-operand comparisons | Byte mask expression |
| `(trivial-simd:select mask a b)` | Choose a numeric value per element |
| `(trivial-simd:count mask)`, `(trivial-simd:any mask)`, `(trivial-simd:all mask)` | Top-level mask reductions |
| `(trivial-simd:sum x)` | Top-level scalar sum |

`+`, `*`, `min`, and `max` with one operand return it unchanged. Empty operand
lists and unsupported forms signal an error when the kernel is defined. Vectors
share one numeric element type, chosen per call. Integer expressions use
[wrapping arithmetic](integers.md); `sqrt` and `fma` remain float-only.
Comparisons use `=`, `/=`, `<`, `<=`, `>`, and `>=` with two operands. A
comparison-only kernel writes a [byte mask](masks.md). Mask reductions return
an integer count or a boolean; they are top-level forms like `sum`.

## Sum kernels

A top-level `(trivial-simd:sum expression)` returns a scalar instead of writing
into a destination vector. It takes input vectors, `:start` and `:end`, and
one `<argument>-start` keyword per input. The first input sets the default length.
The same vector-type and [slice checks](api.md#slices) apply.

```lisp
(trivial-simd:define-kernel kernel-dot (a b)
  (trivial-simd:sum (* a b)))

(kernel-dot x y)                         ; scalar dot product
(kernel-dot x y :end 64 :a-start 8)     ; x[8..72) * y[0..64)
```

Sum kernels require at least one input vector, even for a constant expression.
They return a scalar of the input element type and typed zero for an empty slice.
They leave inputs unchanged, use input precision for accumulation, and need no
full-length intermediate result vector. Results may differ slightly by backend
because the addition order differs.[^sum]

## Numerical behavior

`sqrt` signals an error for a negative operand on every backend; negative zero
is valid. An arithmetic error may leave part of `destination` updated.
`min` and `max` retain the left operand on equal values, including signed zeros.

`fma` uses round-to-nearest-even for finite operands with finite results,
including subnormals. Scalar and packed acceleration retain the same result as
its exact portable fallback. `(+ (* a b) c)` performs separate multiplication and
addition. The exported scalar `fma` function takes three floats of the same type.
An exact cancellation returns positive zero; a negative-zero product plus
negative zero returns negative zero.[^fma]
ARM64 Lisp kernels use [in-process scalar FMA](arm64-fma.md) without requiring
the native library.

```lisp
(trivial-simd:define-kernel rounded-multiply-add (a b c)
  (trivial-simd:fma a b c))
```

## Backends

| Backend | Implementation |
|---|---|
| `:lisp` | Typed scalar loop |
| `:sbcl` | `sb-simd` pack loop with a scalar tail[^sbcl] |
| `:native` | Register bytecode run by a C interpreter over 256-element blocks[^vm] |

Elementwise mask kernels and mask reductions use the native register VM on
`:native`. Its comparison and selection opcodes process scalar lanes;
arithmetic opcodes retain their SIMD paths. A `sum` over a selection uses a
typed scalar loop, as do mask expressions on `:sbcl` and `:lisp`.

The native backend makes one foreign call per kernel call and needs no
full-length intermediate arrays. Expressions that exceed eight registers spill
intermediate values into scratch blocks. Scratch slots are reused, and their
storage is allocated per call and freed before returning. Kernels that fit in
eight registers and empty calls allocate no scratch storage.[^scratch]

Native code and constant buffers are allocated lazily and reclaimed when their
kernel function becomes unreachable. Retained references to an older function
remain callable after redefinition. Reclamation follows garbage collection.[^ownership]

On ECL, native calls share a compiled setup helper by input count and
whether the kernel returns a sum. Different expressions, numeric types, and pointer/copy access reuse
the same helper. This includes `eval` definitions.
The first native call for a signature compiles its helper; later definitions
reuse it. Failed compilation selects each kernel's own interpreted fallback
without retrying that signature until restart. Lisp-only calls do not initialize
helpers. Each invocation owns its pointer table and reduction output storage,
so concurrent calls use separate mutable buffers.[^ecl-runners]
See [ECL measurements](kernel-performance.md#ecl-native-calls) and
[cold-start limitations](#limitations).

See the [runnable examples](../examples/kernels.lisp) for a balanced expression
that uses spilling.

## Performance

Kernels avoid full-length intermediate results. Performance depends on the
expression, vector length, and backend. See [kernel benchmarks](kernel-performance.md)
for measurements and [FMA fallback limitations](#limitations).

## Limitations

- Mask expressions use scalar comparison and selection lanes in the native VM
  and scalar loops on SBCL. Spilling mask reductions allocate scratch per
  256-element block. Packed execution and scratch reuse are tracked in
  [#72](https://todo.sr.ht/~takeiteasy/trivial-simd/72).
- NaNs, infinities, non-default rounding modes, and floating-point traps may
  behave differently by backend. See the
  [IEEE consistency ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/53).
- ARM64 scalar FMA uses guarded compiler versions; other versions use optional
  C helpers or exact arithmetic. See [ARM64 FMA limitations](arm64-fma.md#limitations).
- A native program supports at most 65,536 simultaneous scratch slots; exceeding
  this limit signals an error when defining the kernel. See the
  [scratch addressing ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/50).
- Spilling kernels allocate scratch storage on each native call. See the
  [spill storage evaluation](https://todo.sr.ht/~takeiteasy/trivial-simd/49).
- Native call setup dominates short reductions; specialised bulk `dot` may
  be faster for `sum(a*b)`. See the
  [call setup ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/59).
- Reductions are top-level sums only. Nested reductions and additional reducers
  are unsupported. See the
  [additional reductions ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/42).
- ECL's first native call for each setup signature includes helper compilation,
  costing hundreds of milliseconds. Cold compilation for different signatures
  is serialized. See the
  [concurrent compilation ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/65).
- SBCL kernels are generated only on x86-64 with `sb-simd`.

[^constants]: Float constants must be representable as `single-float`.
    Integer constants must be in range for the vector's element type.
[^vm]: Instructions are four bytes: opcode, destination, and two operands. An
    operand names a register (0-7) or an input vector; the last instruction
    writes to `destination` directly for elementwise kernels. Sum kernels
    evaluate the final instruction into lane accumulators, followed by the
    scalar tail, and add each block total to the result.
    FMA uses its register destination as the addend before writing the result.
    Spill/reload instructions encode an opcode,
    a register, and a little-endian 16-bit scratch index.

[^scratch]: Each scratch slot holds 256 elements: 1 KiB for `single-float` or
    2 KiB for `double-float`. Storage depends on peak live spills, not vector
    length. Allocation failure signals a Lisp error before writing the output.
    Rebuild the native library when updating the Lisp implementation.

[^sbcl]: SIMD operations use temporary variables and ordered assignments, keeping
    nested expressions out of operator macro arguments during compilation. Each
    input pack is loaded once per iteration; temporary variables are reused after
    their values are consumed. Scalar loops and SIMD tails also load each input
    element once per iteration and use flat assignments with reused typed
    temporaries to bound generated compiler scopes.

[^fma]: The portable helper aligns decoded integer significands, computes the
    product and sum exactly, then rounds once to the vector precision. Native
    ARM64 uses NEON FMA. Native x86 selects packed FMA when the CPU and OS
    support it, retaining per-lane `fma`/`fmaf` otherwise. SBCL x86 selects
    scalar and packed `sb-simd` FMA with an OS AVX-state guard. ARM64 Lisp uses guarded scalar compiler
    adapters. Other scalar paths use optional native helpers or exact integer arithmetic. Rebuild the
    native library after updating the kernel implementation.

[^sum]: Lisp sums elements in order; SBCL accumulates SIMD lanes before adding
    the scalar tail; native C sums SIMD lanes within each 256-element block,
    then adds block totals. The native reduction makes one foreign call and
    uses bounded register, lane, and spill storage. Allocation and
    square-root domain errors signal Lisp errors without returning a sum.
    Native builds disable implicit multiply/add contraction; explicit FMA keeps
    its single-rounding behavior.

[^ownership]: CCL's native finalization queue and `trivial-garbage` on other Lisps
    release foreign buffers without retaining
    their owning program. Partial initialization frees completed allocations,
    and active native calls keep the owner reachable until they finish.

[^ecl-runners]: The process retains at most 495 helpers or failure markers under
    the current input-count limit. Cache entries do not retain kernel programs
    or per-call data. Each program stores its selected runner, so warmed calls
    avoid shared-cache lookup and locking. Threaded ECL protects initialization
    with one native lock; non-threaded ECL initializes directly.
