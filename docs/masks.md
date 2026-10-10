# Masks and selection

`compare!` writes `0` or `1` into a plain `(simple-array (unsigned-byte 8)
(*))` mask. `select!` reads zero as false and any nonzero byte as true. Masks
can be allocated and inspected like other specialized vectors.

| Function | Result |
|---|---|
| `(compare! mask operator left right &key ...)` | Mask from `:eq`, `:ne`, `:lt`, `:le`, `:gt`, or `:ge` |
| `(select! destination mask on-true on-false &key ...)` | Numeric choice at each element |
| `(count mask &key ...)` | Number of true elements |
| `(any mask &key ...)` | True if at least one element is true |
| `(all mask &key ...)` | True if every element is true |

`compare!` accepts one scalar operand; both vector operands must have the same
numeric type. `select!` accepts scalar or vector choices matching the numeric
destination type. The [bulk slice rules](api.md#slices) apply. The per-vector
keywords are `:mask-start`, `:left-start`, `:right-start`, `:true-start`,
`:false-start`, and `:destination-start` where relevant. An empty mask gives
`0` for `count`, false for `any`, and true for `all`.
Complex vectors support `:eq` and `:ne` comparisons and complex selection;
ordered comparisons require real or integer operands.

```lisp
(trivial-simd:compare! mask :gt measurements threshold)
(trivial-simd:select! output mask measurements 0.0)
(trivial-simd:count mask)
```

In `define-kernel`, `=`, `/=`, `<`, `<=`, `>`, and `>=` produce mask expressions.
`trivial-simd:select` chooses between two numeric expressions. A comparison
alone writes a byte mask. An outermost `trivial-simd:count`,
`trivial-simd:any`, or `trivial-simd:all` reduces a mask expression; these
reductions do not nest inside elementwise expressions.
Complex kernel inputs support `=` and `/=` with arithmetic operands, nested
selections, and all three mask reductions. Ordered comparisons require real or
integer inputs. See [complex kernel execution](complex.md#kernels).

```lisp
(trivial-simd:define-kernel larger (a b)
  (trivial-simd:select (> a b) a b))

(trivial-simd:define-kernel how-many-larger (a b)
  (trivial-simd:count (> a b)))
```

The [runnable example](../examples/extended.lisp) combines clamping,
conversion, comparison, and mask counting.

## Execution

Native SSE2/NEON comparisons and selection cover both float precisions and all
eight integer types in bulk calls and the register VM. VM comparisons retain
numeric `0`/`1` lanes; byte-mask output also uses `0`/`1`. Selection copies the
chosen bits, including signed zero and NaN payloads.

SBCL uses in-process SSE2 for supported comparisons and selections on f32/f64
and integer types through 32 bits. Bulk f32 selection uses typed Lisp loops;
f32 selection kernels use packed masks. Native and SBCL byte-mask reductions use
packed counting. Scalar tails retain the same results and empty identities.
A spilling native mask call owns one scratch allocation across its 256-element
blocks; concurrent calls have independent scratch.

See [measurements and coverage](conversion-mask-performance.md).

## Limitations

Deeply repeated native selection trees can underperform Lisp; see the
[measured VM limitation](conversion-mask-performance.md#limitations).

Numeric reductions over real/integer selections use scalar execution on every
backend; native integration is tracked in
[#159](https://todo.sr.ht/~takeiteasy/trivial-simd/159).
SBCL s64/u64 masks and unsupported packed arithmetic also use scalar execution.
Floating-point `any`/`all` kernels retain scalar short-circuit evaluation.
Expressions whose packed
execution could evaluate an erroneous unselected branch or a later operand of
`any`/`all` also retain scalar execution. Further SBCL coverage is tracked in
[#157](https://todo.sr.ht/~takeiteasy/trivial-simd/157).
Mask kernels follow the [kernel overlap rule](kernels.md#overlap); shifted
overlap in bulk mask operations follows the [bulk snapshot rules](api.md#overlap).
Floating-point comparisons follow the [IEEE consistency limitation](kernels.md#limitations).
