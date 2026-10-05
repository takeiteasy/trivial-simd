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

```lisp
(trivial-simd:define-kernel larger (a b)
  (trivial-simd:select (> a b) a b))

(trivial-simd:define-kernel how-many-larger (a b)
  (trivial-simd:count (> a b)))
```

The [runnable example](../examples/extended.lisp) combines clamping,
conversion, comparison, and mask counting.

## Limitations

Mask kernels use scalar comparison and selection lanes in the native VM, and
typed scalar loops on SBCL. Packed execution is tracked in
[#72](https://todo.sr.ht/~takeiteasy/trivial-simd/72).
Mask kernels follow the [kernel overlap rule](kernels.md#overlap); shifted
overlap in bulk mask operations follows the [bulk overlap limitation](api.md#limitations).
Floating-point comparisons follow the [IEEE consistency limitation](kernels.md#limitations).
