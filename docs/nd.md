# N-D bulk operations

N-D operations execute a shape over independently strided operands. Signed element strides support reversed views; zero input strides broadcast a value. Every operation returns its destination.

```lisp
(let ((out (make-array 6 :element-type 'single-float))
      (rows (make-array 2 :element-type 'single-float :initial-contents '(1f0 2f0)))
      (columns (make-array 3 :element-type 'single-float :initial-contents '(10f0 20f0 30f0))))
  (trivial-simd:nd-add! out rows columns '(2 3)
                       :left-strides '(1 0) :right-strides '(0 1))
  out)
;; => #(11.0 21.0 31.0 12.0 22.0 32.0)
```

## Calls

| Call | Layout names |
|---|---|
| `(nd-add! out left right shape &key ...)`, also `nd-subtract!`, `nd-multiply!`, `nd-divide!`, `nd-min!`, `nd-max!` | `destination`, `left`, `right` |
| `(nd-negate! out input shape &key ...)`, also `nd-abs!`, `nd-sqrt!`, `nd-reciprocal!`, `nd-log!`, `nd-tanh!`, `nd-sigmoid!` | `destination`, `input` |
| `(nd-clamp! out input lower upper shape &key ...)` | `destination`, `input`, `lower`, `upper` |
| `(nd-compare! mask operator left right shape &key ...)` | `mask`, `left`, `right` |
| `(nd-select! out mask on-true on-false shape &key ...)` | `destination`, `mask`, `true`, `false` |
| `(nd-convert! out input shape &key ...)` | `destination`, `input` |

Each layout name supplies `:NAME-start` and `:NAME-strides`, for example `:destination-start` and `:destination-strides`. Starts default to zero; omitted strides mean contiguous row-major storage. Shape and stride sequences are lists or vectors with matching ranks.

Scalars match the numeric dtype exactly and have no layout keywords. Unary and bounded operations accept scalar numeric inputs; conversion requires vector storage. Comparisons use the existing six operators and write byte masks; their numeric type comes from a vector input, or from the byte destination when both inputs are scalars. Selection requires byte mask storage; any nonzero byte selects the true operand.

Arithmetic uses identical numeric types. Complex `nd-abs!` writes the corresponding real precision. Integer arithmetic wraps; division truncates toward zero and rejects zero divisors. `nd-convert!` accepts the existing [conversion](conversion.md) rounding and encoding keywords, including raw-bit copies for matching encodings.

`nd-log!`, `nd-tanh!` and `nd-sigmoid!` require f32/f64 computation. They use
[scalar system math](kernel-transcendentals.md) with the same layout and alias
rules. Older native libraries without these operations select Lisp traversal.

## Layouts and aliases

Shape `nil` executes one element. Any zero dimension makes the operation empty. Inputs may reuse storage locations; destinations must map each logical element to a distinct location. Validation checks types, layouts and options before writes, including empty calls.[^bounds]

Overlapping inputs are read as if copied before writes. Exact same-dtype in-place mappings need no snapshot. Broadcast axes remain broadcast in snapshots. Different foreign views are compared by byte address. Arithmetic errors may leave part of the destination updated.

Run [the example](../examples/nd.lisp) for broadcasting, reversed inputs and a transposed destination.

## Execution

| Configuration | Execution |
|---|---|
| Native pointer access, real/integer operations and masks | One C traversal; contiguous inner runs use bulk kernels where available, activations use scalar libm |
| Native real-float and encoded conversion | One C traversal; irregular conversion reads and writes elements directly |
| Complex operations and other conversion pairs | Direct Lisp storage traversal |
| SBCL SIMD backend | Supported contiguous array runs use SBCL SIMD; irregular layouts use Lisp |
| Lisp backend, copied-array mode, or missing N-D native symbol | Direct Lisp execution |

Adjacent compatible axes are coalesced. Ordinary execution allocates layout metadata, rather than element buffers or a gather/scatter pass. Unsafe aliases require snapshots; layout validation can require additional workspace. Small calls include fixed validation and dispatch overhead.[^native]

## Limitations

| Limitation | Ticket |
|---|---|
| Irregular output uniqueness may use O(element count) workspace | [#150](https://todo.sr.ht/~takeiteasy/trivial-simd/150) |
| Mixed Lisp/foreign storage conservatively snapshots inputs | [#151](https://todo.sr.ht/~takeiteasy/trivial-simd/151) |
| Short calls pay fixed layout setup and validation overhead | [#152](https://todo.sr.ht/~takeiteasy/trivial-simd/152) |
| Existing 1-D strided APIs retain their separate staging paths | [#101](https://todo.sr.ht/~takeiteasy/trivial-simd/101) |

[^bounds]: Dimensions, starts, counts and stride arithmetic must fit the host's fixnum range and signed 64-bit native metadata. Byte stride/span arithmetic is checked as well. Empty layouts need no reachable storage, but their metadata still has to fit these ranges.
[^native]: C traversal uses typed scalar loads for irregular strides. Native execution requires direct pointer access; otherwise the API uses Lisp rather than copying arrays into native buffers. The floating-point environment and numerical error behavior follow the corresponding bulk operation.
