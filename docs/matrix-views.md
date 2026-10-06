# Matrix storage design

BLAS matrix operands use a simple specialized vector, or a
[vector view](vector-views.md) of foreign memory, plus dimension metadata.
A dense view records its backing vector, row and column counts, independent
row and column strides, and base offset. Regular layouts also expose a leading
dimension. Views share storage; creating a submatrix does not copy
elements. `make-matrix-view` constructs a dense view and `matrix-subview`
selects a rectangular part of it.

```lisp
(let* ((data (make-array 8 :element-type 'single-float))
       (matrix (trivial-simd/blas:make-matrix-view
                data 2 3 :leading-dimension 4))
       (part (trivial-simd/blas:matrix-subview matrix 0 1 2 2)))
  (setf (trivial-simd/blas:matrix-ref part 1 1) 3.0f0)
  (aref data 6)) ; => 3.0
```

| Layout | Element `(row, column)` | Minimum leading dimension |
|---|---|---|
| Row-major | `offset + row*ld + column` | `max(1, columns)` |
| Column-major | `offset + column*ld + row` | `max(1, rows)` |

## Explicit strides

Pass both `:row-stride` and `:column-stride` to `make-matrix-view` to address
`offset + row*row-stride + column*column-stride`. Strides are signed element
counts; zero strides repeat input values. Explicit strides exclude an explicit
`:leading-dimension`. Regular strides derive a row-major or column-major layout;
other strides report `:strided` and a leading dimension of 1.[^strides]

```lisp
(trivial-simd/blas:make-matrix-view
 (make-array 6 :element-type 'single-float :initial-contents '(1f0 2f0 3f0 4f0 5f0 6f0))
 2 3 :offset 5 :row-stride -3 :column-stride -1)
;; Matrix rows are (6 5 4) and (3 2 1).
```

`matrix-view-row-stride` and `matrix-view-column-stride` expose dense strides.
GEMM and [batched GEMM](blas-batches.md) accept arbitrary dense strides. Other
BLAS routines require regular layouts and reject irregular views before writes.

Rows and columns use zero-based indexes. A submatrix preserves its parent's
strides, layout and leading dimension, moves the offset to its first element, and records its own dimensions. Validation checks the full addressed span against
the backing vector, including padding between rows or columns. An empty view
may point one element past the end.

`make-band-matrix-view` constructs a general, symmetric, Hermitian, or
triangular band view. Pass `:kl` and `:ku` for general bands, or `:bandwidth`
for the other kinds. `make-packed-matrix-view` constructs a square packed
triangle. Both constructors accept `:layout` and `:offset`; band views also
accept `:leading-dimension`. The operation's `:upper` or `:lower` flag
selects which triangle is stored. These views share their backing vector.

Native kernels pin or copy a view's backing vector through the same paths as
core numeric vectors, and pass a vector view's own pointer.[^pointer] A native
pointer to a view starts at the element offset. Alignment of the backing vector
does not imply alignment at a nonzero offset. The Lisp loops read a vector view
through its pointer. Matrix views share storage, and overlap checks compare
vector views by address.

The design uses separate real and complex specialized vectors. It does not
assume that a displaced or rank-2 Common Lisp array exposes a contiguous,
pinnable numeric buffer. A caller with such an array copies its data into a
supported simple vector before constructing a BLAS view.

## Limitations

Subviews are available for dense matrices; band and packed views use an explicit
base offset.

[^pointer]: See [native array access](backends.md#array-access) for the tested
    implementations and the copy fallback.

[^strides]: The layout and leading dimension describe regular storage only. Dense indexing and GEMM use the independent strides. Band and packed views retain their existing storage formats.
