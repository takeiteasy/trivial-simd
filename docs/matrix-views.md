# Matrix storage design

BLAS matrix operands use a simple specialized vector plus dimension metadata.
A view records its backing vector, row and column counts, layout, base offset,
and leading dimension. Views share storage; creating a submatrix does not copy
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

Rows and columns use zero-based indexes. A submatrix keeps its parent's
layout and leading dimension, moves the offset to its first element, and
records its own dimensions. Validation checks the full addressed span against
the backing vector, including padding between rows or columns. An empty view
may point one element past the end.

`make-band-matrix-view` constructs a general, symmetric, Hermitian, or
triangular band view. Pass `:kl` and `:ku` for general bands, or `:bandwidth`
for the other kinds. `make-packed-matrix-view` constructs a square packed
triangle. Both constructors accept `:layout` and `:offset`; band views also
accept `:leading-dimension`. The operation's `:upper` or `:lower` flag
selects which triangle is stored. These views share their backing vector.

Level 2 and 3 routines use portable Lisp loops. Native kernels that use a view's
backing vector must pin or copy it through the same paths as core numeric
vectors.[^pointer] A native pointer to a view starts at the element offset.
Alignment of the backing vector does not imply alignment at a nonzero offset.

The design uses separate real and complex specialized vectors. It does not
assume that a displaced or rank-2 Common Lisp array exposes a contiguous,
pinnable numeric buffer. A caller with such an array copies its data into a
supported simple vector before constructing a BLAS view.

## Limitations

Subviews are available for dense matrices; band and packed views use an explicit
base offset.

[^pointer]: See [native array access](backends.md#array-access) for the tested
    implementations and the copy fallback.
