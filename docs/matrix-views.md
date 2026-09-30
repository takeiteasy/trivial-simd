# Matrix storage design

BLAS matrix operands use a simple specialized vector plus dimension metadata.
A view records its backing vector, row and column counts, layout, base offset,
and leading dimension. Views share storage; creating a submatrix does not copy
elements. No public matrix-view constructor is exposed yet.

| Layout | Element `(row, column)` | Minimum leading dimension |
|---|---|---|
| Row-major | `offset + row*ld + column` | `max(1, columns)` |
| Column-major | `offset + column*ld + row` | `max(1, rows)` |

Rows and columns use zero-based indexes. A submatrix keeps its parent's
layout and leading dimension, moves the offset to its first element, and
records its own dimensions. Validation checks the full addressed span against
the backing vector, including padding between rows or columns. An empty view
may point one element past the end.

Native access pins or copies the backing simple vector through the same
implementation-specific paths as core numeric vectors.[^pointer] A native
pointer to a view starts at the element offset. The backing vector's alignment
does not imply the same alignment for a view with a nonzero offset, so kernels
must check the effective pointer before using aligned loads.

The design uses separate real and complex specialized vectors. It does not
assume that a displaced or rank-2 Common Lisp array exposes a contiguous,
pinnable numeric buffer. A caller with such an array copies its data into a
supported simple vector before constructing a BLAS view.

## Limitations

Matrix views and Level 2–3 routines are not implemented yet. Their implementation
is tracked by [#70](https://todo.sr.ht/~takeiteasy/trivial-simd/70) and
[#71](https://todo.sr.ht/~takeiteasy/trivial-simd/71).

[^pointer]: See [native array access](backends.md#array-access) for the tested
    implementations and the copy fallback.
