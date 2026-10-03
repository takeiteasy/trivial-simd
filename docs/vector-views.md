# Vector views

A vector view is a numeric vector in foreign memory: a pointer, an element type,
and a length. Views work wherever a simple numeric vector does, including bulk
operations, reductions, `convert!`, masks, copies, `define-kernel` functions,
and BLAS vectors and matrix storage. Use them for data that cannot be a Lisp
array, such as multi-gigabyte weights in `mmap`'d or foreign-allocated memory.

```lisp
(let* ((count 1000000)
       (pointer (cffi:foreign-alloc :float :count count :initial-element 0.5))
       (weights (trivial-simd:make-vector-view pointer :f32 count))
       (input (make-array count :element-type 'single-float :initial-element 2.0)))
  (unwind-protect (trivial-simd:dot weights input) ; => 1000000.0
    (cffi:foreign-free pointer)))
```

| Function | Result |
|---|---|
| `(make-vector-view pointer type length &key offset)` | A view of `length` elements starting `offset` elements after `pointer` |
| `(vector-view-p object)` | True for a view |
| `vector-view-pointer`, `vector-view-type`, `vector-view-length` | The view's first element, element type and length |

The element type is one of `:f32`, `:f64`, `:c32`, `:c64`, `:s8`, `:u8`,
`:s16`, `:u16`, `:s32`, `:u32`, `:s64` or `:u64`. Complex elements are stored
as interleaved real and imaginary parts. A view does not copy, own or free its
memory, which must stay valid while the view is used. A view of zero elements
may have a null pointer.

## Behavior

Views and Lisp vectors mix freely in one call, and give the same results as
Lisp vectors holding the same elements. Calls validate views like vectors:
element types, lengths and every slice must match, so a view is never read
beyond its length. Slices, per-vector starts and [strides](api.md#strides)
apply to views. `minimum` and `maximum` return an element read from the view.

Views over the same memory are compared by address, even when they are
different objects. `copy!` between overlapping views behaves as if the source
were read first, as it does for one Lisp vector. A destination may also be an
input at the same position, as for Lisp vectors.

## Backends

The `:native` backend passes a contiguous view's pointer to the C library,
without copying, for every operation that has a native path.[^direct] That
includes `define-kernel` functions and the native BLAS kernels. The SBCL and
Lisp backends, strided views, and operations without a native path read and
write views through Lisp buffers of 4,096 elements, one block at a time.[^block]
Memory use stays bounded whatever the view's length. Lisp BLAS kernels read
matrix and vector views through their pointers element by element.

Floating-point reductions of a view on the buffered path add per-block results,
so like a [backend change](api.md#strides) they may round differently from the
same reduction of a Lisp vector. Integer results are identical.

## Performance

| Operation | `:native` | `:sbcl` | `:lisp` |
|---|---:|---:|---:|
| `dot` | 359 / 359 | 4,625 / 3,563 | 3,969 / 3,031 |
| `axpy!` | 289 / 297 | 4,313 / 3,344 | 4,063 / 3,094 |
| `convert!` to double-float | 547 / 488 | 570 / 547 | 146,800 / 146,800 |
| `(sum (* a b))` kernel | 309 / 340 | 3,094 / 1,969 | 4,125 / 2,969 |

Microseconds with a view operand / with a Lisp vector, for 1,048,576
single-float elements, best of five. One operand (the `dot` and kernel input,
the `axpy!` and `convert!` source) is a view; the others are Lisp vectors.
Measured 2026-10-03 on a 2.8 GHz Intel Xeon cloud VM with SBCL 2.2.9. The
native backend reads the view in place; the buffered backends pay one extra
copy of each element, and the `:lisp` conversion is a per-element generic loop
either way.

## Limitations

A view is not a Lisp sequence: `aref`, `length` and the other sequence
functions do not accept it. Use `copy!` to move elements between a view and a
Lisp vector. The [inline small-array path](small-arrays.md) does not apply to
views.

[^direct]: On the native backend a view is read directly by the bulk
    arithmetic operations, `sum`, `dot`, `asum`, `argmin`, `argmax`, the unary
    and bounded operations, `compare!`, `select!`, mask reductions, `fill!` and
    `swap!` above their native size thresholds, and `convert!` between
    `:f32` and `:f64` or bf16/f16 storage (also on the `:sbcl` backend when the
    native library is loaded), and single-float `nrm2`. Complex operations
    other than BLAS, other conversions, `copy!`, double-float `nrm2` and
    numeric reductions of mask kernels use buffers.
[^block]: A buffered call allocates one buffer per view or strided operand.
    When overlapping views need a copy order that blocks cannot keep, such as
    different strides over shared memory, the whole slice moves through one
    buffer, as for a strided Lisp vector.
