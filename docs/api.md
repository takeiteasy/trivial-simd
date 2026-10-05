# API

`trivial-simd` operates on whole, equal-length, simple vectors of
`single-float`, `double-float`, `(complex single-float)`,
`(complex double-float)`, or signed/unsigned 8/16/32/64-bit integers.
Arithmetic inputs and destination have the same element type. Scalars match
that element type exactly; integer scalars fit its signed or unsigned range.
[Conversion](conversion.md) and [masks](masks.md) use mixed element types.
Integer arithmetic [wraps at the vector width](integers.md). A
[vector view](vector-views.md) of foreign memory is accepted wherever a vector
is.

| Function | Result |
|---|---|
| `(add! destination left right &key ...)` | Elementwise sum in `destination` |
| `(subtract! destination left right &key ...)` | Elementwise `left - right` |
| `(multiply! destination left right &key ...)` | Elementwise product |
| `(divide! destination left right &key ...)` | Elementwise `left / right` |
| `(scale! x a &key ...)` | Replace `x` with `a*x` |
| `(axpy! y a x &key ...)` | Replace `y` with `a*x+y` |
| `(fma! destination x y z &key ...)` | Fused elementwise `x*y+z` for floats |
| [`copy!`, `fill!`, `swap!`](copy.md) | Copy, set or exchange slices of any numeric vector |
| [Unary and bounded operations](elementwise.md) | Negate, absolute value, square root, reciprocal, min, max, clamp |
| [`convert!`](conversion.md) | Convert among numeric vector types and bf16/f16 storage |
| [Masks and selection](masks.md) | Compare, select, count, any, all |
| [`sum`, `dot`, `dotc`, `asum`, `nrm2`](reductions.md) | Scalar sum, dot product, absolute sum, Euclidean norm |
| [`minimum`, `maximum`, `argmin`, `argmax`](reductions.md) | Extreme element or its index |
| [`make-vector-view`](vector-views.md) | A vector of foreign memory |
| `(backend)` | `:sbcl`, `:native`, or `:lisp` |
| [Small arrays](small-arrays.md) | Short float vectors run inline, without a backend call |

The four binary `!` functions accept a scalar on either side, with a vector
on the other side. `fma!` accepts scalars or vectors in any mix, including
three scalars. The destination sets the length and type. These calls do not
allocate broadcast vectors. `scale!` and `axpy!` accept one scalar multiplier.

Every `!` function returns its modified vector. A destination may also be an
input at the same slice position. `sum`, `dot`, and `dotc` return a zero of the vector's
element type for empty vectors. Invalid element types and mismatched lengths
signal an error, including on empty slices.

[Complex vectors](complex.md) support the arithmetic operations with a
defined complex result. Their magnitudes and conversions use matching real
vectors where appropriate.

## Slices

Every function takes `:start` (default 0) and `:end` (default the vector
length) to operate on a range. Per-vector keywords override the start for one
vector while the element count stays `end - start`.

| Keyword | Applies to |
|---|---|
| `:start`, `:end` | Every vector |
| `:destination-start`, `:left-start`, `:right-start` | `add!`, `subtract!`, `multiply!`, `divide!` |
| `:y-start`, `:x-start` | `axpy!` |
| `:destination-start`, `:source-start` | `copy!` |
| `:x-start`, `:y-start` | `swap!` |
| `:destination-start`, `:x-start`, `:y-start`, `:z-start` | `fma!` |
| `:left-start`, `:right-start` | `dot`, `dotc` |
| `:input-start` | `sum`, `asum`, `nrm2`, `minimum`, `maximum`, `argmin`, `argmax` |
| `:destination-start`, `:input-start` | Unary operations and `convert!` |
| `:destination-start`, `:left-start`, `:right-start` | `min!`, `max!` |
| `:destination-start`, `:input-start`, `:lower-start`, `:upper-start` | `clamp!` |
| `:mask-start`, `:left-start`, `:right-start` | `compare!` |
| `:destination-start`, `:mask-start`, `:true-start`, `:false-start` | `select!` |
| `:mask-start` | Mask reductions |

```lisp
(trivial-simd:add! out a b :start 4 :end 12)                 ; same range everywhere
(trivial-simd:add! out a b :start 0 :end 8 :left-start 16)   ; a[16..24) + b[0..8)
(trivial-simd:sum a :start 2 :end 10)
(trivial-simd:subtract! out 10.0 a :end 8)                  ; 10 - a[i]
(trivial-simd:axpy! out 2.0 a :end 8)                       ; out[i] += 2*a[i]
```

Slice bounds and per-vector starts are integers. Elements of `destination`
outside the slice are unchanged. Without `:end`, every participating vector
must have the same length. A slice that exceeds a vector signals an error.
`scale!` applies `:start` and `:end` to `x`; a start keyword for a scalar
operand signals an error.

```lisp
(let* ((a (make-array 3 :element-type 'double-float
                      :initial-contents '(1d0 2d0 3d0)))
       (b (make-array 3 :element-type 'double-float
                      :initial-contents '(2d0 3d0 4d0))))
  (trivial-simd:multiply! a a b)
  a) ; => #(2.0d0 6.0d0 12.0d0)
```

## Strides

`:stride` (default 1) steps through a slice: element *i* of a vector is at
`start + i*stride`, and the count stays `end - start`. Every `...-start`
keyword has a matching `...-stride` keyword that overrides the stride for one
vector (`:input-start` and `:input-stride`, `:left-start` and `:left-stride`).
`scale!` and `fill!` take only `:stride`.

```lisp
(trivial-simd:sum a :end 4 :stride 2)                            ; a[0] + a[2] + a[4] + a[6]
(trivial-simd:sum a :end 4 :input-start 7 :stride -2)            ; a[7] + a[5] + a[3] + a[1]
(trivial-simd:add! out a b :end 4 :left-start 7 :left-stride -1) ; out[i] = a[7-i] + b[i]
(trivial-simd:copy! out a :end 4 :destination-stride 2)          ; out[0], out[2], out[4], out[6]
```

- A stride is a nonzero integer. A negative stride walks down from the start
  element, so give that vector's start.
- The first and last visited elements must lie inside the vector; the last is
  `start + (count-1)*stride`.
- A stride keyword for a scalar operand signals an error.
- A call reads every strided input before it writes, so an output that is also
  an input behaves as if the input were copied first.
- `argmin` and `argmax` return the index into the vector, not the position
  along the stride. `define-kernel` functions take no per-element strides; float numeric kernels
  support [row strides](kernel-rows.md).
- Strided calls run the contiguous kernels on gathered copies.[^gather]

The [kernel API](kernels.md#reduction-kernels) composes arithmetic inside a scalar
reduction without a full-length intermediate vector.

Floating-point reductions may differ slightly by backend because SIMD changes
the addition order.[^order] See [reductions](reductions.md) for empty slices,
ties and optional double accumulation.

`axpy!` uses separate multiplication and addition for floats. `fma!` uses
the same single-rounding result as scalar [`fma`](kernels.md#numerical-behavior).
The [BLAS subsystem](blas.md) provides specification-style counts and increments.

## Limitations

The API does not expose SIMD packs; [kernels](kernels.md) compose operations
instead. The available integer operations and backend paths are described in
[Integer vectors](integers.md).

| Area | Ticket |
|---|---|
| Native strided kernels | [#101](https://todo.sr.ht/~takeiteasy/trivial-simd/101) |
| Shifted overlap between bulk-operation destination and input slices; [kernels](kernels.md#overlap) read inputs first | [#67](https://todo.sr.ht/~takeiteasy/trivial-simd/67) |

[^gather]: Each strided vector costs one temporary and one extra pass per call.
    At 1,024 elements a strided call takes 3x to 7x a contiguous native call.
    See [benchmarks](benchmarks.md#strided-access).
[^order]: SIMD reductions accumulate lanes separately before combining them.
    Compare results with a tolerance when the operation order matters.
