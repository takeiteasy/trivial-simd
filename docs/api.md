# API

`trivial-simd` operates on whole, equal-length, simple vectors of
`single-float` or `double-float`. Every vector in an operation has the same
element type.

| Function | Result |
|---|---|
| `(add! destination left right &key ...)` | Elementwise sum in `destination` |
| `(subtract! destination left right &key ...)` | Elementwise `left - right` |
| `(multiply! destination left right &key ...)` | Elementwise product |
| `(divide! destination left right &key ...)` | Elementwise `left / right` |
| `(sum input &key ...)` | Scalar sum |
| `(dot left right &key ...)` | Scalar dot product |
| `(backend)` | `:sbcl`, `:native`, or `:lisp` |

The four `!` functions return `destination`. It may be the same vector as an
input. `sum` and `dot` return a zero of the vector's float type for empty
vectors. Invalid element types and mismatched lengths signal an error.

## Slices

Every function takes `:start` (default 0) and `:end` (default the vector
length) to operate on a range. Per-vector keywords override the start for one
vector while the element count stays `end - start`.

| Keyword | Applies to |
|---|---|
| `:start`, `:end` | Every vector |
| `:destination-start`, `:left-start`, `:right-start` | `add!`, `subtract!`, `multiply!`, `divide!` |
| `:left-start`, `:right-start` | `dot` |
| `:input-start` | `sum` |

```lisp
(trivial-simd:add! out a b :start 4 :end 12)                 ; same range everywhere
(trivial-simd:add! out a b :start 0 :end 8 :left-start 16)   ; a[16..24) + b[0..8)
(trivial-simd:sum a :start 2 :end 10)
```

Elements of `destination` outside the slice are unchanged. Without `:end`,
every vector must have the same length. A slice that exceeds a vector signals
an error.

```lisp
(let* ((a (make-array 3 :element-type 'double-float
                      :initial-contents '(1d0 2d0 3d0)))
       (b (make-array 3 :element-type 'double-float
                      :initial-contents '(2d0 3d0 4d0))))
  (trivial-simd:multiply! a a b)
  a) ; => #(2.0d0 6.0d0 12.0d0)
```

Floating-point reductions may differ slightly by backend because SIMD changes
the addition order.[^order]

## Limitations

The API does not expose composable SIMD packs. The
[composable pack ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/1)
tracks the next API design step.

[^order]: SIMD reductions accumulate lanes separately before combining them.
    Compare results with a tolerance when the operation order matters.
