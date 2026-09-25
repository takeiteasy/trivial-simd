# API

`trivial-simd` operates on whole, equal-length, simple vectors of
`single-float` or `double-float`. Every vector in an operation has the same
element type.

| Function | Result |
|---|---|
| `(add! destination left right)` | Elementwise sum in `destination` |
| `(subtract! destination left right)` | Elementwise `left - right` |
| `(multiply! destination left right)` | Elementwise product |
| `(divide! destination left right)` | Elementwise `left / right` |
| `(sum input)` | Scalar sum |
| `(dot left right)` | Scalar dot product |
| `(backend)` | `:sbcl`, `:native`, or `:lisp` |

The four `!` functions return `destination`. It may be the same vector as an
input. `sum` and `dot` return a zero of the vector's float type for empty
vectors. Invalid element types and mismatched lengths signal an error.

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

The API does not expose composable SIMD packs or array slices. The
[composable pack ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/1)
tracks the next API design step.

[^order]: SIMD reductions accumulate lanes separately before combining them.
    Compare results with a tolerance when the operation order matters.
