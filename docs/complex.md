# Complex vectors

The core API accepts simple vectors of `(complex single-float)` and
`(complex double-float)` for arithmetic, `scale!`, `axpy!`, `sum`, and `dot`.
`dot` is unconjugated; `dotc` conjugates its first vector. `negate!`,
`reciprocal!`, and `sqrt!` also accept complex vectors. Scalar operands match
the vector's complex precision.

| Operation | Complex result |
|---|---|
| `abs!` | Magnitudes in a real vector of matching precision |
| `compare!` | Equality or inequality mask |
| `select!` | Complex destination |
| `convert!` | Real or integer to complex, complex precision changes, and zero-imaginary complex to real or integer |

Converting a complex value to real or integer requires a zero imaginary part;
otherwise it signals an error. Complex magnitudes use Euclidean absolute
value. BLAS absolute sums use a different definition, described in
[CLBLAS Level 1](https://github.com/takeiteasy/CLBLAS/blob/trunk/docs/level1.md).

```lisp
(let ((x (make-array 2 :element-type '(complex single-float)
                       :initial-contents '(#C(1.0f0 2.0f0) #C(3.0f0 4.0f0))))
      (magnitudes (make-array 2 :element-type 'single-float)))
  (trivial-simd:abs! magnitudes x)
  magnitudes) ; => #(2.236068 5.0)
```

The native backend copies complex vectors to interleaved real/imaginary
buffers for SIMD add, subtract, and multiply on supported CPUs. Other complex
operations use Lisp arithmetic. The core slice rules apply to all complex
operations.

`define-kernel` accepts complex arithmetic and `sum` expressions through the
Lisp backend.

## Limitations

Complex numbers have no ordering for `min!`, `max!`, `clamp!`, or ordered
comparisons. `fma!` retains its real floating-point contract. `define-kernel` mask expressions require real or integer vectors; complex mask
kernels are tracked by [#75](https://todo.sr.ht/~takeiteasy/trivial-simd/75).
The native complex path copies vectors into temporary buffers; reducing that
overhead is tracked by [#74](https://todo.sr.ht/~takeiteasy/trivial-simd/74).
