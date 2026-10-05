# Kernel transcendental math

`define-kernel` accepts unary `exp`, `sin`, and `cos` for single- and double-float
computation. They use scalar system math on every backend.

```lisp
(trivial-simd:define-kernel silu (x) (/ x (+ 1 (exp (- x)))))
(trivial-simd:define-kernel rope-sine (angle) (sin angle))
(trivial-simd:define-kernel rope-cosine (angle) (cos angle))
```

These expressions compose with arithmetic, comparisons, selection,
[numeric reductions](kernel-passes.md), [declared inputs](kernel-inputs.md),
slices, and vector views. Integer and complex computation signal an error.
Integer declared inputs convert to the float computation precision before math.

## Numerical contract

Finite arguments have no restricted approximation domain. `sin` and `cos` use
radians. Results retain the computation precision: f32 produces single-floats,
and f64 produces double-floats. These operations use the implementation's ordinary
math routines, without relaxed polynomial approximations or a fixed ULP bound.
Results need not be bit-identical across backends or Lisp implementations.

Overflow, underflow, NaNs, infinities, floating-point traps, and non-default
rounding modes follow the existing [numerical limitations](#limitations).
An arithmetic error may leave part of the destination updated. Stable softmax
subtracts the maximum before exponentiation; this does not give non-finite
inputs a portable softmax result.

## Backends

| Backend | Execution |
|---|---|
| `:lisp` | Typed scalar Common Lisp math |
| `:sbcl` | Typed scalar Lisp loop for a pass containing these operations |
| `:native` | Scalar C `expf`/`sinf`/`cosf` for f32, `exp`/`sin`/`cos` for f64 inside the VM |
| Native library without these opcodes | Typed Lisp fallback |

Arithmetic-only passes retain their existing packed paths. The native VM keeps
arithmetic around transcendental operations in its existing registers and
bounded blocks; scalar math does not imply a separate foreign call per element.

See [softmax, SiLU, and RoPE examples](../examples/inference-stages.lisp) and
[complete-stage performance](kernel-stage-performance.md).

## Limitations

- Packed transcendental approximations are not provided. Any future approximate
  path needs an explicit domain and error contract.
- System math has no project-wide ULP guarantee. Exceptional values, trap
  behavior, and non-default modes may differ by backend; consistency work is
  tracked in [#53](https://todo.sr.ht/~takeiteasy/trivial-simd/53).
- Stable softmax recomputes exponentials in its output pass. Intermediate-storage
  and pass-setup evaluation is tracked in
  [#137](https://todo.sr.ht/~takeiteasy/trivial-simd/137).
