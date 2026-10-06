# Batched GEMM

`sgemm-batch-strided` and `dgemm-batch-strided` multiply a constant-stride batch
of matrix views and return the destination view. Each product computes
`C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]`.

```lisp
(trivial-simd/blas:dgemm-batch-strided
 :no-transpose :no-transpose 1d0 a b 0d0 c
 :batch-count 10 :a-stride 6 :b-stride 0 :c-stride 4)
```

Here `a` is a 2×3 view, `b` is a 3×2 view, and `c` is a 2×2 view.
Backing storage holds ten matrices for `a` and `c`; `b` repeats for every product.
See the [runnable example](../examples/blas.lisp).

## Arguments

The seven positional arguments match GEMM: `transa transb alpha a b beta c`.
All four keywords are required.

| Keyword | Meaning |
|---|---|
| `:batch-count` | Number of products, from 0 through 1,073,741,823 |
| `:a-stride`, `:b-stride` | Signed element distance between input matrix origins; zero broadcasts |
| `:c-stride` | Signed element distance between destination matrix origins |

The first product starts at each view's offset. Product `i` starts at
`offset + i*batch-stride`; negative strides visit earlier origins.
Transpose flags apply within each matrix. Inputs and output have matching
named real precision and supported Lisp or foreign storage.

## Validation and execution

Shapes, types, bounds and output uniqueness are checked across the complete
batch before any writes. Input/output storage spans must be disjoint. Repeated
input elements work; repeated output elements signal an error. A zero batch
count performs no writes after argument validation. Zero contraction or alpha
scales output by beta; zero beta does not read old output values.[^limits]

The native path pins storage once per call and reuses one packing workspace
across products. Its threshold uses total batch work. Portable Lisp execution
loops over products; an older native library lacking batch symbols uses the
existing per-product dispatch.[^native]

## Limitations

Conservative span checks reject disjoint interleaved input/output storage.
The sorted-stride output proof also rejects some valid interleaved destinations.
Stronger bounded validation is tracked in [#149](https://todo.sr.ht/~takeiteasy/trivial-simd/149).

[^limits]: Matrix dimensions and row/column strides follow [GEMM limits](blas-level3.md). Batch strides fit ±(2⁶⁰−1); every accessed offset fits 60 unsigned bits and lies inside backing storage. Stride/count arithmetic is checked before native calls.
[^native]: Native batching requires both batch symbols, the native GEMM backend and pointer array access. Allocation failure precedes native writes. Computation errors can leave output partially updated.
