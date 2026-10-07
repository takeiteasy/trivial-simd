# BLAS Level 3

`trivial-simd/blas` provides matrix-matrix operations on dense
[matrix views](matrix-views.md). GEMM accepts independent signed row and column
strides. Other families require regular row-major or column-major layouts.
All calls return the modified output view. The routines accept simple specialized vectors
of single/double real or complex elements.

| Family | Routines | Operation |
|---|---|---|
| Batched real multiply | `sgemm-batch-strided`, `dgemm-batch-strided` | [Constant-stride batches](blas-batches.md) |
| General multiply | `sgemm`, `dgemm`, `cgemm`, `zgemm` | `C ← α op(A) op(B) + β C` |
| Symmetric multiply | `ssymm`, `dsymm`, `csymm`, `zsymm` | `C ← α A B + β C` or `α B A + β C` |
| Hermitian multiply | `chemm`, `zhemm` | Symmetric form with conjugate reflection |
| Symmetric rank update | `ssyrk`, `dsyrk`, `csyrk`, `zsyrk`; `ssyr2k`, `dsyr2k`, `csyr2k`, `zsyr2k` | Update one triangle of `C` |
| Hermitian rank update | `cherk`, `zherk`; `cher2k`, `zher2k` | Update one triangle of `C`; diagonal is real |
| Triangular multiply/solve | `strmm`, `dtrmm`, `ctrmm`, `ztrmm`; `strsm`, `dtrsm`, `ctrsm`, `ztrsm` | Replace `B` |

Names use `s`, `d`, `c`, or `z` precision prefixes. The strict signatures are:

| Family | Arguments after the routine name |
|---|---|
| `gemm` | `transa transb alpha a b beta c` |
| `symm`, `hemm` | `side uplo alpha a b beta c` |
| `syrk`, `herk` | `uplo transpose alpha a beta c` |
| `syr2k`, `her2k` | `uplo transpose alpha a b beta c` |
| `trmm`, `trsm` | `side uplo transpose diag alpha a b` |

`side` is `:left` or `:right`; `uplo` is `:upper` or `:lower`;
`diag` is `:unit` or `:non-unit`. General multiply and triangular routines
accept `:no-transpose`, `:transpose`, and `:conjugate-transpose`. Hermitian
rank updates accept no transpose or conjugate transpose. Complex symmetric
rank updates accept no transpose or transpose. For `herk`, `alpha` and `beta`
have the matching real type; for `her2k`, `alpha` is complex and `beta` is real.

```lisp
(let* ((a (make-array 4 :element-type 'double-float
                      :initial-contents '(1d0 2d0 3d0 4d0)))
       (b (make-array 4 :element-type 'double-float
                      :initial-contents '(5d0 6d0 7d0 8d0)))
       (c (make-array 4 :element-type 'double-float :initial-element 0d0))
       (av (trivial-simd/blas:make-matrix-view a 2 2))
       (bv (trivial-simd/blas:make-matrix-view b 2 2))
       (cv (trivial-simd/blas:make-matrix-view c 2 2)))
  (trivial-simd/blas:dgemm :no-transpose :no-transpose
                          1d0 av bv 0d0 cv)
  c) ; => #(19.0d0 22.0d0 43.0d0 50.0d0)
```

All arguments are validated before changing output. A distinct input view
cannot overlap the output's addressed elements; disjoint subviews may share
one backing vector. GEMM uses exact arithmetic proofs for output uniqueness
and input/output overlap, including interleaved layouts.[^validation] `trmm` and `trsm` intentionally read and replace `B`,
but `A` and `B` must not overlap. A zero `beta` does not read old output
elements. A zero `alpha` does not read matrix operands when the result is
determined without them. Symmetric and Hermitian rank updates leave the
unselected triangle untouched.

The optional convenience system supplies `gemm! (c alpha a b &key transa
transb beta)` and `trsm! (b alpha a &key side uplo transpose diag)`. `gemm!`
defaults to no transpose and zero `beta`. `trsm!` defaults to left, upper,
no transpose, and nonunit diagonal. Both select precision from the output
view. The [runnable example](../examples/blas.lisp) uses `gemm!`.

## Performance

Real `s` and `d` routines above a size threshold run packed SIMD C kernels
(NEON on ARM64; AVX+FMA on x86-64 CPUs that support it, else SSE2) when the
native library is built; everything else runs typed Lisp loops.[^kernels] On the measured Apple M1 with SBCL, a
256×256 `dgemm` takes 840 µs natively against 17.6 ms in Lisp; see the
[BLAS benchmarks](blas-benchmarks.md).

## Limitations

Exact GEMM validation uses bounded workspace but can require substantial search
time for difficult layouts. Stronger search pruning is tracked in
[#154](https://todo.sr.ht/~takeiteasy/trivial-simd/154).

Complex routines use scalar Lisp loops without cache blocking or SIMD. Native
complex kernels are tracked by [#80](https://todo.sr.ht/~takeiteasy/trivial-simd/80).

[^kernels]: Native `gemm` packs `A` and `B` panels so any layout or transpose
    runs the same micro-kernel; `syrk` and `syr2k` run one `gemm` that skips
    tiles outside the stored triangle, so each operand is packed once per cache
    block. `trsm` walks `TS_BLAS_DEPTH`-row blocks, packing the triangle with
    inverted diagonals and solving each tile in a fused `gemm`+solve
    micro-kernel, then updates the remaining rows with one `gemm` per block.
    `trmm` walks 64-row diagonal blocks (`TS_BLAS_TRIANGLE_BLOCK`) the same way.
    All share one packing workspace allocated per call. The native path needs
    the built library, pointer array access, and `m·n·k` at or above
    `trivial-simd/blas::*native-blas-threshold*` (1,000; 1 on ECL). Binding it
    to `most-positive-fixnum` forces the Lisp kernels. `symm` and `hemm` expand the stored triangle to a dense matrix and
    call the `gemm` kernel. `trmm` and `trsm` operate on a dense row-major
    copy of `B`; right-side calls solve the transposed left-side problem.
    Kernels skip bounds checks after validation, so views must fit
    30-bit dimensions and leading dimensions. GEMM row/column strides fit
    ±1,073,741,823 and accessed storage offsets fit 60 unsigned bits.
    Singleton-axis strides are ignored.

[^validation]: Validation first tries storage-span and sorted-stride proofs. Inconclusive layouts use bounded integer equations over at most six stride variables. Live workspace is independent of element count; exact arithmetic can allocate temporary bignums, and difficult layouts can require substantial search time. Foreign overlap includes partially intersecting element byte ranges.
