# BLAS Level 3

`trivial-simd/blas` provides matrix-matrix operations on dense
[matrix views](matrix-views.md). Each view supplies its dimensions, row-major or
column-major layout, leading dimension, and backing-vector offset. All calls
return the modified output view. The routines accept simple specialized vectors
of single/double real or complex elements.

| Family | Routines | Operation |
|---|---|---|
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
one backing vector. `trmm` and `trsm` intentionally read and replace `B`,
but `A` and `B` must not overlap. A zero `beta` does not read old output
elements. A zero `alpha` does not read matrix operands when the result is
determined without them. Symmetric and Hermitian rank updates leave the
unselected triangle untouched.

The optional convenience system supplies `gemm! (c alpha a b &key transa
transb beta)` and `trsm! (b alpha a &key side uplo transpose diag)`. `gemm!`
defaults to no transpose and zero `beta`. `trsm!` defaults to left, upper,
no transpose, and nonunit diagonal. Both select precision from the output
view. The [runnable example](../examples/blas.lisp) uses `gemm!`.

## Limitations

Level 3 uses portable scalar loops. Matrix kernel optimization is tracked by
[#77](https://todo.sr.ht/~takeiteasy/trivial-simd/77).
