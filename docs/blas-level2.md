# BLAS Level 2

`trivial-simd/blas` provides matrix-vector operations and rank updates for
simple specialized single/double real and complex vectors. Matrix arguments
are [views](matrix-views.md); vector arguments use the same nonzero increments
and optional `:x-offset` / `:y-offset` as [Level 1](blas-level1.md).
Every routine validates its arguments before changing output.

| Family | Routines | Matrix view |
|---|---|---|
| General matrix-vector | `gemv`, `gbmv` | Dense, general band |
| Symmetric matrix-vector | `symv`, `sbmv`, `spmv` | Dense, symmetric band, packed |
| Hermitian matrix-vector | `hemv`, `hbmv`, `hpmv` | Dense, Hermitian band, packed |
| Triangular multiply/solve | `trmv`, `tbmv`, `tpmv`, `trsv`, `tbsv`, `tpsv` | Dense, triangular band, packed |
| General rank update | `ger`, `geru`, `gerc` | Dense |
| Symmetric rank update | `syr`, `spr`, `syr2`, `spr2` | Dense, packed |
| Hermitian rank update | `her`, `hpr`, `her2`, `hpr2` | Dense, packed |

Names use `s`, `d`, `c`, or `z` precision prefixes. Hermitian routines
use complex precision; symmetric routines use real precision. `her` and
`hpr` take a matching real `alpha`.
The standard operation flags are `:no-transpose`, `:transpose`,
`:conjugate-transpose`, `:upper`, `:lower`, `:unit`, and `:non-unit`.
Matrix dimensions, layout, leading dimension, and bandwidth come from the
view.[^storage]

| Calls | Positional arguments after the precision-prefixed name |
|---|---|
| `gemv`, `gbmv` | `transpose alpha view x incx beta y incy` |
| Symmetric/Hermitian matrix-vector | `uplo alpha view x incx beta y incy` |
| Triangular multiply/solve | `uplo transpose diag view x incx` |
| `ger`, `geru`, `gerc` | `alpha x incx y incy view` |
| Symmetric/Hermitian rank one | `uplo alpha x incx view` |
| Symmetric/Hermitian rank two | `uplo alpha x incx y incy view` |

```lisp
(let* ((a (make-array 6 :element-type 'double-float
                      :initial-contents '(1d0 2d0 3d0 4d0 5d0 6d0)))
       (view (trivial-simd/blas:make-matrix-view a 2 3))
       (x (make-array 3 :element-type 'double-float
                      :initial-contents '(1d0 1d0 1d0)))
       (y (make-array 2 :element-type 'double-float
                      :initial-element 0d0)))
  (trivial-simd/blas:dgemv :no-transpose 1d0 view x 1 0d0 y 1)
  y) ; => #(6.0d0 15.0d0)
```

`gemv`, `gbmv`, and the symmetric/Hermitian matrix-vector routines return
the modified `y`. Triangular multiply and solve return `x`; rank updates
return the matrix view. A zero `alpha` skips the matrix contribution. A zero
`beta` replaces each addressed output element without reading its old value.
Only the selected triangle changes during symmetric or Hermitian updates.

The optional convenience package provides `gemv!`, `ger!`, and `trsv!`
with unit increments and precision chosen from the view. `gemv!` defaults
to no transpose and zero `beta`; `ger!` accepts `:conjugate t` for complex
storage; `trsv!` defaults to upper, no transpose, nonunit diagonal.
The [runnable example](../examples/blas.lisp) shows `gemv!`.

## Performance

Each routine walks the stored matrix one contiguous line at a time, so dense,
band, and packed views share one typed kernel per precision.[^runs] Band
routines visit only the band and triangular routines only their triangle. On
the measured Apple M1 with SBCL, a 64×64 `dgemv` takes 2.5 µs against 271 µs
for the earlier loops; see the [BLAS benchmarks](blas-benchmarks.md).

## Limitations

Level 2 kernels are scalar Lisp loops. Native and SIMD kernels are tracked by
[#79](https://todo.sr.ht/~takeiteasy/trivial-simd/79). Core strided SIMD access
is tracked by [#43](https://todo.sr.ht/~takeiteasy/trivial-simd/43).
Shared input/output
storage follows the [overlap limitation](api.md#limitations).

[^storage]: General band views use `:kind :general` with `:kl` and `:ku`.
    Symmetric, Hermitian, and triangular band views use their matching
    `:kind` and `:bandwidth`. Packed views store one triangle of an `n × n`
    matrix. Banded and packed views share storage but do not expose subviews.

[^runs]: A line is a row for row-major views and a column for column-major
    views. Kernels skip bounds checks after validation, so views and
    increments must fit 30 bits.
