# BLAS subsystem

`trivial-simd/blas` provides CBLAS-style functions in Common Lisp. Load
`trivial-simd/blas/convenience` for shorter vector calls. The strict system
loads without the convenience system.

| System | Package | Current functions |
|---|---|---|
| `trivial-simd/blas` | `trivial-simd/blas` | `saxpy`, `daxpy` |
| `trivial-simd/blas/convenience` | `trivial-simd/blas/convenience` | `axpy!` |

## AXPY

`(saxpy n alpha x incx y incy &key x-offset y-offset)` and `daxpy` set
`y ← alpha*x + y` for `n` elements and return `y`. `saxpy` requires
single-float vectors and scalar; `daxpy` requires double-float. Vectors are
simple specialized vectors. Offsets default to zero and name the beginning
of each storage span. Increments are nonzero integers. A negative increment
visits the span in reverse order.[^negative] Invalid types, increments, or
out-of-bounds spans signal an error. Zero `n` and zero `alpha` leave `y`
unchanged after validation.

```lisp
(let ((x (make-array 3 :element-type 'single-float
                     :initial-contents '(1.0 2.0 3.0)))
      (y (make-array 3 :element-type 'single-float
                     :initial-contents '(4.0 5.0 6.0))))
  (trivial-simd/blas:saxpy 3 2.0 x 1 y 1)
  y) ; => #(6.0 9.0 12.0)
```

`(trivial-simd/blas/convenience:axpy! y alpha x &key start end)` uses unit
increments and derives the count from a shared slice. Without `:end`, the
vectors have equal lengths. It returns `y`. The core
[`trivial-simd:axpy!`](api.md) remains a general bulk-vector operation.
The [runnable example](../examples/blas.lisp) uses both BLAS layers.

## Routine inventory

The intended BLAS API covers the standard real and complex single/double
routine variants, with counts, increments, layouts, leading dimensions,
operation flags, and packed or banded storage where the routine requires
them.[^inventory] This table groups routine names; precision prefixes and
real/complex applicability follow the reference BLAS inventory.

| Level | Routine families |
|---|---|
| 1 | `rotg`, `rotmg`, `rot`, `rotm`, `swap`, `scal`, `copy`, `axpy`, `dot`, `dotu`, `dotc`, `sdsdot`, `dsdot`, `nrm2`, `asum`, `iamax` |
| 2 | `gemv`, `gbmv`, `symv`, `sbmv`, `spmv`, `hemv`, `hbmv`, `hpmv`, `trmv`, `tbmv`, `tpmv`, `trsv`, `tbsv`, `tpsv`, `ger`, `geru`, `gerc`, `syr`, `spr`, `syr2`, `spr2`, `her`, `hpr`, `her2`, `hpr2` |
| 3 | `gemm`, `symm`, `hemm`, `syrk`, `herk`, `syr2k`, `her2k`, `trmm`, `trsm` |

The [matrix storage design](matrix-views.md) defines the representation
for Level 2 and 3 operands.

## Limitations

Only `saxpy` and `daxpy` are implemented. The remaining routines are tracked
by [Level 1 #69](https://todo.sr.ht/~takeiteasy/trivial-simd/69),
[Level 2 #70](https://todo.sr.ht/~takeiteasy/trivial-simd/70), and
[Level 3 #71](https://todo.sr.ht/~takeiteasy/trivial-simd/71).
Complex variants depend on [#47](https://todo.sr.ht/~takeiteasy/trivial-simd/47).
The core bulk API does not accept increments yet ([#43](https://todo.sr.ht/~takeiteasy/trivial-simd/43));
BLAS AXPY handles nonunit increments with a scalar loop. Shifted overlap
between `x` and `y` can affect results ([#67](https://todo.sr.ht/~takeiteasy/trivial-simd/67)).

[^negative]: The [reference SAXPY routine](https://www.netlib.org/lapack/explore-html/d5/d4b/group__axpy_gabe0745849954ad2106e633fd2ebfc920.html)
    begins a negative-increment traversal at the far end of the storage span.
[^inventory]: [Netlib's BLAS quick reference](https://www.netlib.org/blas/blasqr.pdf)
    lists routine signatures and precision variants.
