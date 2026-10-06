# BLAS subsystem

`trivial-simd/blas` provides CBLAS-style functions in Common Lisp. Load
`trivial-simd/blas/convenience` for shorter vector calls. The strict system
loads without the convenience system.

| System | Package | Current functions |
|---|---|---|
| `trivial-simd/blas` | `trivial-simd/blas` | Level 1, 2, and 3 real and complex routines |
| `trivial-simd/blas/convenience` | `trivial-simd/blas/convenience` | Short vector and matrix calls |

The [Level 1](blas-level1.md), [Level 2](blas-level2.md), and
[Level 3](blas-level3.md) references list
routines, arguments, returns, and precision variants. The
[benchmarks](blas-benchmarks.md) compare Level 2 and 3 against system CBLAS.
[Complex vectors](complex.md) describes their storage and scalar types.

## AXPY

`(saxpy n alpha x incx y incy &key x-offset y-offset)` and its `daxpy`,
`caxpy`, and `zaxpy` variants set `y ← alpha*x + y` for `n` elements and
return `y`. Scalars and vectors match the named precision. Vectors are
simple specialized vectors or [vector views](vector-views.md). Offsets default to zero and name the beginning
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
vectors have equal lengths. It supports real and complex vectors and returns
`y`. The core
[`trivial-simd:axpy!`](api.md) remains a general bulk-vector operation.
The [runnable example](../examples/blas.lisp) uses both BLAS layers.

## Routine inventory

The BLAS API covers the standard real and complex single/double
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

Level 1 routines pass nonunit increments to the core [strides](api.md#strides),
which gather into temporaries ([#101](https://todo.sr.ht/~takeiteasy/trivial-simd/101)). Shifted overlap
between `x` and `y` can affect results ([#143](https://todo.sr.ht/~takeiteasy/trivial-simd/143)).

[^negative]: The [reference SAXPY routine](https://www.netlib.org/lapack/explore-html/d5/d4b/group__axpy_gabe0745849954ad2106e633fd2ebfc920.html)
    begins a negative-increment traversal at the far end of the storage span.
[^inventory]: [Netlib's BLAS quick reference](https://www.netlib.org/blas/blasqr.pdf)
    lists routine signatures and precision variants.
