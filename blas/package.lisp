(defpackage #:trivial-simd/blas
  (:use #:cl)
  (:export #:saxpy #:daxpy #:caxpy #:zaxpy
           #:sswap #:dswap #:cswap #:zswap
           #:scopy #:dcopy #:ccopy #:zcopy
           #:sscal #:dscal #:cscal #:zscal #:csscal #:zdscal
           #:sdot #:ddot #:cdotu #:zdotu #:cdotc #:zdotc #:sdsdot #:dsdot
           #:snrm2 #:dnrm2 #:scnrm2 #:dznrm2
           #:sasum #:dasum #:scasum #:dzasum
           #:isamax #:idamax #:icamax #:izamax
           #:srotg #:drotg #:crotg #:zrotg #:srotmg #:drotmg
           #:srot #:drot #:csrot #:zdrot #:crot #:zrot #:srotm #:drotm))
