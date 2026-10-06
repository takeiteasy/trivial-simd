(defpackage #:trivial-simd/blas
  (:use #:cl)
  (:shadow #:aref)
  (:export #:saxpy #:daxpy #:caxpy #:zaxpy
           #:sswap #:dswap #:cswap #:zswap
           #:scopy #:dcopy #:ccopy #:zcopy
           #:sscal #:dscal #:cscal #:zscal #:csscal #:zdscal
           #:sdot #:ddot #:cdotu #:zdotu #:cdotc #:zdotc #:sdsdot #:dsdot
           #:snrm2 #:dnrm2 #:scnrm2 #:dznrm2
           #:sasum #:dasum #:scasum #:dzasum
           #:isamax #:idamax #:icamax #:izamax
           #:srotg #:drotg #:crotg #:zrotg #:srotmg #:drotmg
           #:srot #:drot #:csrot #:zdrot #:crot #:zrot #:srotm #:drotm
           #:matrix-view #:matrix-view-p #:matrix-view-data #:matrix-view-rows
           #:matrix-view-cols #:matrix-view-layout #:matrix-view-leading-dimension
           #:matrix-view-row-stride #:matrix-view-column-stride
           #:matrix-view-offset #:matrix-view-kind #:matrix-view-kl #:matrix-view-ku
           #:make-matrix-view #:matrix-subview #:matrix-ref
           #:make-band-matrix-view #:make-packed-matrix-view
           #:sgemv #:dgemv #:cgemv #:zgemv #:sgbmv #:dgbmv #:cgbmv #:zgbmv
           #:ssymv #:dsymv #:ssbmv #:dsbmv
           #:sspmv #:dspmv
           #:chemv #:zhemv #:chbmv #:zhbmv #:chpmv #:zhpmv
           #:strmv #:dtrmv #:ctrmv #:ztrmv #:stbmv #:dtbmv #:ctbmv #:ztbmv
           #:stpmv #:dtpmv #:ctpmv #:ztpmv
           #:strsv #:dtrsv #:ctrsv #:ztrsv #:stbsv #:dtbsv #:ctbsv #:ztbsv
           #:stpsv #:dtpsv #:ctpsv #:ztpsv
           #:sger #:dger #:cgeru #:zgeru #:cgerc #:zgerc
           #:ssyr #:dsyr #:sspr #:dspr
           #:ssyr2 #:dsyr2 #:sspr2 #:dspr2
           #:cher #:zher #:chpr #:zhpr #:cher2 #:zher2 #:chpr2 #:zhpr2
           #:sgemm #:dgemm #:cgemm #:zgemm
           #:sgemm-batch-strided #:dgemm-batch-strided
           #:ssymm #:dsymm #:csymm #:zsymm #:chemm #:zhemm
           #:ssyrk #:dsyrk #:csyrk #:zsyrk #:cherk #:zherk
           #:ssyr2k #:dsyr2k #:csyr2k #:zsyr2k #:cher2k #:zher2k
           #:strmm #:dtrmm #:ctrmm #:ztrmm
           #:strsm #:dtrsm #:ctrsm #:ztrsm))
