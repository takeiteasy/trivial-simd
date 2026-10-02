(defpackage #:trivial-simd
  (:use #:cl)
  (:shadow #:count)
  (:export #:add! #:subtract! #:multiply! #:divide! #:scale! #:axpy! #:fma!
           #:negate! #:abs! #:sqrt! #:reciprocal! #:min! #:max! #:clamp!
           #:copy! #:fill! #:swap!
           #:convert! #:compare! #:select! #:select #:count #:any #:all
           #:sum #:dot #:dotc #:backend #:*inline-small-arrays*
           #:minimum #:maximum #:argmin #:argmax #:asum #:nrm2
           #:define-kernel #:fma))
