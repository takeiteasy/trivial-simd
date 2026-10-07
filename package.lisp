(defpackage #:trivial-simd
  (:use #:cl)
  (:shadow #:count)
  (:export #:nd-add! #:nd-subtract! #:nd-multiply! #:nd-divide!
           #:nd-negate! #:nd-abs! #:nd-sqrt! #:nd-reciprocal! #:nd-min! #:nd-max!
           #:nd-clamp! #:nd-compare! #:nd-select! #:nd-convert!
           #:add! #:subtract! #:multiply! #:divide! #:scale! #:axpy! #:fma!
           #:negate! #:abs! #:sqrt! #:reciprocal! #:min! #:max! #:clamp!
           #:copy! #:fill! #:swap!
           #:convert! #:compare! #:select! #:select #:count #:any #:all
           #:sum #:dot #:dotc #:backend #:*inline-small-arrays*
           #:prod #:minimum #:maximum #:argmin #:argmax #:asum #:nrm2
           #:define-kernel #:fma #:sigmoid
           #:nd-log! #:nd-tanh! #:nd-sigmoid!
           #:vector-view #:make-vector-view #:vector-view-p #:vector-view-pointer
           #:vector-view-type #:vector-view-length))
