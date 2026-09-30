(defpackage #:trivial-simd
  (:use #:cl)
  (:shadow #:count)
  (:export #:add! #:subtract! #:multiply! #:divide! #:scale! #:axpy! #:fma!
           #:negate! #:abs! #:sqrt! #:reciprocal! #:min! #:max! #:clamp!
           #:convert! #:compare! #:select! #:select #:count #:any #:all
           #:sum #:dot #:backend
           #:define-kernel #:fma))
