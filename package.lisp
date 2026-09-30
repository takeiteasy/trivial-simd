(defpackage #:trivial-simd
  (:use #:cl)
  (:export #:add! #:subtract! #:multiply! #:divide! #:scale! #:axpy! #:fma!
           #:sum #:dot #:backend
           #:define-kernel #:fma))
