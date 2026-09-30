(ql:quickload :trivial-simd/blas/convenience)

(let ((x (make-array 4 :element-type 'single-float
                     :initial-contents '(1.0 2.0 3.0 4.0)))
      (y (make-array 4 :element-type 'single-float
                     :initial-contents '(10.0 20.0 30.0 40.0))))
  (trivial-simd/blas:saxpy 2 2.0 x -1 y 2)
  (format t "strided AXPY: ~S~%" y)
  (trivial-simd/blas/convenience:axpy! y 1.0 x)
  (format t "contiguous AXPY: ~S~%" y))
