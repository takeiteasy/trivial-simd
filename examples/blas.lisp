(ql:quickload :trivial-simd/blas/convenience)

(let ((x (make-array 4 :element-type 'single-float
                     :initial-contents '(1.0 2.0 3.0 4.0)))
      (y (make-array 4 :element-type 'single-float
                     :initial-contents '(10.0 20.0 30.0 40.0))))
  (trivial-simd/blas:saxpy 2 2.0 x -1 y 2)
  (format t "strided AXPY: ~S~%" y)
  (trivial-simd/blas/convenience:axpy! y 1.0 x)
  (format t "contiguous AXPY: ~S~%" y))

(let ((x (make-array 2 :element-type '(complex single-float)
                     :initial-contents '(#C(1.0f0 2.0f0) #C(3.0f0 4.0f0))))
      (y (make-array 2 :element-type '(complex single-float)
                     :initial-contents '(#C(5.0f0 0.0f0) #C(0.0f0 6.0f0)))))
  (format t "conjugated dot: ~S~%" (trivial-simd/blas:cdotc 2 x 1 y 1))
  (trivial-simd/blas/convenience:scal! x #C(2.0f0 0.0f0))
  (format t "scaled complex vector: ~S~%" x))
