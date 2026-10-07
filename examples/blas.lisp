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

(let* ((storage (make-array 6 :element-type 'double-float
                            :initial-contents '(1d0 2d0 3d0 4d0 5d0 6d0)))
       (matrix (trivial-simd/blas:make-matrix-view storage 2 3))
       (x (make-array 3 :element-type 'double-float
                      :initial-contents '(1d0 1d0 1d0)))
       (y (make-array 2 :element-type 'double-float :initial-element 0d0)))
  (trivial-simd/blas/convenience:gemv! y 1d0 matrix x)
  (format t "matrix-vector product: ~S~%" y))

(let* ((a (make-array 4 :element-type 'double-float
                      :initial-contents '(1d0 2d0 3d0 4d0)))
       (b (make-array 4 :element-type 'double-float
                      :initial-contents '(5d0 6d0 7d0 8d0)))
       (c (make-array 4 :element-type 'double-float :initial-element 0d0)))
  (trivial-simd/blas/convenience:gemm!
   (trivial-simd/blas:make-matrix-view c 2 2) 1d0
   (trivial-simd/blas:make-matrix-view a 2 2)
   (trivial-simd/blas:make-matrix-view b 2 2))
  (format t "matrix product: ~S~%" c))

(let* ((a (make-array 12 :element-type 'double-float
                       :initial-contents '(1d0 2d0 3d0 4d0 5d0 6d0 7d0 8d0 9d0 10d0 11d0 12d0)))
       (b (make-array 6 :element-type 'double-float :initial-element 1d0))
       (c (make-array 8 :element-type 'double-float :initial-element 0d0)))
  (trivial-simd/blas:dgemm-batch-strided
   :no-transpose :no-transpose 1d0
   (trivial-simd/blas:make-matrix-view a 2 3 :offset 11 :row-stride -3 :column-stride -1)
   (trivial-simd/blas:make-matrix-view b 3 2) 0d0
   (trivial-simd/blas:make-matrix-view c 2 2)
   :batch-count 2 :a-stride -6 :b-stride 0 :c-stride 4)
  (assert (equalp c #(33d0 33d0 24d0 24d0 15d0 15d0 6d0 6d0)))
  (format t "reversed batched matrix product: ~S~%" c))

(let* ((data (make-array 17 :element-type 'double-float :initial-element -1d0))
       (a (make-array 24 :element-type 'double-float :initial-element 1d0))
       (b (make-array 8 :element-type 'double-float :initial-element 1d0))
       (out (trivial-simd/blas:make-matrix-view data 3 2 :row-stride 2 :column-stride 3)))
  (trivial-simd/blas:dgemm-batch-strided
   :no-transpose :no-transpose 1d0
   (trivial-simd/blas:make-matrix-view a 3 4)
   (trivial-simd/blas:make-matrix-view b 4 2) 0d0 out
   :batch-count 2 :a-stride 12 :b-stride 0 :c-stride 8)
  (assert (= 4d0 (aref data 15)))
  (assert (= -1d0 (aref data 1)))
  (format t "interleaved batched destination: ~S~%" data))
