(ql:quickload :trivial-simd)

(trivial-simd:define-kernel fused-multiply-add (a b c) (+ (* a b) c))

(let ((a (make-array 4 :element-type 'single-float :initial-contents '(1.0 2.0 3.0 4.0)))
      (b (make-array 4 :element-type 'single-float :initial-element 2.0))
      (c (make-array 4 :element-type 'single-float :initial-element 0.5))
      (out (make-array 4 :element-type 'single-float)))
  (fused-multiply-add out a b c)
  (format t "backend: ~A, a*b+c: ~S~%" (trivial-simd:backend) out))
