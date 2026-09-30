(ql:quickload :trivial-simd)

(let* ((measurements (make-array 5 :element-type 'single-float
                                 :initial-contents '(1.0 2.5 3.0 4.5 5.0)))
       (clipped (make-array 5 :element-type 'single-float))
       (integers (make-array 5 :element-type '(unsigned-byte 8)))
       (mask (make-array 5 :element-type '(unsigned-byte 8))))
  (trivial-simd:clamp! clipped measurements 2.0 4.0)
  (trivial-simd:convert! integers clipped :rounding :nearest-even)
  (trivial-simd:compare! mask :gt measurements 3.0)
  (format t "Clipped: ~S~%Integers: ~S~%Above 3: ~D~%"
          clipped integers (trivial-simd:count mask)))

(trivial-simd:define-kernel larger-value (a b)
  (trivial-simd:select (> a b) a b))
