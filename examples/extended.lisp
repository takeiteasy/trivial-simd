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

(let ((floats (make-array 2 :element-type 'single-float
                           :initial-contents '(1.0001 -1.0001)))
      (halves (make-array 2 :element-type '(unsigned-byte 16))))
  (dolist (rounding '(:nearest-even :truncate :floor :ceiling))
    (trivial-simd:convert! halves floats :destination-encoding :f16 :rounding rounding)
    (format t "~S f16 bits: ~{~4,'0X~^ ~}~%" rounding (coerce halves 'list))))

(let ((doubles (make-array 3 :element-type 'double-float
                            :initial-contents '(1.0d0 1.0004882812500002d0 -2.0d0)))
      (halves (make-array 3 :element-type '(unsigned-byte 16)))
      (bfloat (make-array 3 :element-type '(unsigned-byte 16))))
  (trivial-simd:convert! halves doubles :destination-encoding :f16)
  (trivial-simd:convert! bfloat halves :input-encoding :f16 :destination-encoding :bf16)
  (trivial-simd:convert! doubles halves :input-encoding :f16)
  (format t "Direct f64/f16: ~S~%bf16 bits: ~{~4,'0X~^ ~}~%"
          doubles (coerce bfloat 'list)))
