(ql:quickload :trivial-simd)

(trivial-simd:define-kernel matching-values (a b)
  (trivial-simd:select (= a b) a 0))

(trivial-simd:define-kernel matching-count (a b)
  (trivial-simd:count (= a b)))

(trivial-simd:define-kernel selected-norm (a b)
  (trivial-simd:nrm2 (trivial-simd:select (/= a b) a b)))

(let* ((a (make-array 3 :element-type '(complex single-float)
                       :initial-contents '(#c(3f0 4f0) #c(1f0 2f0) #c(5f0 0f0))))
       (b (make-array 3 :element-type '(complex single-float)
                       :initial-contents '(#c(3f0 4f0) #c(1f0 3f0) #c(5f0 0f0))))
       (out (copy-seq a)))
  (matching-values out a b)
  (assert (= 2 (matching-count a b)))
  (assert (equalp out #(#c(3f0 4f0) #c(0f0 0f0) #c(5f0 0f0))))
  (format t "Matching: ~S~%Count: ~D~%Selected norm: ~F~%"
          out (matching-count a b) (selected-norm a b)))
