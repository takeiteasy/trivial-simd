(ql:quickload :trivial-simd)

(let ((left (make-array 4 :element-type 'single-float
                        :initial-contents '(1.0 2.0 3.0 4.0)))
      (right (make-array 4 :element-type 'single-float
                         :initial-contents '(2.0 2.0 2.0 2.0)))
      (result (make-array 4 :element-type 'single-float)))
  (trivial-simd:add! result left right)
  (format t "backend: ~A, add: ~S, dot: ~A~%"
          (trivial-simd:backend) result (trivial-simd:dot left right)))

;; Slices: add the last two elements of LEFT to the first two of RIGHT.
(let ((left (make-array 4 :element-type 'single-float
                        :initial-contents '(1.0 2.0 3.0 4.0)))
      (right (make-array 4 :element-type 'single-float
                         :initial-contents '(10.0 20.0 30.0 40.0)))
      (result (make-array 4 :element-type 'single-float :initial-element 0.0)))
  (trivial-simd:add! result left right :start 0 :end 2 :left-start 2)
  (format t "slice add: ~S~%" result))

(let ((x (make-array 3 :element-type 'single-float
                     :initial-contents '(1.0 2.0 3.0)))
      (y (make-array 3 :element-type 'single-float
                     :initial-contents '(4.0 5.0 6.0))))
  (trivial-simd:scale! x 2.0)
  (trivial-simd:axpy! y 3.0 x)
  (trivial-simd:fma! y x 2.0 y)
  (format t "scale/axpy/fma: ~S ~S~%" x y))

(let ((x (make-array 5 :element-type 'single-float
                     :initial-contents '(3.0 -4.0 1.0 -4.0 2.0))))
  (format t "reductions: min ~A at ~A, max ~A at ~A, asum ~A, nrm2 ~A~%"
          (trivial-simd:minimum x) (trivial-simd:argmin x)
          (trivial-simd:maximum x) (trivial-simd:argmax x)
          (trivial-simd:asum x) (trivial-simd:nrm2 x)))
