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
