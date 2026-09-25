(ql:quickload :trivial-simd)

(let ((left (make-array 4 :element-type 'single-float
                        :initial-contents '(1.0 2.0 3.0 4.0)))
      (right (make-array 4 :element-type 'single-float
                         :initial-contents '(2.0 2.0 2.0 2.0)))
      (result (make-array 4 :element-type 'single-float)))
  (trivial-simd:add! result left right)
  (format t "backend: ~A, add: ~S, dot: ~A~%"
          (trivial-simd:backend) result (trivial-simd:dot left right)))
