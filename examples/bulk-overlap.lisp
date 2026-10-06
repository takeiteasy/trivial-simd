(in-package #:cl-user)

(let ((values (make-array 4 :element-type 'single-float
                           :initial-contents '(1f0 2f0 3f0 4f0))))
  (trivial-simd:add! values values 10f0 :end 3 :destination-start 1)
  (assert (equalp values #(1f0 11f0 12f0 13f0)))
  (trivial-simd:swap! values values :end 2 :stride 2 :y-start 1)
  (assert (equalp values #(11f0 1f0 13f0 12f0)))
  (format t "~&Snapshot add and disjoint interleaved swap: ~S~%" values))
