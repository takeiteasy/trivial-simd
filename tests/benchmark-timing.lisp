(defvar *benchmark-result* nil)

(defun benchmark-batch (function iterations)
  (let ((result nil)
        (start (get-internal-real-time)))
    (dotimes (i iterations)
      (setf result (funcall function)))
    (let ((elapsed (- (get-internal-real-time) start)))
      (setf *benchmark-result* result)
      elapsed)))

(defun benchmark-time (function)
  (funcall function)
  (let ((iterations 1))
    (loop while (< (benchmark-batch function iterations)
                   (max 1 (ceiling (* 0.05d0 internal-time-units-per-second))))
          do (setf iterations (* 2 iterations)))
    (second
     (sort (loop repeat 3 collect
             (/ (* 1000000.0d0 (benchmark-batch function iterations))
                internal-time-units-per-second iterations))
           #'<))))

