(ql:quickload :trivial-simd/blas)

;; A 4x3 single-float weight matrix in foreign memory, as it would be after
;; mapping a weights file, used without copying it into a Lisp array.
(trivial-simd:define-kernel scaled-difference (a b) (* 2 (- a b)))

(let* ((rows 4) (cols 3)
       (pointer (cffi:foreign-alloc :float :count (* rows cols)))
       (weights (trivial-simd:make-vector-view pointer :f32 (* rows cols)))
       (row (trivial-simd:make-vector-view pointer :f32 cols :offset cols))
       (input (make-array cols :element-type 'single-float
                               :initial-contents '(1.0 2.0 3.0)))
       (output (make-array rows :element-type 'single-float :initial-element 0.0))
       (wide (make-array cols :element-type 'double-float)))
  (unwind-protect
       (progn
         (dotimes (i (* rows cols))
           (setf (cffi:mem-aref pointer :float i) (float i)))
         (format t "row 1 . input: ~A~%" (trivial-simd:dot row input))
         (trivial-simd:axpy! input 0.5 row)
         (format t "input + 0.5*row 1: ~S~%" input)
         (trivial-simd:convert! wide row)
         (format t "row 1 as doubles: ~S~%" wide)
         (format t "2*(row 1 - input): ~S~%"
                 (scaled-difference (make-array cols :element-type 'single-float) row input))
         (trivial-simd/blas:sgemv :no-transpose 1.0
                                  (trivial-simd/blas:make-matrix-view weights rows cols)
                                  input 1 0.0 output 1)
         (format t "weights x input: ~S~%" output))
    (cffi:foreign-free pointer)))
