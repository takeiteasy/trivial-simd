(defpackage #:trivial-simd/blas/convenience
  (:use #:cl)
  (:export #:axpy!))

(in-package #:trivial-simd/blas/convenience)

(defun axpy! (y alpha x &key (start 0) end)
  "Set the contiguous slice of Y to ALPHA*X+Y and return Y."
  (unless (and (integerp start) (<= 0 start)
               (or (null end) (and (integerp end) (<= start end))))
    (error "Invalid AXPY slice"))
  (unless (or end (= (length x) (length y)))
    (error "Vectors must have equal lengths without :END"))
  (let ((end (or end (length y))))
    (unless (and (<= end (length x)) (<= end (length y)))
      (error "AXPY slice exceeds a vector"))
    (cond ((typep y '(simple-array single-float (*)))
           (trivial-simd/blas:saxpy (- end start) alpha x 1 y 1
                                    :x-offset start :y-offset start))
          ((typep y '(simple-array double-float (*)))
           (trivial-simd/blas:daxpy (- end start) alpha x 1 y 1
                                    :x-offset start :y-offset start))
          (t (error 'type-error :datum y
                    :expected-type '(or (simple-array single-float (*))
                                        (simple-array double-float (*))))))))
