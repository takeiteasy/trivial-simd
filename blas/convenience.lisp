(defpackage #:trivial-simd/blas/convenience
  (:use #:cl)
  (:export #:axpy! #:scal! #:copy! #:swap! #:dot #:norm2))

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
          ((typep y '(simple-array (complex single-float) (*)))
           (trivial-simd/blas:caxpy (- end start) alpha x 1 y 1
                                    :x-offset start :y-offset start))
          ((typep y '(simple-array (complex double-float) (*)))
           (trivial-simd/blas:zaxpy (- end start) alpha x 1 y 1
                                    :x-offset start :y-offset start))
          (t (error 'type-error :datum y
                    :expected-type '(or (simple-array single-float (*))
                                        (simple-array double-float (*))
                                        (simple-array (complex single-float) (*))
                                        (simple-array (complex double-float) (*))))))))

(defun convenience-range (x y start end)
  (let ((type (trivial-simd::vector-type x)))
    (when y
      (unless (eq type (trivial-simd::vector-type y))
        (error "Vectors must have the same element type"))
      (unless (or end (= (length x) (length y)))
        (error "Vectors must have equal lengths without :END")))
    (let ((end (or end (length x))))
      (unless (and (integerp start) (integerp end) (<= 0 start end)
                   (<= end (length x)) (or (null y) (<= end (length y))))
        (error "Invalid BLAS convenience slice"))
      (values type (- end start)))))

(defun scal! (x alpha &key (start 0) end)
  (multiple-value-bind (type count) (convenience-range x nil start end)
    (funcall (ecase type
               (:f32 #'trivial-simd/blas:sscal)
               (:f64 #'trivial-simd/blas:dscal)
               (:c32 #'trivial-simd/blas:cscal)
               (:c64 #'trivial-simd/blas:zscal))
             count alpha x 1 :x-offset start)))

(defun copy! (y x &key (start 0) end)
  (multiple-value-bind (type count) (convenience-range x y start end)
    (funcall (ecase type
               (:f32 #'trivial-simd/blas:scopy)
               (:f64 #'trivial-simd/blas:dcopy)
               (:c32 #'trivial-simd/blas:ccopy)
               (:c64 #'trivial-simd/blas:zcopy))
             count x 1 y 1 :x-offset start :y-offset start)))

(defun swap! (x y &key (start 0) end)
  (multiple-value-bind (type count) (convenience-range x y start end)
    (funcall (ecase type
               (:f32 #'trivial-simd/blas:sswap)
               (:f64 #'trivial-simd/blas:dswap)
               (:c32 #'trivial-simd/blas:cswap)
               (:c64 #'trivial-simd/blas:zswap))
             count x 1 y 1 :x-offset start :y-offset start)))

(defun dot (x y &key (start 0) end conjugate)
  (multiple-value-bind (type count) (convenience-range x y start end)
    (when (and conjugate (not (member type '(:c32 :c64))))
      (error "Conjugation requires complex vectors"))
    (funcall (ecase type
               (:f32 #'trivial-simd/blas:sdot)
               (:f64 #'trivial-simd/blas:ddot)
               (:c32 (if conjugate #'trivial-simd/blas:cdotc
                         #'trivial-simd/blas:cdotu))
               (:c64 (if conjugate #'trivial-simd/blas:zdotc
                         #'trivial-simd/blas:zdotu)))
             count x 1 y 1 :x-offset start :y-offset start)))

(defun norm2 (x &key (start 0) end)
  (multiple-value-bind (type count) (convenience-range x nil start end)
    (funcall (ecase type
               (:f32 #'trivial-simd/blas:snrm2)
               (:f64 #'trivial-simd/blas:dnrm2)
               (:c32 #'trivial-simd/blas:scnrm2)
               (:c64 #'trivial-simd/blas:dznrm2))
             count x 1 :x-offset start)))
