(defpackage #:trivial-simd/blas/convenience
  (:use #:cl)
  (:export #:axpy! #:scal! #:copy! #:swap! #:dot #:norm2
           #:gemv! #:ger! #:trsv! #:gemm! #:trsm!))

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

(defun view-precision (view)
  (let ((type (trivial-simd/blas::matrix-element-type
               (trivial-simd/blas:matrix-view-data view))))
    (cond ((eq type 'single-float) :f32)
          ((eq type 'double-float) :f64)
          ((equal type '(complex single-float)) :c32)
          (t :c64))))

(defun gemv! (y alpha view x &key (transpose :no-transpose) beta)
  (let* ((precision (view-precision view))
         (zero (ecase precision
                 (:f32 0.0f0) (:f64 0.0d0)
                 (:c32 #C(0.0f0 0.0f0)) (:c64 #C(0.0d0 0.0d0)))))
    (funcall (ecase precision
               (:f32 #'trivial-simd/blas:sgemv)
               (:f64 #'trivial-simd/blas:dgemv)
               (:c32 #'trivial-simd/blas:cgemv)
               (:c64 #'trivial-simd/blas:zgemv))
             transpose alpha view x 1 (or beta zero) y 1)))

(defun ger! (view alpha x y &key conjugate)
  (let ((precision (view-precision view)))
    (when (and conjugate (member precision '(:f32 :f64)))
      (error "Conjugated GER requires complex storage"))
    (funcall (ecase precision
               (:f32 #'trivial-simd/blas:sger)
               (:f64 #'trivial-simd/blas:dger)
               (:c32 (if conjugate #'trivial-simd/blas:cgerc
                         #'trivial-simd/blas:cgeru))
               (:c64 (if conjugate #'trivial-simd/blas:zgerc
                         #'trivial-simd/blas:zgeru)))
             alpha x 1 y 1 view)))

(defun trsv! (x view &key (uplo :upper) (transpose :no-transpose)
                           (diag :non-unit))
  (funcall (ecase (view-precision view)
             (:f32 #'trivial-simd/blas:strsv)
             (:f64 #'trivial-simd/blas:dtrsv)
             (:c32 #'trivial-simd/blas:ctrsv)
             (:c64 #'trivial-simd/blas:ztrsv))
           uplo transpose diag view x 1))

(defun gemm! (c alpha a b &key (transa :no-transpose)
                                 (transb :no-transpose) beta)
  (let* ((precision (view-precision c))
         (zero (ecase precision
                 (:f32 0.0f0) (:f64 0.0d0)
                 (:c32 #C(0.0f0 0.0f0)) (:c64 #C(0.0d0 0.0d0)))))
    (funcall (ecase precision
               (:f32 #'trivial-simd/blas:sgemm)
               (:f64 #'trivial-simd/blas:dgemm)
               (:c32 #'trivial-simd/blas:cgemm)
               (:c64 #'trivial-simd/blas:zgemm))
             transa transb alpha a b (or beta zero) c)))

(defun trsm! (b alpha a &key (side :left) (uplo :upper)
                                  (transpose :no-transpose) (diag :non-unit))
  (funcall (ecase (view-precision b)
             (:f32 #'trivial-simd/blas:strsm)
             (:f64 #'trivial-simd/blas:dtrsm)
             (:c32 #'trivial-simd/blas:ctrsm)
             (:c64 #'trivial-simd/blas:ztrsm))
           side uplo transpose diag alpha a b))
