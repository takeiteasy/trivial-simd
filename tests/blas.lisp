(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defun blas-vector (type values)
  (make-array (length values) :element-type type :initial-contents values))

(test blas-axpy-spans
  (dolist (type '(single-float double-float))
    (let* ((x (blas-vector type (mapcar (lambda (n) (coerce n type)) '(1 2 3 4 5 6 7))))
           (alpha (coerce 2 type))
           (routine (if (eq type 'single-float)
                        #'trivial-simd/blas:saxpy #'trivial-simd/blas:daxpy)))
      (dolist (case '((3 1 1 1 2)
                      (3 2 2 0 0)
                      (3 -2 -2 0 0)
                      (3 -1 1 1 2)
                      (0 1 1 7 7)))
        (destructuring-bind (n incx incy x-offset y-offset) case
          (let ((y (blas-vector type (make-list 7 :initial-element (coerce 0 type)))))
            (when (plusp n)
              (setf (aref y y-offset) (coerce 10 type)))
            (let ((before (copy-seq y)))
              (funcall routine n alpha x incx y incy
                       :x-offset x-offset :y-offset y-offset)
              (when (zerop n)
                (is (equalp y before)))
              (unless (zerop n)
                (let ((ix (+ x-offset (if (minusp incx) (* (1- n) (- incx)) 0)))
                      (iy (+ y-offset (if (minusp incy) (* (1- n) (- incy)) 0))))
                  (dotimes (i n)
                    (incf (aref before iy) (* alpha (aref x ix)))
                    (incf ix incx)
                    (incf iy incy))
                  (is (equalp y before)))))))))))

(test blas-axpy-validation
  (let ((x (blas-vector 'single-float '(1.0 2.0 3.0)))
        (y (blas-vector 'single-float '(4.0 5.0 6.0)))
        (double (blas-vector 'double-float '(1d0 2d0 3d0))))
    (signals error (trivial-simd/blas:saxpy -1 1.0 x 1 y 1))
    (signals error (trivial-simd/blas:saxpy 2 1.0 x 0 y 1))
    (signals error (trivial-simd/blas:saxpy 2 1.0 x 1 y 0))
    (signals error (trivial-simd/blas:saxpy 2 1.0 x 2 y 1 :x-offset 1))
    (signals error (trivial-simd/blas:saxpy 0 1.0 x 1 y 1 :y-offset 4))
    (signals type-error (trivial-simd/blas:saxpy 1 1d0 x 1 y 1))
    (signals type-error (trivial-simd/blas:saxpy 1 1.0 double 1 y 1))
    (is (equalp y (trivial-simd/blas:saxpy 3 0.0 x 1 y 1)))))

(test blas-convenience-axpy
  (dolist (type '(single-float double-float))
    (let* ((x (blas-vector type (mapcar (lambda (n) (coerce n type)) '(1 2 3 4))))
           (y (blas-vector type (mapcar (lambda (n) (coerce n type)) '(5 6 7 8))))
           (alpha (coerce 2 type)))
      (is (eq y (trivial-simd/blas/convenience:axpy! y alpha x :start 1 :end 3)))
      (is (equalp y (blas-vector type (mapcar (lambda (n) (coerce n type))
                                                  '(5 10 13 8)))))))
  (signals error
    (trivial-simd/blas/convenience:axpy!
     (blas-vector 'single-float '(1.0)) 1.0
     (blas-vector 'single-float '(1.0 2.0)))))

(defun reference-blas-available-p ()
  (loop for library in '("/System/Library/Frameworks/Accelerate.framework/Accelerate"
                         "libopenblas.so" "libblas.so.3" "libblas.so")
        thereis (handler-case
                    (progn
                      (cffi:load-foreign-library library)
                      (cffi:foreign-symbol-pointer "cblas_saxpy"))
                  (error () nil))))

(defmacro without-float-traps (&body body)
  #+sbcl `(sb-int:with-float-traps-masked (:divide-by-zero :invalid :overflow :inexact)
            ,@body)
  #+ecl `(progn (ext:trap-fpe t nil)
                (unwind-protect (progn ,@body) (ext:trap-fpe t t)))
  #+ccl `(let ((mode (ccl:get-fpu-mode)))
           (ccl:set-fpu-mode :division-by-zero nil :invalid nil :overflow nil)
           (unwind-protect (progn ,@body) (apply #'ccl:set-fpu-mode mode)))
  #-(or sbcl ecl ccl) `(progn ,@body))

(test blas-axpy-reference
  (when (reference-blas-available-p)
    (dolist (type '(single-float double-float))
      (let* ((foreign-type (if (eq type 'single-float) :float :double))
             (routine (if (eq type 'single-float)
                          #'trivial-simd/blas:saxpy #'trivial-simd/blas:daxpy))
             (x (blas-vector type (loop for i from 1 to 9 collect (coerce (/ i 4) type)))))
        (dolist (case '((0 1 1) (5 1 1) (3 2 3) (3 -2 -3) (3 -1 2)))
          (destructuring-bind (n incx incy) case
            (let ((actual (blas-vector type (loop for i from 1 to 9
                                                   collect (coerce (/ i 3) type)))))
              (cffi:with-foreign-objects ((fx foreign-type 9) (fy foreign-type 9))
                (dotimes (i 9)
                  (setf (cffi:mem-aref fx foreign-type i) (aref x i)
                        (cffi:mem-aref fy foreign-type i) (aref actual i)))
                (if (eq type 'single-float)
                    (cffi:foreign-funcall "cblas_saxpy" :int n :float 1.25 :pointer fx
                                          :int incx :pointer fy :int incy :void)
                    (cffi:foreign-funcall "cblas_daxpy" :int n :double 1.25d0 :pointer fx
                                          :int incx :pointer fy :int incy :void))
                (funcall routine n (coerce 1.25 type) x incx actual incy)
                (dotimes (i 9)
                  (is (close-enough-p (aref actual i)
                                      (cffi:mem-aref fy foreign-type i) type)))))))))))
