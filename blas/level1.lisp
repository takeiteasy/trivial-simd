(in-package #:trivial-simd/blas)

(defun validate-axpy (n alpha x incx y incy x-offset y-offset type)
  (unless (and (integerp n) (<= 0 n))
    (error "N must be a nonnegative integer"))
  (unless (typep alpha type)
    (error 'type-error :datum alpha :expected-type type))
  (dolist (entry (list (list x incx x-offset) (list y incy y-offset)))
    (destructuring-bind (vector increment offset) entry
      (unless (typep vector `(simple-array ,type (*)))
        (error 'type-error :datum vector :expected-type `(simple-array ,type (*))))
      (unless (and (integerp increment) (not (zerop increment)))
        (error "BLAS increment must be a nonzero integer"))
      (unless (and (integerp offset) (<= 0 offset)
                   (if (zerop n)
                       (<= offset (length vector))
                       (< (+ offset (* (1- n) (abs increment))) (length vector))))
        (error "BLAS vector span exceeds its storage")))))

(defun axpy (n alpha x incx y incy x-offset y-offset type)
  (validate-axpy n alpha x incx y incy x-offset y-offset type)
  (unless (or (zerop n) (zerop alpha))
    (if (and (= incx 1) (= incy 1))
        (trivial-simd:axpy! y alpha x :start 0 :end n
                                  :x-start x-offset :y-start y-offset)
        (let ((ix (+ x-offset (if (minusp incx) (* (1- n) (- incx)) 0)))
              (iy (+ y-offset (if (minusp incy) (* (1- n) (- incy)) 0))))
          (loop repeat n do
            (setf (aref y iy) (+ (* alpha (aref x ix)) (aref y iy)))
            (incf ix incx)
            (incf iy incy)))))
  y)

(defun saxpy (n alpha x incx y incy &key (x-offset 0) (y-offset 0))
  "Set Y to ALPHA*X+Y over N single-float elements with BLAS increments."
  (axpy n alpha x incx y incy x-offset y-offset 'single-float))

(defun daxpy (n alpha x incx y incy &key (x-offset 0) (y-offset 0))
  "Set Y to ALPHA*X+Y over N double-float elements with BLAS increments."
  (axpy n alpha x incx y incy x-offset y-offset 'double-float))
