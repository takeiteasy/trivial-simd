(in-package #:trivial-simd/blas)

(defun validate-span (n vector increment offset type)
  (unless (and (integerp n) (<= 0 n))
    (error "N must be a nonnegative integer"))
  (unless (typep vector `(simple-array ,type (*)))
    (error 'type-error :datum vector :expected-type `(simple-array ,type (*))))
  (unless (and (integerp increment) (not (zerop increment)))
    (error "BLAS increment must be a nonzero integer"))
  (unless (and (integerp offset) (<= 0 offset)
               (if (zerop n)
                   (<= offset (length vector))
                   (< (+ offset (* (1- n) (abs increment))) (length vector))))
    (error "BLAS vector span exceeds its storage")))

(defun span-index (n increment offset)
  (+ offset (if (minusp increment) (* (1- n) (- increment)) 0)))

(defun span-start (n increment offset)
  "First visited index of a span, or OFFSET when the span is empty."
  (if (zerop n) offset (span-index n increment offset)))

(defun validate-axpy (n alpha x incx y incy x-offset y-offset type)
  (unless (typep alpha type)
    (error 'type-error :datum alpha :expected-type type))
  (validate-span n x incx x-offset type)
  (validate-span n y incy y-offset type))

(defun axpy (n alpha x incx y incy x-offset y-offset type)
  (validate-axpy n alpha x incx y incy x-offset y-offset type)
  (unless (or (zerop n) (zerop alpha))
    (trivial-simd:axpy! y alpha x :start 0 :end n
                                  :x-start (span-index n incx x-offset) :x-stride incx
                                  :y-start (span-index n incy y-offset) :y-stride incy))
  y)

(defun saxpy (n alpha x incx y incy &key (x-offset 0) (y-offset 0))
  "Set Y to ALPHA*X+Y over N single-float elements with BLAS increments."
  (axpy n alpha x incx y incy x-offset y-offset 'single-float))

(defun daxpy (n alpha x incx y incy &key (x-offset 0) (y-offset 0))
  "Set Y to ALPHA*X+Y over N double-float elements with BLAS increments."
  (axpy n alpha x incx y incy x-offset y-offset 'double-float))
