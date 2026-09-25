(in-package #:trivial-simd)

(defmacro define-sbcl-binary (name element-type width aref operator scalar)
  `(defun ,name (destination left right count
                 destination-offset left-offset right-offset)
     (declare (type (simple-array ,element-type (*)) destination left right)
              (type fixnum count destination-offset left-offset right-offset))
     (let ((i 0))
       (declare (type fixnum i))
       (loop while (<= (+ i ,width) count) do
         (setf (,aref destination (+ destination-offset i))
               (,operator (,aref left (+ left-offset i))
                          (,aref right (+ right-offset i))))
         (incf i ,width))
       (loop while (< i count) do
         (setf (aref destination (+ destination-offset i))
               (,scalar (aref left (+ left-offset i))
                        (aref right (+ right-offset i))))
         (incf i)))
     destination))

(define-sbcl-binary %sbcl-add-f32 single-float 4
  sb-simd-sse:f32.4-aref sb-simd-sse:f32.4+ +)
(define-sbcl-binary %sbcl-subtract-f32 single-float 4
  sb-simd-sse:f32.4-aref sb-simd-sse:f32.4- -)
(define-sbcl-binary %sbcl-multiply-f32 single-float 4
  sb-simd-sse:f32.4-aref sb-simd-sse:f32.4* *)
(define-sbcl-binary %sbcl-divide-f32 single-float 4
  sb-simd-sse:f32.4-aref sb-simd-sse:f32.4/ /)
(define-sbcl-binary %sbcl-add-f64 double-float 2
  sb-simd-sse2:f64.2-aref sb-simd-sse2:f64.2+ +)
(define-sbcl-binary %sbcl-subtract-f64 double-float 2
  sb-simd-sse2:f64.2-aref sb-simd-sse2:f64.2- -)
(define-sbcl-binary %sbcl-multiply-f64 double-float 2
  sb-simd-sse2:f64.2-aref sb-simd-sse2:f64.2* *)
(define-sbcl-binary %sbcl-divide-f64 double-float 2
  sb-simd-sse2:f64.2-aref sb-simd-sse2:f64.2/ /)

(defun %sbcl-sum-f32 (input count offset)
  (declare (type (simple-array single-float (*)) input)
           (type fixnum count offset))
  (let ((i 0) (result 0.0f0))
    (declare (type fixnum i))
    (loop while (<= (+ i 4) count) do
      (multiple-value-bind (a b c d)
          (sb-simd-sse:f32.4-values (sb-simd-sse:f32.4-aref input (+ offset i)))
        (incf result (+ a b c d)))
      (incf i 4))
    (loop while (< i count) do
      (incf result (aref input (+ offset i)))
      (incf i))
    result))

(defun %sbcl-dot-f32 (left right count left-offset right-offset)
  (declare (type (simple-array single-float (*)) left right)
           (type fixnum count left-offset right-offset))
  (let ((i 0) (result 0.0f0))
    (declare (type fixnum i))
    (loop while (<= (+ i 4) count) do
      (multiple-value-bind (a b c d)
          (sb-simd-sse:f32.4-values
           (sb-simd-sse:f32.4*
            (sb-simd-sse:f32.4-aref left (+ left-offset i))
            (sb-simd-sse:f32.4-aref right (+ right-offset i))))
        (incf result (+ a b c d)))
      (incf i 4))
    (loop while (< i count) do
      (incf result (* (aref left (+ left-offset i))
                      (aref right (+ right-offset i))))
      (incf i))
    result))

(defun %sbcl-sum-f64 (input count offset)
  (declare (type (simple-array double-float (*)) input)
           (type fixnum count offset))
  (let ((i 0) (result 0.0d0))
    (declare (type fixnum i))
    (loop while (<= (+ i 2) count) do
      (multiple-value-bind (a b)
          (sb-simd-sse2:f64.2-values (sb-simd-sse2:f64.2-aref input (+ offset i)))
        (incf result (+ a b)))
      (incf i 2))
    (loop while (< i count) do
      (incf result (aref input (+ offset i)))
      (incf i))
    result))

(defun %sbcl-dot-f64 (left right count left-offset right-offset)
  (declare (type (simple-array double-float (*)) left right)
           (type fixnum count left-offset right-offset))
  (let ((i 0) (result 0.0d0))
    (declare (type fixnum i))
    (loop while (<= (+ i 2) count) do
      (multiple-value-bind (a b)
          (sb-simd-sse2:f64.2-values
           (sb-simd-sse2:f64.2*
            (sb-simd-sse2:f64.2-aref left (+ left-offset i))
            (sb-simd-sse2:f64.2-aref right (+ right-offset i))))
        (incf result (+ a b)))
      (incf i 2))
    (loop while (< i count) do
      (incf result (* (aref left (+ left-offset i))
                      (aref right (+ right-offset i))))
      (incf i))
    result))

(mapc #'compile
      '(%sbcl-add-f32 %sbcl-subtract-f32 %sbcl-multiply-f32 %sbcl-divide-f32
        %sbcl-add-f64 %sbcl-subtract-f64 %sbcl-multiply-f64 %sbcl-divide-f64
        %sbcl-sum-f32 %sbcl-sum-f64 %sbcl-dot-f32 %sbcl-dot-f64))
