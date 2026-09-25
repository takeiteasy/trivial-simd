(in-package #:trivial-simd)

(defmacro define-sbcl-binary (name element-type width aref operator scalar)
  `(defun ,name (destination left right)
     (declare (type (simple-array ,element-type (*)) destination left right))
     (let ((i 0)
           (n (length destination)))
       (loop while (<= (+ i ,width) n) do
         (setf (,aref destination i)
               (,operator (,aref left i) (,aref right i)))
         (incf i ,width))
       (loop while (< i n) do
         (setf (aref destination i) (,scalar (aref left i) (aref right i)))
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

(defun %sbcl-sum-f32 (input)
  (declare (type (simple-array single-float (*)) input))
  (let ((i 0) (n (length input)) (result 0.0f0))
    (loop while (<= (+ i 4) n) do
      (multiple-value-bind (a b c d)
          (sb-simd-sse:f32.4-values (sb-simd-sse:f32.4-aref input i))
        (incf result (+ a b c d)))
      (incf i 4))
    (loop while (< i n) do (incf result (aref input i)) (incf i))
    result))

(defun %sbcl-sum-f64 (input)
  (declare (type (simple-array double-float (*)) input))
  (let ((i 0) (n (length input)) (result 0.0d0))
    (loop while (<= (+ i 2) n) do
      (multiple-value-bind (a b)
          (sb-simd-sse2:f64.2-values (sb-simd-sse2:f64.2-aref input i))
        (incf result (+ a b)))
      (incf i 2))
    (loop while (< i n) do (incf result (aref input i)) (incf i))
    result))

(defun %sbcl-dot-f32 (left right)
  (declare (type (simple-array single-float (*)) left right))
  (let ((i 0) (n (length left)) (result 0.0f0))
    (loop while (<= (+ i 4) n) do
      (multiple-value-bind (a b c d)
          (sb-simd-sse:f32.4-values
           (sb-simd-sse:f32.4*
            (sb-simd-sse:f32.4-aref left i)
            (sb-simd-sse:f32.4-aref right i)))
        (incf result (+ a b c d)))
      (incf i 4))
    (loop while (< i n) do
      (incf result (* (aref left i) (aref right i)))
      (incf i))
    result))

(defun %sbcl-dot-f64 (left right)
  (declare (type (simple-array double-float (*)) left right))
  (let ((i 0) (n (length left)) (result 0.0d0))
    (loop while (<= (+ i 2) n) do
      (multiple-value-bind (a b)
          (sb-simd-sse2:f64.2-values
           (sb-simd-sse2:f64.2*
            (sb-simd-sse2:f64.2-aref left i)
            (sb-simd-sse2:f64.2-aref right i)))
        (incf result (+ a b)))
      (incf i 2))
    (loop while (< i n) do
      (incf result (* (aref left i) (aref right i)))
      (incf i))
    result))

(mapc #'compile
      '(%sbcl-add-f32 %sbcl-subtract-f32 %sbcl-multiply-f32 %sbcl-divide-f32
        %sbcl-add-f64 %sbcl-subtract-f64 %sbcl-multiply-f64 %sbcl-divide-f64
        %sbcl-sum-f32 %sbcl-sum-f64 %sbcl-dot-f32 %sbcl-dot-f64))
