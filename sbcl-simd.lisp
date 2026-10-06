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

(defmacro define-sbcl-extended-unary (name element width aref pack xor andc1 sqrt divide &optional extract)
  (let ((lanes (loop repeat width collect (gensym "LANE"))))
    `(defun ,name (operation destination input count d-offset i-offset)
     (declare (type (simple-array ,element (*)) destination input)
              (type fixnum count d-offset i-offset))
     ,@(unless extract
         '((when (member operation '(:sqrt :reciprocal))
       (dotimes (i count)
         (let ((value (aref input (+ i-offset i))))
           (when (or (and (eq operation :sqrt) (minusp value))
                     (and (eq operation :reciprocal) (zerop value)))
             (error "Invalid operand for ~A" operation)))))))
     (let ((i 0))
       (declare (type fixnum i))
       (loop while (<= (+ i ,width) count) do
         (let ((value (,aref input (+ i-offset i))))
           ,@(when extract
               `((when (member operation '(:sqrt :reciprocal))
                   (multiple-value-bind (,@lanes) (,extract value)
                     (when (if (eq operation :sqrt)
                               (or ,@(loop for lane in lanes collect `(minusp ,lane)))
                               (or ,@(loop for lane in lanes collect `(zerop ,lane))))
                       (error "Invalid operand for ~A" operation))))))
           (setf (,aref destination (+ d-offset i))
                 (ecase operation
                   (:negate (,xor value (,pack ,(coerce -0.0 element))))
                   (:abs (,andc1 (,pack ,(coerce -0.0 element)) value))
                   (:sqrt (,sqrt value))
                   (:reciprocal (,divide (,pack ,(coerce 1 element)) value)))))
         (incf i ,width))
       (loop while (< i count) do
         (let ((value (aref input (+ i-offset i))))
           (setf (aref destination (+ d-offset i))
                 (ecase operation
                   (:negate (- value)) (:abs (abs value))
                   (:sqrt (kernel-sqrt value))
                   (:reciprocal (/ ,(coerce 1 element) value)))))
         (incf i)))
     destination)))

(define-sbcl-extended-unary %sbcl-extended-unary-f32 single-float 4
  sb-simd-sse:f32.4-aref sb-simd-sse:f32.4 sb-simd-sse:f32.4-xor
  sb-simd-sse:f32.4-andc1 sb-simd-sse:f32.4-sqrt sb-simd-sse:f32.4/)
(define-sbcl-extended-unary %sbcl-extended-unary-f64 double-float 2
  sb-simd-sse2:f64.2-aref sb-simd-sse2:f64.2 sb-simd-sse2:f64.2-xor
  sb-simd-sse2:f64.2-andc1 sb-simd-sse2:f64.2-sqrt sb-simd-sse2:f64.2/)

(define-sbcl-extended-unary %sbcl-nd-unary-f32 single-float 4
  sb-simd-sse:f32.4-aref sb-simd-sse:f32.4 sb-simd-sse:f32.4-xor
  sb-simd-sse:f32.4-andc1 sb-simd-sse:f32.4-sqrt sb-simd-sse:f32.4/ sb-simd-sse:f32.4-values)
(define-sbcl-extended-unary %sbcl-nd-unary-f64 double-float 2
  sb-simd-sse2:f64.2-aref sb-simd-sse2:f64.2 sb-simd-sse2:f64.2-xor
  sb-simd-sse2:f64.2-andc1 sb-simd-sse2:f64.2-sqrt sb-simd-sse2:f64.2/ sb-simd-sse2:f64.2-values)

(defmacro define-sbcl-scalar (suffix element width aref pack add sub mul div)
  (let ((binary (intern (format nil "%SBCL-SCALAR-BINARY-~A" suffix)))
        (axpy (intern (format nil "%SBCL-AXPY-~A" suffix))))
    `(progn
       (defun ,binary (operation destination left right count d-offset l-offset r-offset)
         (declare (type (simple-array ,element (*)) destination)
                  (type fixnum count d-offset))
         (let ((i 0) (left-vector (vectorp left)) (right-vector (vectorp right)))
           (declare (type fixnum i))
           (loop while (<= (+ i ,width) count) do
             (let ((a (if left-vector (,aref left (+ l-offset i)) (,pack left)))
                   (b (if right-vector (,aref right (+ r-offset i)) (,pack right))))
               (setf (,aref destination (+ d-offset i))
                     (ecase operation
                       (:add (,add a b)) (:subtract (,sub a b))
                       (:multiply (,mul a b)) (:divide (,div a b)))))
             (incf i ,width))
           (loop while (< i count) do
             (let ((a (if left-vector (aref left (+ l-offset i)) left))
                   (b (if right-vector (aref right (+ r-offset i)) right)))
               (setf (aref destination (+ d-offset i))
                     (ecase operation
                       (:add (+ a b)) (:subtract (- a b))
                       (:multiply (* a b)) (:divide (/ a b)))))
             (incf i))
           destination))
       (defun ,axpy (y a x count y-offset x-offset)
         (declare (type (simple-array ,element (*)) y x)
                  (type ,element a) (type fixnum count y-offset x-offset))
         (let ((i 0) (scalar (,pack a)))
           (declare (type fixnum i))
           (loop while (<= (+ i ,width) count) do
             (setf (,aref y (+ y-offset i))
                   (,add (,mul scalar (,aref x (+ x-offset i)))
                         (,aref y (+ y-offset i))))
             (incf i ,width))
           (loop while (< i count) do
             (setf (aref y (+ y-offset i))
                   (+ (* a (aref x (+ x-offset i))) (aref y (+ y-offset i))))
             (incf i))
           y)))))

(define-sbcl-scalar f32 single-float 4 sb-simd-sse:f32.4-aref sb-simd-sse:f32.4
  sb-simd-sse:f32.4+ sb-simd-sse:f32.4- sb-simd-sse:f32.4* sb-simd-sse:f32.4/)
(define-sbcl-scalar f64 double-float 2 sb-simd-sse2:f64.2-aref sb-simd-sse2:f64.2
  sb-simd-sse2:f64.2+ sb-simd-sse2:f64.2- sb-simd-sse2:f64.2* sb-simd-sse2:f64.2/)

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
        %sbcl-scalar-binary-f32 %sbcl-scalar-binary-f64
        %sbcl-axpy-f32 %sbcl-axpy-f64
        %sbcl-sum-f32 %sbcl-sum-f64 %sbcl-dot-f32 %sbcl-dot-f64))
