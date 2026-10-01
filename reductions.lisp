(in-package #:trivial-simd)

(defmacro define-lisp-reductions ()
  `(progn
     (defun lisp-argext (type input count offset maximum-p)
       (ecase type
         ,@(loop for (key element) in *numeric-types*
                 collect
                 `(,key
                   (locally
                       (declare (type (simple-array ,element (*)) input)
                                (type fixnum count offset))
                     (let ((best (aref input offset)) (index 0))
                       (declare (type ,element best) (type fixnum index))
                       (loop for i of-type fixnum from 1 below count
                             for value of-type ,element = (aref input (+ offset i))
                             do (when (if maximum-p (> value best) (< value best))
                                  (setf best value index i)))
                       (+ offset index)))))))
     (defun lisp-asum (type input count offset wide)
       (ecase type
         ,@(loop for (key element) in *numeric-types*
                 collect
                 `(,key
                   (locally
                       (declare (type (simple-array ,element (*)) input)
                                (type fixnum count offset))
                     ,(cond
                        ((integer-type-p key)
                         `(let ((result 0))
                            (declare (type ,element result))
                            (dotimes (i count result)
                              (setf result (,(integer-operation-symbol :add key)
                                            result
                                            (,(integer-operation-symbol :abs key)
                                             (aref input (+ offset i))))))))
                        ((eq key :f32)
                         `(if wide
                              (let ((result 0.0d0))
                                (declare (type double-float result))
                                (dotimes (i count result)
                                  (incf result (abs (coerce (aref input (+ offset i)) 'double-float)))))
                              (let ((result 0.0f0))
                                (declare (type single-float result))
                                (dotimes (i count result)
                                  (incf result (abs (aref input (+ offset i))))))))
                        (t
                         `(let ((result 0.0d0))
                            (declare (type double-float result))
                            (dotimes (i count result)
                              (incf result (abs (aref input (+ offset i)))))))))))))
     (defun lisp-sum-wide (input count offset)
       (declare (type (simple-array single-float (*)) input) (type fixnum count offset))
       (let ((result 0.0d0))
         (declare (type double-float result))
         (dotimes (i count result)
           (incf result (coerce (aref input (+ offset i)) 'double-float)))))
     (defun lisp-dot-wide (left right count left-offset right-offset)
       (declare (type (simple-array single-float (*)) left right)
                (type fixnum count left-offset right-offset))
       (let ((result 0.0d0))
         (declare (type double-float result))
         (dotimes (i count result)
           (incf result (* (coerce (aref left (+ left-offset i)) 'double-float)
                           (coerce (aref right (+ right-offset i)) 'double-float))))))))

(define-lisp-reductions)

(defmacro define-lisp-iamax ()
  `(defun lisp-iamax (type input count offset)
     (ecase type
       ,@(loop for (key element) in '((:f32 single-float) (:f64 double-float))
               collect
               `(,key
                 (locally
                     (declare (type (simple-array ,element (*)) input)
                              (type fixnum count offset))
                   (let ((best (abs (aref input offset))) (index 0))
                     (declare (type ,element best) (type fixnum index))
                     (loop for i of-type fixnum from 1 below count
                           for value of-type ,element = (abs (aref input (+ offset i)))
                           do (when (> value best) (setf best value index i)))
                     (+ offset index))))))))

(define-lisp-iamax)

(defun absolute-argmax (input count offset)
  "Index in INPUT of the first element of largest magnitude among COUNT > 0 real floats."
  (if (eq *backend* :native)
      (native-iamax input count offset)
      (lisp-iamax (vector-type input) input count offset)))

;; TODO: SBCL and integer vectors use typed scalar loops; SIMD paths in #92
(defun real-argext (maximum-p input count offset)
  (if (and (eq *backend* :native) (member (vector-type input) '(:f32 :f64)))
      (if maximum-p
          (native-argmax input count offset)
          (native-argmin input count offset))
      (lisp-argext (vector-type input) input count offset maximum-p)))

(defun argext (name maximum-p input start end input-start)
  (multiple-value-bind (type count offsets)
      (resolve-slice (list input) (list input-start) start end)
    (when (complex-type-p type)
      (error "~A requires real vectors" name))
    (unless (zerop count)
      (real-argext maximum-p input count (first offsets)))))

(defun argmin (input &key start end input-start)
  "Return the index in INPUT of the first smallest element of the slice, or NIL if empty."
  (argext 'argmin nil input start end input-start))

(defun argmax (input &key start end input-start)
  "Return the index in INPUT of the first largest element of the slice, or NIL if empty."
  (argext 'argmax t input start end input-start))

(defun minimum (input &key start end input-start)
  "Return the smallest element of the slice, or NIL if empty."
  (let ((index (argext 'minimum nil input start end input-start)))
    (and index (aref input index))))

(defun maximum (input &key start end input-start)
  "Return the largest element of the slice, or NIL if empty."
  (let ((index (argext 'maximum t input start end input-start)))
    (and index (aref input index))))

(defun asum (input &key start end input-start accumulate)
  "Return the sum of absolute values (complex moduli) of the slice."
  (multiple-value-bind (type count offsets)
      (resolve-slice (list input) (list input-start) start end)
    (let ((wide (wide-accumulation-p type accumulate))
          (offset (first offsets)))
      (cond ((complex-type-p type) (complex-asum input count offset wide))
            ((and (eq *backend* :native) (not (integer-type-p type)))
             (if wide
                 (native-asum-acc input count offset)
                 (native-asum input count offset)))
            (t (lisp-asum type input count offset wide))))))

(defun scaled-norm-function (count element)
  "Euclidean norm of the real or complex values (funcall ELEMENT i) for i below COUNT,
scaled to avoid overflow and underflow."
  (flet ((components (value)
           (if (complexp value)
               (list (realpart value) (imagpart value))
               (list value))))
    (let ((largest 0d0))
      (dotimes (i count)
        (dolist (part (components (funcall element i)))
          (setf largest (max largest (abs part)))))
      (if (zerop largest)
          0d0
          (let ((sum 0d0))
            (dotimes (i count)
              (dolist (part (components (funcall element i)))
                (incf sum (expt (/ part largest) 2))))
            (* largest (sqrt sum)))))))

(defun scaled-norm (input count offset)
  "Euclidean norm of double-float or complex double-float elements without overflow."
  (scaled-norm-function count (lambda (i) (aref input (+ offset i)))))

(defconstant +nrm2-fast-lower+ 1d-280
  "Squared sums below this may have lost terms to underflow.")

(defun nrm2-double (input count offset)
  (let ((sum (handler-case (dot input input :start offset :end (+ offset count))
               (arithmetic-error () nil))))
    (if (and sum (<= +nrm2-fast-lower+ sum most-positive-double-float))
        (sqrt sum)
        (scaled-norm input count offset))))

(defun nrm2 (input &key start end input-start accumulate)
  "Return the Euclidean norm of the slice, scaled to avoid overflow and underflow."
  (multiple-value-bind (type count offsets)
      (resolve-slice (list input) (list input-start) start end)
    (when (integer-type-p type)
      (error "NRM2 requires float or complex vectors"))
    (let ((wide (wide-accumulation-p type accumulate))
          (offset (first offsets)))
      (flet ((narrow (value) (if wide value (coerce value 'single-float))))
        (ecase type
          (:f32 (narrow (sqrt (dot input input :start offset :end (+ offset count)
                                               :accumulate :f64))))
          (:f64 (nrm2-double input count offset))
          (:c32 (narrow (complex-nrm2 input count offset)))
          (:c64 (scaled-norm input count offset)))))))
