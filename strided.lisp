(in-package #:trivial-simd)

(defun resolve-strides (own-strides stride count)
  "Return COUNT nonzero integer strides: each vector's own stride, else STRIDE,
else 1. Return NIL, which every consumer reads as all 1, when no stride is given."
  (when (or stride (some #'identity own-strides))
    (let ((strides (loop for index below count
                         for value = (or (nth index own-strides) stride 1)
                         do (unless (and (integerp value) (/= value 0))
                              (error "Invalid stride ~S: a stride is a nonzero integer" value))
                         collect value)))
      (unless (every (lambda (value) (eql value 1)) strides)
        strides))))

(declaim (inline span-valid-p))
(defun span-valid-p (offset count stride length)
  "True when COUNT elements from OFFSET at STRIDE all lie inside LENGTH."
  (and (integerp offset) (<= 0 offset)
       (if (zerop count)
           (<= offset length)
           (let ((last (+ offset (* (1- count) stride))))
             (and (< offset length) (<= 0 last) (< last length))))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *strided-element-types*
    (append (mapcar #'second *numeric-types*)
            '((complex single-float) (complex double-float)))))

;; Public callers validate every span before these unchecked loops.
(defmacro define-strided-transfers ()
  `(progn
     (defun gather-slice (vector offset stride count)
       "Return a new vector of the COUNT elements of VECTOR from OFFSET at STRIDE."
       (declare (type fixnum offset stride count))
       (etypecase vector
         ,@(loop for element in *strided-element-types*
                 collect
                 `((simple-array ,element (*))
                   (let ((result (make-array count :element-type ',element))
                         (index offset))
                     (declare (type fixnum index) (optimize (speed 3) (safety 0)))
                     (dotimes (i count result)
                       (setf (aref result i) (aref vector index))
                       (incf index stride)))))))
     (defun scatter-slice (values vector offset stride count)
       "Store the COUNT elements of VALUES into VECTOR from OFFSET at STRIDE."
       (declare (type fixnum offset stride count))
       (etypecase vector
         ,@(loop for element in *strided-element-types*
                 collect
                 `((simple-array ,element (*))
                   (let ((index offset))
                     (declare (type fixnum index) (optimize (speed 3) (safety 0))
                              (type (simple-array ,element (*)) values))
                     (dotimes (i count)
                       (setf (aref vector index) (aref values i))
                       (incf index stride)))))))
     (defun strided-temporary (vector count)
       "Return an uninitialised vector like VECTOR with COUNT elements."
       (declare (type fixnum count))
       (etypecase vector
         ,@(loop for element in *strided-element-types*
                 collect `((simple-array ,element (*))
                           (make-array count :element-type ',element)))))))

(define-strided-transfers)

;; TODO: strided calls gather into temporaries and run the contiguous kernels;
;; native strided kernels would avoid the extra copies (#49).
(defmacro with-gathered ((&rest bindings) count &body body)
  "Run BODY with each (VECTOR OFFSET STRIDE [ROLE]) whose STRIDE is not 1 rebound
to a contiguous temporary at offset 0. ROLE is :IN (default), :OUT or :IN-OUT;
:OUT and :IN-OUT temporaries are scattered back afterwards. Scalars and NIL
strides pass through."
  (if (null bindings)
      `(progn ,@body)
      (destructuring-bind (vector offset stride &optional (role :in)) (first bindings)
        (let ((count-value (gensym "COUNT")) (original (gensym "ORIGINAL"))
              (original-offset (gensym "OFFSET")) (stride-value (gensym "STRIDE"))
              (temporary (gensym "TEMPORARY")))
          `(let* ((,count-value ,count) (,original ,vector)
                  (,original-offset ,offset) (,stride-value ,stride)
                  (,temporary (and (vectorp ,original) ,stride-value (/= ,stride-value 1)
                                   ,(if (eq role :out)
                                        `(strided-temporary ,original ,count-value)
                                        `(gather-slice ,original ,original-offset
                                                       ,stride-value ,count-value)))))
             (multiple-value-prog1
                 (let ((,vector (or ,temporary ,original))
                       (,offset (if ,temporary 0 ,original-offset)))
                   (with-gathered ,(rest bindings) ,count-value ,@body))
               ,@(unless (eq role :in)
                   `((when ,temporary
                       (scatter-slice ,temporary ,original ,original-offset
                                      ,stride-value ,count-value))))))))))
