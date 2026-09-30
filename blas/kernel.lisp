(in-package #:trivial-simd/blas)

(deftype blas-dim () '(integer 0 1073741823))
(deftype blas-offset () '(unsigned-byte 60))

(defun kernel-symbol (base type)
  (intern (format nil "~A/~A" base
                  (if (consp type) (format nil "COMPLEX-~A" (second type)) type))
          :trivial-simd/blas))

(defmacro conj-if (type flag value)
  (if (consp type)
      `(if ,flag (conjugate ,value) ,value)
      `(progn ,flag ,value)))

(defmacro realify (type value)
  (if (consp type)
      (let ((v (gensym)))
        `(let ((,v ,value)) (complex (realpart ,v) (coerce 0 ',(second type)))))
      value))

(defmacro do-matrix-positions ((position base row-stride column-stride rows columns)
                               &body body)
  "Visit every element position, keeping the smaller stride in the inner loop."
  (let ((i (gensym)) (j (gensym)))
    `(if (>= ,row-stride ,column-stride)
         (dotimes (,i ,rows)
           (let ((,position (+ ,base (* ,i ,row-stride))))
             (declare (type fixnum ,position))
             (dotimes (,j ,columns)
               ,@body
               (incf ,position ,column-stride))))
         (dotimes (,j ,columns)
           (let ((,position (+ ,base (* ,j ,column-stride))))
             (declare (type fixnum ,position))
             (dotimes (,i ,rows)
               ,@body
               (incf ,position ,row-stride)))))))

(defmacro scale-matrix (type data base row-stride column-stride rows columns beta)
  "Set the matrix to BETA times itself, without reading it when BETA is zero."
  (let ((position (gensym)))
    `(unless (= ,beta 1)
       (if (zerop ,beta)
           (do-matrix-positions (,position ,base ,row-stride ,column-stride
                                 ,rows ,columns)
             (setf (aref ,data ,position) (coerce 0 ',type)))
           (do-matrix-positions (,position ,base ,row-stride ,column-stride
                                 ,rows ,columns)
             (setf (aref ,data ,position) (* ,beta (aref ,data ,position))))))))

(defun dense-strides (view transpose)
  "Return VIEW's storage, first-element offset, and the row and column strides
of op(VIEW)."
  (let* ((ld (matrix-view-leading-dimension view))
         (row-major (eq (matrix-view-layout view) :row-major))
         (row-stride (if row-major ld 1))
         (column-stride (if row-major 1 ld)))
    (if (eq transpose :no-transpose)
        (values (matrix-view-data view) (matrix-view-offset view)
                row-stride column-stride)
        (values (matrix-view-data view) (matrix-view-offset view)
                column-stride row-stride))))

(defun validate-kernel-limits (view)
  (unless (and (typep (matrix-view-rows view) 'blas-dim)
               (typep (matrix-view-cols view) 'blas-dim)
               (typep (matrix-view-leading-dimension view) 'blas-dim)
               (typep (matrix-view-offset view) 'blas-offset))
    (error "Matrix view is too large for the BLAS kernels")))
