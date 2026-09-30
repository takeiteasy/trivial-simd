(in-package #:trivial-simd/blas)

(deftype blas-dim () '(integer 0 1073741823))
(deftype blas-offset () '(unsigned-byte 60))

(deftype blas-increment () '(integer -1073741823 1073741823))

(defun kernel-symbol (base type)
  (intern (format nil "~A/~A" base
                  (if (consp type) (format nil "COMPLEX-~A" (second type)) type))
          :trivial-simd/blas))

(defun find-kernel (base type)
  (cdr (assoc type (get base 'kernels) :test #'equal)))

(defmacro define-kernel (base type lambda-list &body body)
  "Define the BASE kernel for element TYPE and register it for FIND-KERNEL."
  (let ((name (kernel-symbol base type)))
    `(progn
       (defun ,name ,lambda-list ,@body)
       (push (cons ',type #',name) (get ',base 'kernels))
       ',name)))

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
  (unless (and (typep (matrix-view-kl view) 'blas-dim)
               (typep (matrix-view-ku view) 'blas-dim)
               (typep (matrix-view-rows view) 'blas-dim)
               (typep (matrix-view-cols view) 'blas-dim)
               (typep (matrix-view-leading-dimension view) 'blas-dim)
               (typep (matrix-view-offset view) 'blas-offset))
    (error "Matrix view is too large for the BLAS kernels")))

(declaim (inline matrix-run))
(defun matrix-run (code row-major upper rows cols ld kl ku n major base)
  "Return LO, HI and the data position of minor index LO for storage-major
index MAJOR. CODE selects the storage: 0 dense, 1 dense triangle, 2 general
band, 3 triangular or symmetric band, 4 packed."
  (declare (type (integer 0 4) code) (type blas-dim rows cols ld kl ku n major)
           (type blas-offset base))
  (let ((minor (if row-major cols rows))
        (from-diagonal (eq (and upper t) (and row-major t))))
    (declare (type blas-dim minor))
    (ecase code
      (0 (values 0 (1- minor) (+ base (* major ld))))
      (1 (let ((lo (if from-diagonal major 0)))
           (values lo (if from-diagonal (1- minor) major)
                   (+ base (* major ld) lo))))
      (2 (let* ((below (if row-major kl ku)) (above (if row-major ku kl))
                (lo (max 0 (- major below))))
           (values lo (min (1- minor) (+ major above))
                   (+ base (* major ld) (- lo major) below))))
      (3 (let ((k (max kl ku)))
           (if from-diagonal
               (values major (min (1- minor) (+ major k)) (+ base (* major ld)))
               (let ((lo (max 0 (- major k))))
                 (values lo major (+ base (* major ld) k (- lo major)))))))
      (4 (if from-diagonal
             (values major (1- n)
                     (+ base (ash (* major (- (+ (* 2 n) 1) major)) -1)))
             (values 0 major (+ base (ash (* major (1+ major)) -1))))))))
