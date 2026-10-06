(in-package #:trivial-simd/blas)

(deftype blas-dim () '(integer 0 1073741823))
(deftype blas-offset () '(unsigned-byte 60))
(deftype blas-batch-stride () '(integer -1152921504606846975 1152921504606846975))

(deftype blas-increment () '(integer -1073741823 1073741823))

(defun kernel-symbol (base type)
  (intern (format nil "~A/~A" base
                  (if (consp type) (format nil "COMPLEX-~A" (second type)) type))
          :trivial-simd/blas))

(defun find-kernel (base type)
  (cdr (assoc type (get base 'kernels) :test #'equal)))

(defmacro define-kernel (base type lambda-list &body body)
  "Define the BASE kernel for element TYPE and register it for FIND-KERNEL.
Parameters declared as simple vectors of TYPE may also be vector views: calls
with a view run a variant that reads every such parameter through its pointer."
  (let ((name (kernel-symbol base type)))
    (multiple-value-bind (storage declarations forms) (storage-parameters body type)
      (if (null storage)
          `(progn
             (defun ,name ,lambda-list ,@body)
             (push (cons ',type #',name) (get ',base 'kernels))
             ',name)
          (let ((view-name (kernel-symbol (format nil "~A/VIEW" base) type))
                (dispatch-name (kernel-symbol (format nil "~A/DISPATCH" base) type)))
            (when (intersection lambda-list lambda-list-keywords)
              (error "A BLAS kernel takes only required parameters"))
            `(progn
               (defun ,name ,lambda-list ,@body)
               (defun ,view-name ,lambda-list
                 ,@(view-kernel-body type storage declarations forms))
               (defun ,dispatch-name ,lambda-list
                 (if (or ,@(loop for variable in storage
                                 collect `(trivial-simd:vector-view-p ,variable)))
                     (,view-name ,@lambda-list)
                     (,name ,@lambda-list)))
               (push (cons ',type #',dispatch-name) (get ',base 'kernels))
               ',name))))))

(defvar *native-blas-threshold* #+ecl 1 #-ecl 1000
  "Smallest operation count (multiply-adds) sent to the native BLAS kernels.")

(defvar *native-blas-level2-threshold* #+ecl 1 #-ecl 1000
  "Smallest matrix element count sent to the native Level 2 kernels.")

(defun native-blas-p (type work &optional (threshold *native-blas-threshold*))
  "True when real TYPE and WORK multiply-adds should use the native kernels."
  (and (member type '(single-float double-float))
       trivial-simd::*native-blas-available-p*
       (not (eq (trivial-simd:backend) :lisp))
       (eq trivial-simd::*native-array-access* :pointer)
       (plusp work)
       (>= work threshold)))

(defun check-native-blas-status (status)
  (unless (zerop status)
    (error "Unable to allocate native BLAS scratch storage")))

(defmacro call-native-blas (type (single-function double-function) (&rest arrays)
                            &rest arguments)
  "Call the native function for real TYPE. Each (POINTER ARRAY OFFSET) in
ARRAYS binds POINTER to the pinned element at OFFSET for use in ARGUMENTS."
  (flet ((call (function foreign)
           `(trivial-simd::with-pinned-pointers
                ,(loop for (pointer array) in arrays collect (list pointer array))
              (let ,(loop for (pointer nil offset) in arrays
                          collect `(,pointer (trivial-simd::element-pointer
                                              ,pointer ,foreign ,offset)))
                (check-native-blas-status (,function ,@arguments))))))
    `(if (eq ,type 'single-float)
         ,(call single-function :float)
         ,(call double-function :double))))

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
  (let ((row-stride (if (= (matrix-view-rows view) 1) 0 (matrix-view-row-stride view)))
        (column-stride (if (= (matrix-view-cols view) 1) 0 (matrix-view-column-stride view))))
    (if (eq transpose :no-transpose)
        (values (matrix-view-data view) (matrix-view-offset view)
                row-stride column-stride)
        (values (matrix-view-data view) (matrix-view-offset view)
                column-stride row-stride))))

(defun validate-kernel-limits (view &optional strided)
  (when (and (not strided) (eq (matrix-view-layout view) :strided))
    (error "This BLAS operation requires a regular matrix layout"))
  (unless (and (typep (matrix-view-kl view) 'blas-dim)
               (typep (matrix-view-ku view) 'blas-dim)
               (typep (matrix-view-rows view) 'blas-dim)
               (typep (matrix-view-cols view) 'blas-dim)
               (if (and strided (eq (matrix-view-kind view) :dense))
                   (and (or (= (matrix-view-rows view) 1)
                            (typep (matrix-view-row-stride view) 'blas-increment))
                        (or (= (matrix-view-cols view) 1)
                            (typep (matrix-view-column-stride view) 'blas-increment)))
                   (typep (matrix-view-leading-dimension view) 'blas-dim))
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
