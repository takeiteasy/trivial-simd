(in-package #:trivial-simd)

(defvar *native-available-p* nil)

(defun native-library-path ()
  (let ((name (case (uiop:operating-system)
                (:macosx #p"build/libtrivial_simd.dylib")
                (:linux #p"build/libtrivial_simd.so")
                ((:windows :win) #p"build/trivial_simd.dll"))))
    (when name
      (merge-pathnames name (asdf:system-source-directory "trivial-simd")))))

(let ((path (native-library-path)))
  (when (and path (probe-file path))
    (cffi:load-foreign-library path)
    (setf *native-available-p* t)))

(defmacro define-native ((c-name lisp-name) return-type &rest arguments)
  "Define LISP-NAME as a foreign call to C-NAME."
  #-ecl
  `(cffi:defcfun (,c-name ,lisp-name) ,return-type ,@arguments)
  ;; ECL's CFFI call by name looks up the symbol on every call (~22 us).
  #+ecl
  `(let ((pointer nil))
     (defun ,lisp-name ,(mapcar #'first arguments)
       (cffi:foreign-funcall-pointer
        (or pointer (setf pointer (cffi:foreign-symbol-pointer ,c-name))) ()
        ,@(loop for (name type) in arguments append (list type name))
        ,return-type))))

(define-native ("ts_add_f32" %native-add-f32) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(define-native ("ts_subtract_f32" %native-subtract-f32) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(define-native ("ts_multiply_f32" %native-multiply-f32) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(define-native ("ts_divide_f32" %native-divide-f32) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(define-native ("ts_sum_f32" %native-sum-f32) :float
  (input :pointer) (length :size))
(define-native ("ts_dot_f32" %native-dot-f32) :float
  (left :pointer) (right :pointer) (length :size))

(define-native ("ts_add_f64" %native-add-f64) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(define-native ("ts_subtract_f64" %native-subtract-f64) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(define-native ("ts_multiply_f64" %native-multiply-f64) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(define-native ("ts_divide_f64" %native-divide-f64) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(define-native ("ts_sum_f64" %native-sum-f64) :double
  (input :pointer) (length :size))
(define-native ("ts_dot_f64" %native-dot-f64) :double
  (left :pointer) (right :pointer) (length :size))

(define-native ("ts_kernel_f32" %native-kernel-f32) :int
  (code :pointer) (code-length :size) (constants :pointer)
  (inputs :pointer) (output :pointer) (length :size) (scratch-count :size))
(define-native ("ts_kernel_f64" %native-kernel-f64) :int
  (code :pointer) (code-length :size) (constants :pointer)
  (inputs :pointer) (output :pointer) (length :size) (scratch-count :size))

(define-native ("ts_kernel_sum_f32" %native-kernel-sum-f32) :int
  (code :pointer) (code-length :size) (constants :pointer)
  (inputs :pointer) (output :pointer) (length :size) (scratch-count :size))
(define-native ("ts_kernel_sum_f64" %native-kernel-sum-f64) :int
  (code :pointer) (code-length :size) (constants :pointer)
  (inputs :pointer) (output :pointer) (length :size) (scratch-count :size))

(defvar *native-array-access*
  #+(or sbcl ccl ecl) :pointer
  #-(or sbcl ccl ecl) :copy
  "How native calls reach Lisp float vectors: :POINTER (pinned) or :COPY.")

(defun foreign-type (vector)
  (if (typep vector '(simple-array single-float (*))) :float :double))

;; Public callers validate slice bounds before these unchecked transfers.
(macrolet ((define-copy-loops (element foreign to from)
             `(progn
                (defun ,to (vector pointer start count)
                  (declare (type (simple-array ,element (*)) vector)
                           (type fixnum start count)
                           (optimize (speed 3) (safety 0)))
                  (loop for index of-type fixnum below count
                        for source of-type fixnum from start
                        do (setf (cffi:mem-aref pointer ,foreign index)
                                 (aref vector source))))
                (defun ,from (pointer vector start count)
                  (declare (type (simple-array ,element (*)) vector)
                           (type fixnum start count)
                           (optimize (speed 3) (safety 0)))
                  (loop for index of-type fixnum below count
                        for destination of-type fixnum from start
                        do (setf (aref vector destination)
                                 (cffi:mem-aref pointer ,foreign index)))))))
  (define-copy-loops single-float :float copy-to-f32 copy-from-f32)
  (define-copy-loops double-float :double copy-to-f64 copy-from-f64))

(declaim (notinline copy-to-foreign))

(defun copy-to-foreign (vector pointer type start count)
  (ecase type
    (:float (copy-to-f32 vector pointer start count))
    (:double (copy-to-f64 vector pointer start count))))

(defun copy-from-foreign (pointer vector type start count)
  (ecase type
    (:float (copy-from-f32 pointer vector start count))
    (:double (copy-from-f64 pointer vector start count))))

(defmacro with-pinned-pointers ((&rest bindings) &body body)
  #+(or sbcl ccl ecl)
  (if bindings
      `(cffi:with-pointer-to-vector-data ,(first bindings)
         (with-pinned-pointers ,(rest bindings) ,@body))
      `(progn ,@body))
  #-(or sbcl ccl ecl)
  (progn bindings body
         '(error "Pinned array access is unsupported on this implementation")))

(defmacro with-copied-pointers ((&rest bindings) type outputs count &body body)
  `(cffi:with-foreign-objects
       ,(loop for (pointer) in bindings collect `(,pointer ,type (max 1 ,count)))
     ,@(loop for (pointer vector start) in bindings
             unless (member pointer outputs)
               collect `(copy-to-foreign ,vector ,pointer ,type ,start ,count))
     (multiple-value-prog1 (progn ,@body)
       ,@(loop for (pointer vector start) in bindings
               when (member pointer outputs)
                 collect `(copy-from-foreign ,pointer ,vector ,type ,start ,count)))))

(defmacro with-native-vectors ((type-var (&rest bindings) &key outputs count) &body body)
  "Bind (POINTER VECTOR START) to COUNT elements using *NATIVE-ARRAY-ACCESS*."
  (let ((function (gensym "BODY"))
        (pointers (mapcar #'first bindings)))
    `(flet ((,function ,pointers ,@body))
       (declare (dynamic-extent #',function))
       (ecase *native-array-access*
         (:pointer
          (with-pinned-pointers ,(mapcar (lambda (binding) (subseq binding 0 2)) bindings)
            (,function ,@(loop for (pointer nil start) in bindings
                               collect `(element-pointer ,pointer ,type-var ,start)))))
         (:copy (with-copied-pointers ,bindings ,type-var ,outputs ,count
                  (,function ,@pointers)))))))

(defun call-native-binary (operation type output a b length)
  (funcall (ecase operation
             (:add (if (eq type :float) #'%native-add-f32 #'%native-add-f64))
             (:subtract (if (eq type :float) #'%native-subtract-f32 #'%native-subtract-f64))
             (:multiply (if (eq type :float) #'%native-multiply-f32 #'%native-multiply-f64))
             (:divide (if (eq type :float) #'%native-divide-f32 #'%native-divide-f64)))
           output a b length))

(defun element-pointer (pointer type offset)
  (if (zerop offset)
      pointer
      (cffi:inc-pointer pointer (* offset (cffi:foreign-type-size type)))))

(defun native-binary (operation destination left right count
                      destination-offset left-offset right-offset)
  (let ((type (foreign-type destination)))
    (with-native-vectors (type ((output destination destination-offset)
                               (a left left-offset) (b right right-offset))
                          :outputs (output) :count count)
      (call-native-binary operation type output a b count)))
  destination)

(defun native-sum (input count offset)
  (let ((type (foreign-type input)))
    (with-native-vectors (type ((a input offset)) :count count)
      (if (eq type :float)
          (%native-sum-f32 a count)
          (%native-sum-f64 a count)))))

(defun native-dot (left right count left-offset right-offset)
  (let ((type (foreign-type left)))
    (with-native-vectors (type ((a left left-offset) (b right right-offset)) :count count)
      (if (eq type :float)
          (%native-dot-f32 a b count)
          (%native-dot-f64 a b count)))))

(defstruct (native-program (:constructor %make-native-program))
  code code-length f32-constants f64-constants scratch-count
  #+ecl runner)

(declaim (notinline foreign-copy free-native-program-buffers))

(defun free-native-program-buffers (buffers)
  (loop for tail on buffers
        for pointer = (shiftf (car tail) nil)
        when pointer do (cffi:foreign-free pointer)))

(defun native-program-finalizer (buffers)
  ;; Keep the owner out of the finalizer's lexical environment on interpreted ECL.
  (lambda () (free-native-program-buffers buffers)))

(defun foreign-copy (values type coerce-type)
  (let* ((buffers (list (cffi:foreign-alloc type :count (max 1 (length values)))))
         (pointer (first buffers))
         (complete nil))
    (unwind-protect
         (progn
           (when (cffi:null-pointer-p pointer)
             (error "Unable to allocate native kernel program storage"))
           (loop for value in values
                 for index from 0
                 do (setf (cffi:mem-aref pointer type index) (coerce value coerce-type)))
           (setf complete t)
           pointer)
      (unless complete (free-native-program-buffers buffers)))))

(defun make-native-program (code constants scratch-count)
  (declare (notinline trivial-garbage:finalize))
  (let ((buffers nil) (complete nil))
    (unwind-protect
         (flet ((copy (values type coerce-type)
                  (let ((pointer (foreign-copy values type coerce-type)))
                    (push pointer buffers)
                    pointer)))
           (let ((program (%make-native-program
                           :code (copy code :uint8 'integer)
                           :code-length (length code)
                           :scratch-count scratch-count
                           :f32-constants (copy constants :float 'single-float)
                           :f64-constants (copy constants :double 'double-float))))
             (trivial-garbage:finalize program (native-program-finalizer buffers))
             (setf complete t)
             program))
      (unless complete (free-native-program-buffers buffers)))))

;; The post-call reference keeps the owner live while C uses its raw pointers.
(declaim (notinline keep-native-program-alive))
(defun keep-native-program-alive (program)
  (native-program-code program))

(defun call-native-kernel (program type inputs output count &key sum-p)
  (unwind-protect
       (case (funcall (if sum-p
                         (if (eq type :float) #'%native-kernel-sum-f32 #'%native-kernel-sum-f64)
                         (if (eq type :float) #'%native-kernel-f32 #'%native-kernel-f64))
                      (native-program-code program) (native-program-code-length program)
                      (if (eq type :float)
                          (native-program-f32-constants program)
                          (native-program-f64-constants program))
                      inputs output count (native-program-scratch-count program))
         (0 nil)
         (-2 (error "Negative kernel square root operand"))
         (otherwise (error "Unable to allocate native kernel scratch storage")))
    (keep-native-program-alive program)))
