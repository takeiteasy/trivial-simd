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

(define-native ("ts_kernel_f32" %native-kernel-f32) :void
  (code :pointer) (code-length :size) (constants :pointer)
  (inputs :pointer) (output :pointer) (length :size))
(define-native ("ts_kernel_f64" %native-kernel-f64) :void
  (code :pointer) (code-length :size) (constants :pointer)
  (inputs :pointer) (output :pointer) (length :size))

(defvar *native-array-access*
  #+(or sbcl ccl ecl) :pointer
  #-(or sbcl ccl ecl) :copy
  "How native calls reach Lisp float vectors: :POINTER (pinned) or :COPY.")

(defun foreign-type (vector)
  (if (typep vector '(simple-array single-float (*))) :float :double))

;; TODO: generic per-element copy is ~3000x slower than pinned access (ticket #33).
(defun copy-to-foreign (vector pointer type)
  (dotimes (index (length vector))
    (setf (cffi:mem-aref pointer type index) (aref vector index))))

(defun copy-from-foreign (pointer vector type)
  (dotimes (index (length vector))
    (setf (aref vector index) (cffi:mem-aref pointer type index))))

(defmacro with-pinned-pointers ((&rest bindings) &body body)
  #+(or sbcl ccl ecl)
  (if bindings
      `(cffi:with-pointer-to-vector-data ,(first bindings)
         (with-pinned-pointers ,(rest bindings) ,@body))
      `(progn ,@body))
  #-(or sbcl ccl ecl)
  (progn bindings body
         '(error "Pinned array access is unsupported on this implementation")))

(defmacro with-copied-pointers ((&rest bindings) type outputs &body body)
  `(cffi:with-foreign-objects
       ,(loop for (pointer vector) in bindings
              collect `(,pointer ,type (length ,vector)))
     ,@(loop for (pointer vector) in bindings
             collect `(copy-to-foreign ,vector ,pointer ,type))
     (multiple-value-prog1 (progn ,@body)
       ,@(loop for (pointer vector) in bindings
               when (member pointer outputs)
                 collect `(copy-from-foreign ,pointer ,vector ,type)))))

(defmacro with-native-vectors ((type-var (&rest bindings) &key outputs) &body body)
  "Bind each (POINTER VECTOR) to a native pointer using *NATIVE-ARRAY-ACCESS*.
Vectors named by OUTPUTS' pointers are copied back in :COPY mode."
  (let ((function (gensym "BODY"))
        (pointers (mapcar #'first bindings)))
    `(flet ((,function ,pointers ,@body))
       (declare (dynamic-extent #',function))
       (ecase *native-array-access*
         (:pointer (with-pinned-pointers ,bindings (,function ,@pointers)))
         (:copy (with-copied-pointers ,bindings ,type-var ,outputs
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
    (with-native-vectors (type ((output destination) (a left) (b right))
                          :outputs (output))
      (call-native-binary operation type
                          (element-pointer output type destination-offset)
                          (element-pointer a type left-offset)
                          (element-pointer b type right-offset)
                          count)))
  destination)

(defun native-sum (input count offset)
  (let ((type (foreign-type input)))
    (with-native-vectors (type ((a input)))
      (let ((a (element-pointer a type offset)))
        (if (eq type :float)
            (%native-sum-f32 a count)
            (%native-sum-f64 a count))))))

(defun native-dot (left right count left-offset right-offset)
  (let ((type (foreign-type left)))
    (with-native-vectors (type ((a left) (b right)))
      (let ((a (element-pointer a type left-offset))
            (b (element-pointer b type right-offset)))
        (if (eq type :float)
            (%native-dot-f32 a b count)
            (%native-dot-f64 a b count))))))

(defstruct (native-program (:constructor %make-native-program))
  code code-length f32-constants f64-constants)

(defun foreign-copy (values type coerce-type)
  (let ((pointer (cffi:foreign-alloc type :count (max 1 (length values)))))
    (loop for value in values
          for index from 0
          do (setf (cffi:mem-aref pointer type index) (coerce value coerce-type)))
    pointer))

;; TODO: program memory is never freed; free it if kernels become dynamic (ticket #34).
(defun make-native-program (code constants)
  (%make-native-program :code (foreign-copy code :uint8 'integer)
                        :code-length (length code)
                        :f32-constants (foreign-copy constants :float 'single-float)
                        :f64-constants (foreign-copy constants :double 'double-float)))

(defun call-native-kernel (program type inputs output count)
  (if (eq type :float)
      (%native-kernel-f32 (native-program-code program) (native-program-code-length program)
                          (native-program-f32-constants program) inputs output count)
      (%native-kernel-f64 (native-program-code program) (native-program-code-length program)
                          (native-program-f64-constants program) inputs output count)))
