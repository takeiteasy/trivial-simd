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

(define-native ("ts_fma_scalar_f32" %native-fma-f32) :float
  (a :float) (b :float) (c :float))
(define-native ("ts_fma_scalar_f64" %native-fma-f64) :double
  (a :double) (b :double) (c :double))
(define-native ("ts_fma_supported" %native-fma-supported) :int)

(defvar *native-fma-available-p*
  (and *native-available-p*
       (every (lambda (name) (ignore-errors (cffi:foreign-symbol-pointer name)))
              '("ts_fma_scalar_f32" "ts_fma_scalar_f64" "ts_fma_supported"))))

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

(defmacro define-native-integers ()
  `(progn
     ,@(loop for (key) in *numeric-types* when (integer-type-p key)
             append
             (loop for operation in '(:add :subtract :multiply :divide :sum :dot :kernel :kernel-sum)
                   for c-name = (format nil "ts_~A_~A" (substitute #\_ #\- (string-downcase operation))
                                        (string-downcase key))
                   for name = (intern (format nil "%NATIVE-~A-~A" operation key))
                   collect
                   `(define-native (,c-name ,name) :int
                      ,@(case operation
                          (:sum '((input :pointer) (output :pointer) (count :size)))
                          (:dot '((left :pointer) (right :pointer) (output :pointer) (count :size)))
                          ((:kernel :kernel-sum)
                           '((code :pointer) (code-length :size) (constants :pointer)
                             (inputs :pointer) (output :pointer) (count :size) (scratch-count :size)))
                          (t '((output :pointer) (left :pointer) (right :pointer) (count :size)))))))))

(define-native-integers)

(defun check-native-integer-status (status)
  (case status
    (0 nil)
    (-3 (error 'division-by-zero :operation 'truncate :operands nil))
    (otherwise (error "Unable to allocate native integer kernel scratch storage"))))

(defmacro native-integer-result (pointer foreign)
  (let* ((info (find foreign *numeric-types* :key #'third))
         (value `(cffi:mem-ref ,pointer ,foreign)))
    (if (fifth info)
        `(,(integer-operation-symbol :wrap (first info)) ,value)
        value)))

(defvar *native-array-access*
  #+(or sbcl ccl ecl) :pointer
  #-(or sbcl ccl ecl) :copy
  "How native calls reach Lisp numeric vectors: :POINTER (pinned) or :COPY.")

(defun foreign-type (vector)
  (third (numeric-type (vector-type vector))))

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
  (define-copy-loops double-float :double copy-to-f64 copy-from-f64)
  (define-copy-loops (signed-byte 8) :int8 copy-to-s8 copy-from-s8)
  (define-copy-loops (unsigned-byte 8) :uint8 copy-to-u8 copy-from-u8)
  (define-copy-loops (signed-byte 16) :int16 copy-to-s16 copy-from-s16)
  (define-copy-loops (unsigned-byte 16) :uint16 copy-to-u16 copy-from-u16)
  (define-copy-loops (signed-byte 32) :int32 copy-to-s32 copy-from-s32)
  (define-copy-loops (unsigned-byte 32) :uint32 copy-to-u32 copy-from-u32)
  (define-copy-loops (signed-byte 64) :int64 copy-to-s64 copy-from-s64)
  (define-copy-loops (unsigned-byte 64) :uint64 copy-to-u64 copy-from-u64))

(declaim (notinline copy-to-foreign))

(defun copy-to-foreign (vector pointer type start count)
  (ecase type
    (:float (copy-to-f32 vector pointer start count))
    (:double (copy-to-f64 vector pointer start count))
    (:int8 (copy-to-s8 vector pointer start count))
    (:uint8 (copy-to-u8 vector pointer start count))
    (:int16 (copy-to-s16 vector pointer start count))
    (:uint16 (copy-to-u16 vector pointer start count))
    (:int32 (copy-to-s32 vector pointer start count))
    (:uint32 (copy-to-u32 vector pointer start count))
    (:int64 (copy-to-s64 vector pointer start count))
    (:uint64 (copy-to-u64 vector pointer start count))))

(defun copy-from-foreign (pointer vector type start count)
  (ecase type
    (:float (copy-from-f32 pointer vector start count))
    (:double (copy-from-f64 pointer vector start count))
    (:int8 (copy-from-s8 pointer vector start count))
    (:uint8 (copy-from-u8 pointer vector start count))
    (:int16 (copy-from-s16 pointer vector start count))
    (:uint16 (copy-from-u16 pointer vector start count))
    (:int32 (copy-from-s32 pointer vector start count))
    (:uint32 (copy-from-u32 pointer vector start count))
    (:int64 (copy-from-s64 pointer vector start count))
    (:uint64 (copy-from-u64 pointer vector start count))))

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

(defmacro define-native-binary-dispatch ()
  `(defun call-native-binary (operation type output a b length)
     (ecase type
       ,@(loop for (key element foreign) in *numeric-types*
               collect
               `(,foreign
                 ,(let ((call `(funcall (ecase operation
                                         ,@(loop for op in '(:add :subtract :multiply :divide)
                                                 collect `(,op #',(intern (format nil "%NATIVE-~A-~A" op key)))))
                                       output a b length)))
                    (if (integer-type-p key) `(check-native-integer-status ,call) call)))))))

(define-native-binary-dispatch)

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

(defmacro define-native-reductions ()
  `(progn
     ,@(loop for dot-p in '(nil t)
             for name = (if dot-p 'native-dot 'native-sum)
             collect
             `(defun ,name ,(if dot-p '(left right count left-offset right-offset) '(input count offset))
                (let ((type (foreign-type ,(if dot-p 'left 'input))))
                  (with-native-vectors
                      (type ,(if dot-p '((a left left-offset) (b right right-offset)) '((a input offset))) :count count)
                    (ecase type
                      ,@(loop for (key element foreign) in *numeric-types*
                              for function = (intern (format nil "%NATIVE-~A-~A" (if dot-p :dot :sum) key))
                              collect
                              `(,foreign
                                ,(if (integer-type-p key)
                                     `(cffi:with-foreign-object (output ,foreign)
                                        (check-native-integer-status (,function a ,@(when dot-p '(b)) output count))
                                        (native-integer-result output ,foreign))
                                     `(,function a ,@(when dot-p '(b)) count)))))))))))

(define-native-reductions)

(defstruct (native-program (:constructor %make-native-program))
  code code-length f32-constants f64-constants constants constant-type scratch-count
  #+ecl runner)

(declaim (notinline foreign-copy free-native-program-buffers))

(defun free-native-program-buffers (buffers)
  (loop for tail on buffers
        for pointer = (shiftf (car tail) nil)
        when pointer do (cffi:foreign-free pointer)))

(defun native-program-finalizer (buffers)
  ;; Keep the owner out of the finalizer's lexical environment on interpreted ECL.
  (lambda () (free-native-program-buffers buffers)))

#+ccl
(defun ccl-native-finalizer (callback)
  (lambda (owner) (declare (ignore owner)) (funcall callback)))

(defun register-native-program-finalizer (program callback)
  ;; Register directly without a second weak bookkeeping table.
  #+ccl (ccl:terminate-when-unreachable program (ccl-native-finalizer callback))
  #-ccl (trivial-garbage:finalize program callback))

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

(defun make-native-program (code constants scratch-count &optional type)
  (declare (notinline register-native-program-finalizer))
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
                           :constant-type type
                           :constants (when type
                                        (let ((info (numeric-type type)))
                                          (copy constants (third info) (second info))))
                           :f32-constants (unless type (copy constants :float 'single-float))
                           :f64-constants (unless type (copy constants :double 'double-float)))))
             (register-native-program-finalizer program (native-program-finalizer buffers))
             (setf complete t)
             program))
      (unless complete (free-native-program-buffers buffers)))))

;; The post-call reference keeps the owner live while C uses its raw pointers.
(declaim (notinline keep-native-program-alive))
(defun keep-native-program-alive (program)
  (native-program-code program))

(defmacro define-native-kernel-dispatch ()
  `(defun call-native-kernel (program type inputs output count &key sum-p)
     (unwind-protect
          (let ((status
                  (funcall
                   (ecase type
                     ,@(loop for (key element foreign) in *numeric-types*
                             collect
                             `(,foreign (if sum-p
                                            #',(intern (format nil "%NATIVE-KERNEL-SUM-~A" key))
                                            #',(intern (format nil "%NATIVE-KERNEL-~A" key))))))
                   (native-program-code program) (native-program-code-length program)
                   (or (native-program-constants program)
                       (if (eq type :float) (native-program-f32-constants program)
                           (native-program-f64-constants program)))
                   inputs output count (native-program-scratch-count program))))
            (case status
              (0 nil)
              (-2 (error "Negative kernel square root operand"))
              (-3 (error 'division-by-zero :operation 'truncate :operands nil))
              (otherwise (error "Unable to allocate native kernel scratch storage"))))
       (keep-native-program-alive program))))

(define-native-kernel-dispatch)
