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

(defmacro define-native-bulk-calls ()
  `(progn
     ,@(loop for (key element foreign) in *numeric-types*
             for scalar-foreign = (if (integer-type-p key)
                                      (intern (format nil "UINT~D" (fourth (numeric-type key))) :keyword)
                                      foreign)
             for suffix = (string-downcase key)
             for binary = (intern (format nil "%NATIVE-BULK-BINARY-~A" key))
             for axpy = (intern (format nil "%NATIVE-BULK-AXPY-~A" key))
             append
             `((define-native (,(format nil "ts_bulk_binary_~A" suffix) ,binary)
                   ,(if (integer-type-p key) :int :void)
                 (operation :uint) (output :pointer) (left :pointer) (right :pointer)
                 (left-scalar ,scalar-foreign) (right-scalar ,scalar-foreign) (count :size))
               (define-native (,(format nil "ts_bulk_axpy_~A" suffix) ,axpy)
                   ,(if (integer-type-p key) :int :void)
                 (output :pointer) (scalar ,scalar-foreign) (input :pointer) (count :size))))
     ,@(loop for key in '(:f32 :f64)
             for foreign = (third (numeric-type key))
             for name = (intern (format nil "%NATIVE-BULK-FMA-~A" key))
             collect
             `(define-native (,(format nil "ts_bulk_fma_~A" (string-downcase key)) ,name) :void
                (output :pointer) (x :pointer) (y :pointer) (z :pointer)
                (x-scalar ,foreign) (y-scalar ,foreign) (z-scalar ,foreign) (count :size)))))

(define-native-bulk-calls)

(defmacro define-native-extended-calls ()
  `(progn
     ,@(loop for (key element foreign) in *numeric-types*
             for suffix = (string-downcase key)
             append
             (loop for (operation arguments) in
                     '((unary ((opcode :uint) (output :pointer) (input :pointer) (count :size)))
                       (minmax ((opcode :uint) (output :pointer) (left :pointer) (right :pointer)
                                (left-scalar scalar) (right-scalar scalar) (count :size)))
                       (clamp ((output :pointer) (input :pointer) (lower :pointer) (upper :pointer)
                               (lower-scalar scalar) (upper-scalar scalar) (count :size)))
                       (compare ((opcode :uint) (mask :pointer) (left :pointer) (right :pointer)
                                 (left-scalar scalar) (right-scalar scalar) (count :size)))
                       (select ((output :pointer) (mask :pointer) (on-true :pointer) (on-false :pointer)
                                (true-scalar scalar) (false-scalar scalar) (count :size))))
                   collect
                   `(define-native (,(format nil "ts_extended_~A_~A"
                                            (string-downcase operation) suffix)
                                    ,(intern (format nil "%NATIVE-EXTENDED-~A-~A" operation key)))
                        :int
                      ,@(loop for (name type) in arguments
                              collect (list name (if (eq type 'scalar) foreign type))))))
     (define-native ("ts_extended_mask_reduce" %native-extended-mask-reduce) :int
       (opcode :uint) (mask :pointer) (count :size) (result :pointer))
     (define-native ("ts_extended_f32_to_f64" %native-extended-f32-to-f64) :void
       (output :pointer) (input :pointer) (count :size))
     (define-native ("ts_extended_f64_to_f32" %native-extended-f64-to-f32) :void
       (output :pointer) (input :pointer) (count :size))
     ,@(loop for (c-name lisp-name) in
               '(("ts_extended_bf16_to_f32" %native-extended-bf16-to-f32)
                 ("ts_extended_f16_to_f32" %native-extended-f16-to-f32))
             collect `(define-native (,c-name ,lisp-name) :void
                        (output :pointer) (input :pointer) (count :size)))
     ,@(loop for (c-name lisp-name) in
               '(("ts_extended_f32_to_bf16" %native-extended-f32-to-bf16)
                 ("ts_extended_f32_to_f16" %native-extended-f32-to-f16))
             collect `(define-native (,c-name ,lisp-name) :void
                        (output :pointer) (input :pointer) (count :size) (truncate :int)))))

(define-native-extended-calls)

(defvar *native-float-encoding-symbols*
  '("ts_extended_bf16_to_f32" "ts_extended_f32_to_bf16"
    "ts_extended_f16_to_f32" "ts_extended_f32_to_f16"))

(defun missing-native-symbols (names)
  (remove-if (lambda (name) (ignore-errors (cffi:foreign-symbol-pointer name))) names))

(defvar *native-float-encoding-available-p*
  (and *native-available-p*
       (null (missing-native-symbols *native-float-encoding-symbols*))))

(when (and *native-available-p* (not *native-float-encoding-available-p*))
  (warn "The native library lacks ~{~A~^, ~}, so bf16/f16 CONVERT! uses the slower ~
         Lisp path. Rebuild it with cmake."
        (missing-native-symbols *native-float-encoding-symbols*)))

(define-native ("ts_blas_gemm_f32" %native-blas-gemm-f32) :int
  (m :int64) (n :int64) (k :int64) (alpha :float)
  (a :pointer) (a-row-stride :int64) (a-column-stride :int64)
  (b :pointer) (b-row-stride :int64) (b-column-stride :int64) (beta :float)
  (c :pointer) (c-row-stride :int64) (c-column-stride :int64))
(define-native ("ts_blas_gemm_f64" %native-blas-gemm-f64) :int
  (m :int64) (n :int64) (k :int64) (alpha :double)
  (a :pointer) (a-row-stride :int64) (a-column-stride :int64)
  (b :pointer) (b-row-stride :int64) (b-column-stride :int64) (beta :double)
  (c :pointer) (c-row-stride :int64) (c-column-stride :int64))

(define-native ("ts_blas_rank_f32" %native-blas-rank-f32) :int
  (n :int64) (k :int64) (alpha :float)
  (a :pointer) (a-row-stride :int64) (a-column-stride :int64)
  (b :pointer) (b-row-stride :int64) (b-column-stride :int64) (beta :float)
  (c :pointer) (c-row-stride :int64) (c-column-stride :int64)
  (upper :int) (second :int))
(define-native ("ts_blas_rank_f64" %native-blas-rank-f64) :int
  (n :int64) (k :int64) (alpha :double)
  (a :pointer) (a-row-stride :int64) (a-column-stride :int64)
  (b :pointer) (b-row-stride :int64) (b-column-stride :int64) (beta :double)
  (c :pointer) (c-row-stride :int64) (c-column-stride :int64)
  (upper :int) (second :int))
(define-native ("ts_blas_triangular_f32" %native-blas-triangular-f32) :int
  (solve :int) (left :int) (upper :int) (unit :int) (alpha :float)
  (a :pointer) (a-row-stride :int64) (a-column-stride :int64)
  (b :pointer) (b-row-stride :int64) (b-column-stride :int64)
  (m :int64) (n :int64))
(define-native ("ts_blas_triangular_f64" %native-blas-triangular-f64) :int
  (solve :int) (left :int) (upper :int) (unit :int) (alpha :double)
  (a :pointer) (a-row-stride :int64) (a-column-stride :int64)
  (b :pointer) (b-row-stride :int64) (b-column-stride :int64)
  (m :int64) (n :int64))

(define-native ("ts_blas_gemv_f32" %native-blas-gemv-f32) :int
  (m :int64) (n :int64) (alpha :float)
  (a :pointer) (a-row-stride :int64) (a-column-stride :int64)
  (x :pointer) (x-increment :int64) (beta :float)
  (y :pointer) (y-increment :int64))
(define-native ("ts_blas_gemv_f64" %native-blas-gemv-f64) :int
  (m :int64) (n :int64) (alpha :double)
  (a :pointer) (a-row-stride :int64) (a-column-stride :int64)
  (x :pointer) (x-increment :int64) (beta :double)
  (y :pointer) (y-increment :int64))
(define-native ("ts_blas_ger_f32" %native-blas-ger-f32) :int
  (m :int64) (n :int64) (alpha :float)
  (x :pointer) (x-increment :int64) (y :pointer) (y-increment :int64)
  (a :pointer) (a-row-stride :int64) (a-column-stride :int64))
(define-native ("ts_blas_ger_f64" %native-blas-ger-f64) :int
  (m :int64) (n :int64) (alpha :double)
  (x :pointer) (x-increment :int64) (y :pointer) (y-increment :int64)
  (a :pointer) (a-row-stride :int64) (a-column-stride :int64))
(define-native ("ts_blas_trsv_f32" %native-blas-trsv-f32) :int
  (upper :int) (unit :int)
  (a :pointer) (a-row-stride :int64) (a-column-stride :int64)
  (n :int64) (x :pointer) (x-increment :int64))
(define-native ("ts_blas_trsv_f64" %native-blas-trsv-f64) :int
  (upper :int) (unit :int)
  (a :pointer) (a-row-stride :int64) (a-column-stride :int64)
  (n :int64) (x :pointer) (x-increment :int64))

(defvar *native-blas-available-p*
  (and *native-available-p*
       (every (lambda (name) (ignore-errors (cffi:foreign-symbol-pointer name)))
              '("ts_blas_gemm_f32" "ts_blas_gemm_f64"
                "ts_blas_rank_f32" "ts_blas_rank_f64"
                "ts_blas_triangular_f32" "ts_blas_triangular_f64"
                "ts_blas_gemv_f32" "ts_blas_gemv_f64"
                "ts_blas_ger_f32" "ts_blas_ger_f64"
                "ts_blas_trsv_f32" "ts_blas_trsv_f64"))))

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

(defmacro with-vector-pointer ((pointer vector) &body body)
  "Bind POINTER to the first element of VECTOR: a vector view's own memory, or a
Lisp vector's data pinned for BODY."
  (let ((value (gensym "VECTOR")))
    #+sbcl
    `(let ((,value ,vector))
       (sb-sys:with-pinned-objects (,value)
         (let ((,pointer (if (vector-view-p ,value)
                             (vector-view-pointer ,value)
                             (sb-sys:vector-sap
                              (the (sb-kernel:simple-unboxed-array (*)) ,value)))))
           ,@body)))
    ;; ECL does not move vectors.
    #+ecl
    `(let* ((,value ,vector)
            (,pointer (if (vector-view-p ,value)
                          (vector-view-pointer ,value)
                          (si:make-foreign-data-from-array ,value))))
       ,@body)
    ;; CCL pins a vector only for a body, so the body is a local function here.
    #-(or sbcl ecl)
    (let ((function (gensym "BODY")) (raw (gensym "RAW")))
      `(let ((,value ,vector))
         (flet ((,function (,pointer) ,@body))
           (declare (dynamic-extent #',function))
           (if (vector-view-p ,value)
               (,function (vector-view-pointer ,value))
               #+ccl (cffi:with-pointer-to-vector-data (,raw ,value) (,function ,raw))
               #-ccl (error "Pinned array access is unsupported on this implementation")))))))

(defmacro with-pinned-pointers ((&rest bindings) &body body)
  (if bindings
      `(with-vector-pointer ,(first bindings)
         (with-pinned-pointers ,(rest bindings) ,@body))
      `(progn ,@body)))

(defmacro with-copied-pointers ((&rest bindings) type outputs count &body body)
  "Bind each (POINTER VECTOR START) to a foreign copy of COUNT elements, copied
back for the POINTERs in OUTPUTS. A vector view binds its own memory instead."
  (let ((copies (loop for nil in bindings collect (gensym "COPY"))))
    `(cffi:with-foreign-objects
         ,(loop for copy in copies for (nil vector) in bindings
                collect `(,copy ,type (if (vector-view-p ,vector) 1 (max 1 ,count))))
       (let ,(loop for (pointer vector start) in bindings for copy in copies
                   collect `(,pointer (if (vector-view-p ,vector)
                                          (element-pointer (vector-view-pointer ,vector)
                                                           ,type ,start)
                                          ,copy)))
         ,@(loop for (pointer vector start) in bindings
                 unless (member pointer outputs)
                   collect `(unless (vector-view-p ,vector)
                              (copy-to-foreign ,vector ,pointer ,type ,start ,count)))
         (multiple-value-prog1 (progn ,@body)
           ,@(loop for (pointer vector start) in bindings
                   when (member pointer outputs)
                     collect `(unless (vector-view-p ,vector)
                                (copy-from-foreign ,pointer ,vector ,type ,start ,count))))))))

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

(defmacro with-native-bulk-output ((pointer vector offset type count &key read) &body body)
  (let ((raw (gensym "RAW")))
    `(if (or (eq *native-array-access* :pointer) (vector-view-p ,vector))
         (with-vector-pointer (,raw ,vector)
           (let ((,pointer (element-pointer ,raw ,type ,offset)))
             ,@body))
         (cffi:with-foreign-object (,pointer ,type (max 1 ,count))
           ,@(when read `((copy-to-foreign ,vector ,pointer ,type ,offset ,count)))
           (multiple-value-prog1 (progn ,@body)
             (copy-from-foreign ,pointer ,vector ,type ,offset ,count))))))

(defmacro with-native-bulk-input ((pointer value offset type count) &body body)
  (let ((raw (gensym "RAW")))
    `(cond ((not (operand-vector-p ,value))
            (let ((,pointer (cffi:null-pointer))) ,@body))
           ((or (eq *native-array-access* :pointer) (vector-view-p ,value))
            (with-vector-pointer (,raw ,value)
              (let ((,pointer (element-pointer ,raw ,type ,offset))) ,@body)))
           (t
            (cffi:with-foreign-object (,pointer ,type (max 1 ,count))
              (copy-to-foreign ,value ,pointer ,type ,offset ,count)
              ,@body)))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun native-bulk-symbol (prefix type)
    (intern (format nil "~A~A" prefix type) :trivial-simd)))

(defun native-bulk-function (prefix type)
  (symbol-function (native-bulk-symbol prefix type)))

(define-compiler-macro native-bulk-function (&whole form prefix type)
  (if (stringp prefix)
      `(ecase ,type
         ,@(loop for key in (append (mapcar #'first *numeric-types*) '(:c32 :c64))
                 collect `(,key (symbol-function ',(native-bulk-symbol prefix key)))))
      form))

(defun native-scalar-bits (type value)
  (if (integer-type-p type)
      (logand value (1- (ash 1 (fourth (numeric-type type)))))
      value))

(defun native-scalar-binary (operation destination left right count d-offset l-offset r-offset)
  (let* ((type (vector-type destination))
         (foreign (third (numeric-type type)))
         (opcode (+ (if (integer-type-p type) 2 0)
                    (position operation '(:add :subtract :multiply :divide)))))
    (with-native-bulk-output (output destination d-offset foreign count)
      (with-native-bulk-input (a left l-offset foreign count)
        (with-native-bulk-input (b right r-offset foreign count)
          (let ((status (funcall (native-bulk-function "%NATIVE-BULK-BINARY-" type)
                                 opcode output a b
                                 (if (operand-vector-p left) (coerce 0 (second (numeric-type type)))
                                     (native-scalar-bits type left))
                                 (if (operand-vector-p right) (coerce 0 (second (numeric-type type)))
                                     (native-scalar-bits type right))
                                 count)))
            (when (integer-type-p type) (check-native-integer-status status))))))
  destination))

(defun native-scale (x a count offset)
  (native-scalar-binary :multiply x x a count offset offset nil))

(defun native-axpy (y a x count y-offset x-offset)
  (let* ((type (vector-type y)) (foreign (third (numeric-type type))))
    (with-native-bulk-output (output y y-offset foreign count :read t)
      (with-native-bulk-input (input x x-offset foreign count)
        (let ((status (funcall (native-bulk-function "%NATIVE-BULK-AXPY-" type)
                               output (native-scalar-bits type a) input count)))
          (when (integer-type-p type) (check-native-integer-status status))))))
  y)

(defun native-bulk-fma (destination x y z count d-offset x-offset y-offset z-offset)
  (let* ((type (vector-type destination)) (foreign (third (numeric-type type))))
    (with-native-bulk-output (output destination d-offset foreign count)
      (with-native-bulk-input (a x x-offset foreign count)
        (with-native-bulk-input (b y y-offset foreign count)
          (with-native-bulk-input (c z z-offset foreign count)
            (funcall (native-bulk-function "%NATIVE-BULK-FMA-" type)
                     output a b c
                     (if (operand-vector-p x) (coerce 0 (second (numeric-type type))) x)
                     (if (operand-vector-p y) (coerce 0 (second (numeric-type type))) y)
                     (if (operand-vector-p z) (coerce 0 (second (numeric-type type))) z)
                     count)))))
  destination))

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

(macrolet ((define-float-reductions ()
             `(progn
                ,@(loop for (key foreign) in '((:f32 :float) (:f64 :double))
                        append
                        `((define-native (,(format nil "ts_asum_~(~A~)" key)
                                          ,(intern (format nil "%NATIVE-ASUM-~A" key)))
                              ,foreign (input :pointer) (length :size))
                          ,@(loop for name in '("ARGMIN" "ARGMAX" "IAMAX")
                                  collect
                                  `(define-native (,(format nil "ts_~(~A~)_~(~A~)" name key)
                                                   ,(intern (format nil "%NATIVE-~A-~A" name key)))
                                       :size (input :pointer) (length :size)))))
                (define-native ("ts_sum_acc_f32" %native-sum-acc-f32) :double
                  (input :pointer) (length :size))
                (define-native ("ts_asum_acc_f32" %native-asum-acc-f32) :double
                  (input :pointer) (length :size))
                (define-native ("ts_dot_acc_f32" %native-dot-acc-f32) :double
                  (left :pointer) (right :pointer) (length :size)))))
  (define-float-reductions))

(defun native-float-reduction (prefix input count offset)
  "Call the float reduction named by PREFIX on INPUT's slice."
  (let ((function (native-bulk-function prefix (vector-type input)))
        (type (foreign-type input)))
    (with-native-vectors (type ((a input offset)) :count count)
      (funcall function a count))))

(defun native-sum-acc (input count offset)
  (native-float-reduction "%NATIVE-SUM-ACC-" input count offset))

(defun native-asum (input count offset)
  (native-float-reduction "%NATIVE-ASUM-" input count offset))

(defun native-asum-acc (input count offset)
  (native-float-reduction "%NATIVE-ASUM-ACC-" input count offset))

(defun native-argmin (input count offset)
  (+ offset (native-float-reduction "%NATIVE-ARGMIN-" input count offset)))

(defun native-argmax (input count offset)
  (+ offset (native-float-reduction "%NATIVE-ARGMAX-" input count offset)))

(defun native-iamax (input count offset)
  (+ offset (native-float-reduction "%NATIVE-IAMAX-" input count offset)))

(defun native-dot-acc (left right count left-offset right-offset)
  (with-native-vectors (:float ((a left left-offset) (b right right-offset)) :count count)
    (%native-dot-acc-f32 a b count)))

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

(defun check-native-kernel-status (status)
  (case status
    (0 nil)
    (-2 (error "Negative kernel square root operand"))
    (-3 (error 'division-by-zero :operation 'truncate :operands nil))
    (otherwise (error "Unable to allocate native kernel scratch storage"))))

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
            (check-native-kernel-status status))
       (keep-native-program-alive program))))

(define-native-kernel-dispatch)

(defmacro define-native-mask-kernel-calls ()
  `(progn
     ,@(loop for (key) in *numeric-types*
             collect
             `(define-native (,(format nil "ts_kernel_mask_~A" (string-downcase key))
                              ,(intern (format nil "%NATIVE-KERNEL-MASK-~A" key))) :int
                (code :pointer) (code-length :size) (constants :pointer)
                (inputs :pointer) (input-count :size) (mask :pointer)
                (count :size) (scratch-count :size) (reduction :uint) (result :pointer)))))

(define-native-mask-kernel-calls)

(defun call-native-mask-kernel (program type inputs input-count mask count reduction result)
  (unwind-protect
       (let ((status
               (funcall
                (native-bulk-function "%NATIVE-KERNEL-MASK-"
                                      (first (find type *numeric-types* :key #'third)))
                (native-program-code program) (native-program-code-length program)
                (or (native-program-constants program)
                    (if (eq type :float) (native-program-f32-constants program)
                        (native-program-f64-constants program)))
                inputs input-count mask count (native-program-scratch-count program)
                reduction result)))
         (check-native-kernel-status status))
    (keep-native-program-alive program)))

(defmacro define-native-kernel-reduction-calls ()
  `(progn
     ,@(loop for (key) in *numeric-types*
             collect
             `(define-native (,(format nil "ts_kernel_reduction_~A" (string-downcase key))
                              ,(intern (format nil "%NATIVE-KERNEL-REDUCTION-~A" key))) :int
                (code :pointer) (code-length :size) (constants :pointer)
                (inputs :pointer) (input-count :size) (count :size) (scratch-count :size)
                (reducer :uint) (scale :double)
                (value :pointer) (wide :pointer) (index :pointer)))))

(define-native-kernel-reduction-calls)

(defun call-native-kernel-reduction (program type inputs input-count count reducer scale
                                     value wide index)
  "Run REDUCER (1 argmin, 2 argmax, 3 asum, 4 sum of squares, 5 largest magnitude,
6 sum of squares divided by SCALE) over the program's values."
  (unwind-protect
       (check-native-kernel-status
        (funcall
         (native-bulk-function "%NATIVE-KERNEL-REDUCTION-"
                               (first (find type *numeric-types* :key #'third)))
         (native-program-code program) (native-program-code-length program)
         (or (native-program-constants program)
             (if (eq type :float) (native-program-f32-constants program)
                 (native-program-f64-constants program)))
         inputs input-count count (native-program-scratch-count program)
         reducer scale value wide index))
    (keep-native-program-alive program)))
