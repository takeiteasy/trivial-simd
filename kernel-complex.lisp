(in-package #:trivial-simd)

(macrolet ((define-complex-kernel-calls ()
             `(progn
                ,@(loop for suffix in '(c32 c64)
                        append
                        `((define-native (,(format nil "ts_kernel_~(~A~)" suffix)
                                          ,(intern (format nil "%NATIVE-KERNEL-~A" suffix))) :int
                            (code :pointer) (code-length :size) (constants :pointer)
                            (inputs :pointer) (output :pointer) (count :size) (scratch-count :size))
                          (define-native (,(format nil "ts_kernel_mask_~(~A~)" suffix)
                                          ,(intern (format nil "%NATIVE-KERNEL-MASK-~A" suffix))) :int
                            (code :pointer) (code-length :size) (constants :pointer)
                            (inputs :pointer) (input-count :size) (mask :pointer)
                            (count :size) (scratch-count :size) (reduction :uint) (result :pointer))
                          (define-native (,(format nil "ts_kernel_reduction_~(~A~)" suffix)
                                          ,(intern (format nil "%NATIVE-KERNEL-REDUCTION-~A" suffix))) :int
                            (code :pointer) (code-length :size) (constants :pointer)
                            (inputs :pointer) (input-count :size) (count :size) (scratch-count :size)
                            (reducer :uint) (scale :double) (value :pointer) (wide :pointer) (index :pointer)))))))
  (define-complex-kernel-calls))

(defvar *native-complex-kernels-p*
  (and *native-available-p*
       (null (missing-native-symbols
              '("ts_kernel_c32" "ts_kernel_c64" "ts_kernel_mask_c32" "ts_kernel_mask_c64"
                "ts_kernel_reduction_c32" "ts_kernel_reduction_c64")))))

(defun native-complex-kernel-p ()
  (and (eq *backend* :native) *native-complex-kernels-p*))

(defstruct (complex-kernel (:constructor make-complex-kernel (arguments tree kind reducer)))
  arguments tree kind reducer
  (programs (make-array 2 :initial-element nil))
  (functions (make-array 2 :initial-element nil)))

(defun complex-tree-expression (tree arguments)
  (labels ((walk (node)
             (ecase (first node)
               (:argument (nth (second node) arguments))
               (:constant (second node))
               (:negate (list '- (walk (second node))))
               (:unary (list (car (rassoc (second node) *kernel-operators*)) (walk (third node))))
               (:fma (cons 'fma (mapcar #'walk (rest node))))
               (:select (cons 'select (mapcar #'walk (rest node))))
               (:operation
                (cons (or (car (rassoc (second node) '((= . :eq) (/= . :ne) (< . :lt) (<= . :le)
                                                      (> . :gt) (>= . :ge))))
                          (car (rassoc (second node) *kernel-operators*)))
                      (mapcar #'walk (cddr node)))))))
    (walk tree)))

(defun complex-expression-function (kernel type)
  (let ((slot (if (eq type :c32) 0 1)) (cache (complex-kernel-functions kernel)))
    (or (aref cache slot)
        (setf (aref cache slot)
              (let* ((arguments (complex-kernel-arguments kernel)) (reducer (complex-kernel-reducer kernel))
                     (body (complex-tree-expression (complex-kernel-tree kernel) arguments))
                     (expression (if reducer (list reducer body) body))
                     (form (declared-kernel-loop arguments (loop for nil in arguments collect '(nil 1))
                                                expression (complex-kernel-kind kernel) reducer type)))
                ;; TODO: CCL ARM64 drops typed complex operands; inline after a compiler fix (#138).
                #+ccl (setf form `(lambda ,(second form)
                                   (declare (notinline + - * / sqrt abs = /=))
                                   ,@(cddr form)))
                (#+ecl eval #-ecl compile #-ecl nil form))))))

(declaim (notinline call-native-complex-kernel))

(defun call-native-complex-kernel (program type inputs input-count output mask count kind reducer)
  (with-kernel-program (program type)
    (multiple-value-bind (foreign element) (native-constant-types type)
      (declare (ignore element))
      (cffi:with-foreign-objects ((value foreign 2) (wide :double) (index :size))
        (let ((prefix (cond ((eq kind :mask) "%NATIVE-KERNEL-MASK-")
                            (reducer "%NATIVE-KERNEL-REDUCTION-")
                            (t "%NATIVE-KERNEL-"))))
          (unwind-protect
               (let* ((function (native-bulk-function prefix type))
                      (common (list (native-program-code program) (native-program-code-length program)
                                    (native-program-constants program) inputs))
                      (status
                        (apply function
                               (append common
                                       (cond ((eq kind :mask)
                                              (list input-count mask count (native-program-scratch-count program)
                                                    (position reducer '(nil count any all)) index))
                                             (reducer
                                              (list input-count count (native-program-scratch-count program)
                                                    (ecase reducer (sum 0) (asum 3) (nrm2 4))
                                                    1d0 value wide index))
                                             (t (list output count (native-program-scratch-count program))))))))
                 (check-native-kernel-status status)
                 (case reducer
                   (count (cffi:mem-ref index :size))
                   ((any all) (not (zerop (cffi:mem-ref index :size))))
                   (sum (complex (cffi:mem-aref value foreign 0) (cffi:mem-aref value foreign 1)))
                   (asum (cffi:mem-aref value foreign 0))
                   (nrm2 (if (eq type :c32) (coerce (cffi:mem-ref wide :double) 'single-float)
                             (cffi:mem-ref wide :double)))))
            (keep-native-program-alive program)))))))

(defun run-native-complex-kernel (program type destination inputs offsets output-start count kind reducer)
  ;; TODO: Lisp complex slices copy per call; pin validated layouts or reuse buffers (#74).
  (let ((buffers nil))
    (unwind-protect
         (multiple-value-bind (foreign element) (native-constant-types type)
           (declare (ignore element))
           (flet ((allocate (foreign count)
                    (let ((pointer (cffi:foreign-alloc foreign :count (max 1 count))))
                      (push pointer buffers)
                      pointer)))
             (let* ((pointers
                      (loop for input in inputs for offset in offsets
                            collect
                            (if (vector-view-p input)
                                (cffi:inc-pointer (vector-view-pointer input)
                                                  (* offset (vector-element-size type)))
                                (let ((pointer (allocate foreign (* 2 count))))
                                  (dotimes (i count)
                                    (let ((value (aref input (+ offset i))))
                                      (setf (cffi:mem-aref pointer foreign (* 2 i)) (realpart value)
                                            (cffi:mem-aref pointer foreign (1+ (* 2 i))) (imagpart value))))
                                  pointer))))
                    (output-foreign (if (eq kind :mask) :uint8 foreign))
                    (output (if (and destination (vector-view-p destination))
                                (cffi:inc-pointer (vector-view-pointer destination)
                                                  (* output-start (vector-element-size (vector-type destination))))
                                (if destination
                                    (allocate output-foreign (* (if (eq kind :mask) 1 2) count))
                                    (cffi:null-pointer)))))
               (cffi:with-foreign-object (table :pointer (length inputs))
                 (loop for pointer in pointers for i from 0
                       do (setf (cffi:mem-aref table :pointer i) pointer))
                 (let ((result (call-native-complex-kernel program type table (length inputs)
                                                          output output count kind reducer)))
                   (when (and destination (not (vector-view-p destination)))
                     (dotimes (i count)
                       (setf (aref destination (+ output-start i))
                             (if (eq kind :mask) (cffi:mem-aref output :uint8 i)
                                 (complex (cffi:mem-aref output foreign (* 2 i))
                                          (cffi:mem-aref output foreign (1+ (* 2 i))))))))
                   (if destination destination result))))))
      (mapc #'cffi:foreign-free buffers))))

(defun run-complex-expression (kernel type destination inputs offsets output-start count)
  (let ((tree (complex-kernel-tree kernel))
        (kind (complex-kernel-kind kernel)) (reducer (complex-kernel-reducer kernel)))
    ;; TODO: complex products use typed Lisp; add native product kernels (#148).
    (if (and (native-complex-kernel-p) (not (eq reducer 'prod)))
        (let* ((slot (if (eq type :c32) 0 1)) (programs (complex-kernel-programs kernel))
               (program (or (aref programs slot)
                            (setf (aref programs slot)
                                  (multiple-value-bind (instructions constants scratch-count) (lower-kernel tree)
                                    (make-native-program (kernel-bytes instructions) constants scratch-count type))))))
          (run-native-complex-kernel program type destination inputs offsets output-start count kind reducer))
        (funcall (complex-expression-function kernel type)
                 destination output-start count inputs offsets (make-list (length inputs) :initial-element 0)))))
