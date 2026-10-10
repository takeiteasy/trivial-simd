(in-package #:trivial-simd)

(macrolet ((define-row-calls ()
             `(progn
                ,@(loop for suffix in '(f32 f64)
                        collect
                        `(define-native (,(format nil "ts_kernel_sum_rows_~(~A~)" suffix)
                                         ,(intern (format nil "%NATIVE-KERNEL-SUM-ROWS-~A" suffix))) :int
                           (code :pointer) (code-length :size) (constants :pointer)
                           (inputs :pointer) (input-count :size) (strides :pointer)
                           (output :pointer) (rows :size) (row-length :size)
                           (scratch-count :size))))))
  (define-row-calls))

(defvar *native-kernel-rows-available-p*
  (and *native-available-p*
       (every (lambda (symbol) (ignore-errors (cffi:foreign-symbol-pointer symbol)))
              '("ts_kernel_sum_rows_f32" "ts_kernel_sum_rows_f64"))))


(macrolet ((define-reducer-rows ()
             `(progn
                ,@(loop for suffix in '(f32 f64)
                        collect
                        `(define-native (,(format nil "ts_kernel_reduction_rows_~(~A~)" suffix)
                                         ,(intern (format nil "%NATIVE-REDUCTION-ROWS-~A" suffix))) :int
                           (code :pointer) (code-length :size) (constants :pointer)
                           (inputs :pointer) (input-count :size) (strides :pointer)
                           (output :pointer) (rows :size) (row-length :size)
                           (scratch-count :size) (reducer :uint) (indices :uint))))))
  (define-reducer-rows))

(defvar *native-row-reducers-p*
  (and *native-available-p*
       (null (missing-native-symbols '("ts_kernel_reduction_rows_f32" "ts_kernel_reduction_rows_f64")))))

(declaim (notinline call-native-reduction-rows))
(defun call-native-reduction-rows (program type inputs input-count strides output rows row-length reducer)
  (with-kernel-program (program type)
    (unwind-protect
         (check-native-kernel-status
          (funcall (ecase type (:f32 #'%native-reduction-rows-f32) (:f64 #'%native-reduction-rows-f64))
                   (native-program-code program) (native-program-code-length program)
                   (native-program-constants program) inputs input-count strides output rows row-length
                   (native-program-scratch-count program)
                   (ecase reducer ((minimum argmin) 1) ((maximum argmax) 2) (prod 7))
                   (if (member reducer '(argmin argmax)) 1 0)))
      (keep-native-program-alive program))))

(declaim (notinline call-native-kernel-rows))
(defun call-native-kernel-rows (program type inputs input-count strides output rows row-length)
  (with-kernel-program (program type)
    (unwind-protect
         (check-native-kernel-status
          (funcall (ecase type
                     (:f32 #'%native-kernel-sum-rows-f32)
                     (:f64 #'%native-kernel-sum-rows-f64))
                   (native-program-code program) (native-program-code-length program)
                   (native-program-constants program)
                   inputs input-count strides output rows row-length
                   (native-program-scratch-count program)))
      (keep-native-program-alive program))))

(defun call-with-row-pointers (vectors function &optional pointers)
  (if vectors
      (with-vector-pointer (pointer (first vectors))
        (call-with-row-pointers (rest vectors) function (cons pointer pointers)))
      (funcall function (nreverse pointers))))

(defun check-row-overlap (inputs destination spans destination-start rows bytes &optional pointers (output-bytes bytes))
  (when (plusp rows)
    (let* ((output-pointer (and pointers (car (last pointers))))
           (output-low (if output-pointer
                           (+ (cffi:pointer-address output-pointer) (* output-bytes destination-start))
                           destination-start))
           (output-high (+ output-low (* (if pointers output-bytes 1) rows))))
      (loop for input in inputs for (low high) in spans
            for index from 0
            when (and (< low high)
                      (if pointers (and output-pointer (nth index pointers)) (eq input destination)))
              do (let* ((base (if pointers (cffi:pointer-address (nth index pointers)) 0))
                        (scale (if pointers bytes 1))
                        (input-low (+ base (* scale low)))
                        (input-high (+ base (* scale high))))
                   (when (and (< output-low input-high) (< input-low output-high))
                     (error "Batch destination overlaps an input span")))))))

(defun resolve-kernel-rows (inputs destination rows row-length starts strides destination-start &optional (reducer 'sum))
  (let* ((type (vector-type (first inputs)))
         (bytes (vector-element-size type))
         (limit (min most-positive-fixnum
                     (floor (1- (ash 1 (1- (* 8 (cffi:foreign-type-size :pointer))))) bytes))))
    (when (and (complex-type-p type) (member reducer '(minimum maximum argmin argmax)))
      (error "Complex values have no ordering"))
    (when (and (plusp rows) (zerop row-length) (member reducer '(minimum maximum argmin argmax)))
      (error "Empty rows have no extrema"))
    (unless (and (typep rows `(integer 0 ,limit))
                 (typep row-length `(integer 0 ,limit))
                 (typep destination-start `(integer 0 ,limit)))
      (error "Invalid batch dimensions or destination start"))
    (unless (eq (if (member reducer '(argmin argmax)) :s64 type) (vector-type destination)) (error "Batch destination has the wrong type"))
    (unless (<= (+ destination-start rows) (vector-length destination))
      (error "Batch destination is too short"))
    (let ((spans
            (loop for input in inputs for start in starts for stride in strides
                  collect
                  (progn
                    (unless (eq type (vector-type input)) (error "Batch inputs have different types"))
                    (unless (and (typep start `(integer 0 ,limit))
                                 (typep stride '(signed-byte 64)))
                      (error "Invalid batch input start or row stride"))
                    (let* ((shift (if (and (plusp rows) (plusp row-length)) (* (1- rows) stride) 0))
                           (low (+ start (min 0 shift)))
                           (high (+ start (max 0 shift) (if (plusp rows) row-length 0))))
                      (unless (and (<= 0 low high (vector-length input)) (<= high limit))
                        (error "Batch input span is out of bounds"))
                      (list low high))))))
      (check-row-overlap inputs destination spans destination-start rows bytes nil (vector-element-size (vector-type destination)))
      (values type spans))))

(defun native-kernel-rows (program type inputs destination starts strides spans
                          destination-start rows row-length pointers &optional (reducer 'sum))
  (let ((foreign (third (numeric-type type)))
        (output-foreign (third (numeric-type (vector-type destination)))) (copies nil))
    (unwind-protect
         (labels ((copy-span (vector low count)
                    (let ((pointer (cffi:foreign-alloc foreign :count (max 1 count))))
                      (push pointer copies)
                      (copy-to-foreign vector pointer foreign low count)
                      pointer)))
           (let* ((input-pointers
                    (loop for input in inputs for start in starts
                          for (low high) in spans for pointer in pointers
                          collect
                          (if (and (eq *native-array-access* :copy) (not (vector-view-p input)))
                              (element-pointer (copy-span input low (- high low)) foreign (- start low))
                              (element-pointer pointer foreign start))))
                  (copy-output (and (eq *native-array-access* :copy) (not (vector-view-p destination))))
                  (output (if copy-output
                              (let ((pointer (cffi:foreign-alloc output-foreign :count rows)))
                                (push pointer copies)
                                pointer)
                              (element-pointer (car (last pointers)) output-foreign destination-start))))
             (cffi:with-foreign-objects ((table :pointer (length inputs))
                                         (row-strides :int64 (length inputs)))
               (loop for pointer in input-pointers for stride in strides for i from 0
                     do (setf (cffi:mem-aref table :pointer i) pointer
                              (cffi:mem-aref row-strides :int64 i) stride))
               (if (eq reducer 'sum)
                   (call-native-kernel-rows program type table (length inputs) row-strides output rows row-length)
                   (call-native-reduction-rows program type table (length inputs) row-strides output rows row-length reducer)))
             (when copy-output
               (copy-from-foreign output destination output-foreign destination-start rows))))
      (dolist (pointer copies) (cffi:foreign-free pointer)))))

;; Compiled once with the system, including for ECL kernels defined through EVAL.
(defun run-kernel-rows (scalar inputs start-keys destination rows row-length starts strides
                       destination-start programs bytes constants scratch-count native-p &optional (reducer 'sum))
  (multiple-value-bind (type spans)
      (resolve-kernel-rows inputs destination rows row-length starts strides destination-start reducer)
    (labels ((run (pointers)
               (when pointers
                 (check-row-overlap inputs destination spans destination-start rows
                                    (vector-element-size type) pointers (vector-element-size (vector-type destination))))
               (cond
                 ((zerop rows))
                 ((zerop row-length)
                  (let ((zero (coerce (if (eq reducer 'prod) 1 0) (second (numeric-type type)))))
                    (dotimes (row rows)
                      (setf (vector-ref destination (+ destination-start row)) zero))))
                 ((and native-p (eq *backend* :native) (member type '(:f32 :f64))
                       (if (eq reducer 'sum) *native-kernel-rows-available-p* *native-row-reducers-p*))
                  (let* ((slot (if (eq type :f32) 0 1))
                         (program (or (aref programs slot)
                                      (setf (aref programs slot)
                                            (make-native-program bytes constants scratch-count type)))))
                    (native-kernel-rows program type inputs destination starts strides spans
                                        destination-start rows row-length pointers reducer)))
                 ;; TODO: integer/complex setup scales with rows; add native batches (#62).
                 (t
                  (dotimes (row rows)
                    (setf (vector-ref destination (+ destination-start row))
                          (apply scalar
                                 (append inputs (list :end row-length)
                                         (loop for key in start-keys for start in starts for stride in strides
                                               append (list key (+ start (* row stride))))))))))
               destination))
      #+(or sbcl ccl ecl)
      (call-with-row-pointers (append inputs (list destination)) #'run)
      #-(or sbcl ccl ecl)
      (run (mapcar (lambda (vector) (when (vector-view-p vector) (vector-view-pointer vector)))
                   (append inputs (list destination)))))))

(defmacro define-basic-kernel (name (&rest arguments) expression)
  "Define an elementwise or reduction kernel. Reduction kernels also accept
ROWS, ROW-LENGTH, DESTINATION, DESTINATION-START and per-input ROW-STRIDE."
  (when (some #'consp arguments)
    (return-from define-basic-kernel (declared-kernel-expansion name arguments expression)))
  (let* ((batch-p (and (consp expression) (member (first expression) '(sum minimum maximum argmin argmax prod))))
         (scalar-name (if batch-p (gensym "SCALAR-KERNEL") name))
         (expansion (macroexpand-1 `(define-single-row-kernel ,scalar-name ,arguments ,expression))))
    (unless batch-p
      (return-from define-basic-kernel expansion))
    (let* ((scalar (gensym "SCALAR")) (programs (gensym "PROGRAMS"))
           (start-names (mapcar (lambda (arg) (intern (format nil "~A-START" arg))) arguments))
           (stride-names (mapcar (lambda (arg) (intern (format nil "~A-ROW-STRIDE" arg))) arguments))
           (supplied (loop repeat (+ 3 (length arguments)) collect (gensym "SUPPLIED")))
           (native-p (not (mask-kernel-expression-p expression)))
           (lowered (when native-p (multiple-value-list
                                   (lower-kernel (parse-kernel-expression (second expression) arguments)))))
           (bytes (when native-p (kernel-bytes (first lowered)))))
      `(let ((,scalar (progn ,expansion #',scalar-name))
             (,programs (make-array 2 :initial-element nil)))
         (defun ,name (,@arguments &key (start nil start-p) (end nil end-p) ,@start-names
                                     (rows nil rows-p)
                                     (row-length nil ,(first supplied))
                                     (destination nil ,(second supplied))
                                     (destination-start 0 ,(third supplied))
                                     ,@(loop for stride in stride-names for flag in (cdddr supplied)
                                             collect `(,stride row-length ,flag)))
           "Return the reduction, or write one logical result per row into DESTINATION."
           (if rows-p
               (progn
                 (when (or start-p end-p) (error "Batch calls use :ROW-LENGTH instead of :START/:END"))
                 (unless (and ,(first supplied) ,(second supplied))
                   (error "Batch calls require :ROW-LENGTH and :DESTINATION"))
                 (run-kernel-rows ,scalar (list ,@arguments)
                                  ',(mapcar (lambda (symbol) (intern (symbol-name symbol) :keyword)) start-names)
                                  destination rows row-length
                                  (list ,@(mapcar (lambda (start) `(or ,start 0)) start-names))
                                  (list ,@stride-names)
                                  destination-start ,programs ',bytes ',(second lowered) ,(or (third lowered) 0)
                                  ,native-p ',(first expression)))
               (progn
                 (when (or ,@supplied)
                   (error "Batch keywords require :ROWS"))
                 (funcall ,scalar ,@arguments :start start :end end
                          ,@(loop for symbol in start-names
                                  append (list (intern (symbol-name symbol) :keyword) symbol))))))))))
