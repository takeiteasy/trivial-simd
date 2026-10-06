(in-package #:trivial-simd)

(defun resolve-mixed-slice (vectors starts start end &optional own-strides stride)
  "Like RESOLVE-SLICE for vectors of different element types; returns the count,
per-vector offsets and per-vector strides."
  (let* ((length (vector-length (first vectors)))
         (start (or start 0))
         (end-supplied-p end)
         (end (or end length))
         (strides (resolve-strides own-strides stride (length vectors))))
    (unless (and (integerp start) (integerp end) (<= 0 start end))
      (error "Invalid slice: :START ~S, :END ~S" start end))
    (unless (or end-supplied-p (every (lambda (vector) (= (vector-length vector) length)) vectors))
      (error "All vectors must have the same length without :END"))
    (let ((count (- end start))
          (offsets (mapcar (lambda (own) (or own start)) starts)))
      (loop for vector in vectors for offset in offsets for index from 0
            for stride = (if strides (nth index strides) 1) do
        (unless (span-valid-p offset count stride (vector-length vector))
          (error "Invalid slice of ~D elements at offset ~S, stride ~D, for vector length ~D"
                 count offset stride (vector-length vector))))
      (values count offsets strides))))

(defun whole-extended-slice (destination operands operand-starts start end destination-start
                             numeric-type mask-operand operand-strides stride destination-stride)
  "Resolve a call with no slice or stride keywords whose operands already agree,
or return NIL for EXTENDED-SLICE to resolve and validate."
  (when (and (null start) (null end) (null destination-start)
             (null stride) (null destination-stride)
             (every #'null operand-starts) (every #'null operand-strides))
    (let ((type (or numeric-type (vector-type destination)))
          (length (vector-length destination))
          (offsets (list 0)))
      (loop for operand in operands
            for index from 0
            do (cond ((operand-vector-p operand)
                      (unless (and (eq (vector-type operand)
                                       (if (eql index mask-operand) :u8 type))
                                   (= (vector-length operand) length))
                        (return-from whole-extended-slice nil))
                      (push 0 offsets))
                     ((and (not (eql index mask-operand))
                           (typep operand (second (numeric-type type))))
                      (push nil offsets))
                     (t (return-from whole-extended-slice nil))))
      (values type length (nreverse offsets)))))

(defun extended-slice (destination operands operand-starts start end destination-start
                       &key numeric-type mask-operand operand-strides stride destination-stride)
  "Resolve DESTINATION and OPERANDS; return the type, count, and the offsets and
strides of DESTINATION then every operand, with NIL for scalars."
  (multiple-value-bind (type count offsets)
      (whole-extended-slice destination operands operand-starts start end destination-start
                            numeric-type mask-operand operand-strides stride destination-stride)
    (when type
      (return-from extended-slice (values type count offsets nil))))
  (let* ((type (or numeric-type (vector-type destination)))
         (strided (or stride destination-stride (some #'identity operand-strides)))
         (vectors (list destination))
         (starts (list destination-start))
         (own-strides (and strided (list destination-stride))))
    (loop for operand in operands
          for own in operand-starts
          for index from 0
          for own-stride = (nth index operand-strides)
          do (if (operand-vector-p operand)
                 (progn
                   (unless (eq (vector-type operand) (if (eql index mask-operand) :u8 type))
                     (error "Operand has the wrong element type"))
                   (push operand vectors)
                   (push own starts)
                   (when strided (push own-stride own-strides)))
                 (progn
                   (when (or (eql index mask-operand) own)
                     (error "A mask must be a vector and a scalar has no start offset"))
                   (when own-stride (error "A scalar operand has no stride"))
                   (unless (typep operand (second (numeric-type type)))
                     (error 'type-error :datum operand
                            :expected-type (second (numeric-type type)))))))
    (multiple-value-bind (count vector-offsets vector-strides)
        (resolve-mixed-slice (nreverse vectors) (nreverse starts) start end
                             (nreverse own-strides) stride)
      (let ((remaining-offsets (rest vector-offsets)) (offsets (list (first vector-offsets))))
        (dolist (operand operands)
          (push (when (operand-vector-p operand) (pop remaining-offsets)) offsets))
        (values type count (nreverse offsets)
                (and vector-strides (spread-strides operands vector-strides)))))))

(defun operand-value (operand offset index)
  (if offset (aref operand (+ offset index)) operand))

(defun wrapped-unary (operation type value)
  (if (integer-type-p type)
      (funcall (symbol-function (integer-operation-symbol operation type)) value)
      (ecase operation (:negate (- value)) (:abs (abs value)))))

(defun native-extended-function (operation type)
  (native-bulk-function (format nil "%NATIVE-EXTENDED-~A-" operation) type))

(define-compiler-macro native-extended-function (&whole form operation type)
  (if (stringp operation)
      `(native-bulk-function ,(format nil "%NATIVE-EXTENDED-~A-" operation) ,type)
      form))

(defun check-native-extended-status (status)
  (unless (zerop status)
    (error "Native extended operation failed (status ~D)" status)))

(defun native-extended-unary (operation destination input count d-offset i-offset)
  (let* ((type (vector-type destination))
         (foreign (third (numeric-type type)))
         (opcode (position operation '(:negate :abs :sqrt :reciprocal))))
    (with-native-bulk-output (output destination d-offset foreign count)
      (with-native-bulk-input (source input i-offset foreign count)
        (check-native-extended-status
         (funcall (native-extended-function "UNARY" type)
                  opcode output source count))))))

(defun native-extended-minmax (operation destination left right count d-offset l-offset r-offset)
  (let* ((type (vector-type destination))
         (foreign (third (numeric-type type)))
         (zero (coerce 0 (second (numeric-type type)))))
    (with-native-bulk-output (output destination d-offset foreign count)
      (with-native-bulk-input (a left l-offset foreign count)
        (with-native-bulk-input (b right r-offset foreign count)
          (check-native-extended-status
           (funcall (native-extended-function "MINMAX" type)
                    (if (eq operation :min) 0 1) output a b
                    (if l-offset zero left) (if r-offset zero right) count)))))))

(defun native-extended-clamp (destination input lower upper count d-offset i-offset l-offset u-offset)
  (let* ((type (vector-type destination))
         (foreign (third (numeric-type type)))
         (zero (coerce 0 (second (numeric-type type)))))
    (with-native-bulk-output (output destination d-offset foreign count)
      (with-native-bulk-input (source input i-offset foreign count)
        (with-native-bulk-input (lo lower l-offset foreign count)
          (with-native-bulk-input (hi upper u-offset foreign count)
            (check-native-extended-status
             (funcall (native-extended-function "CLAMP" type)
                      output source lo hi
                      (if l-offset zero lower) (if u-offset zero upper) count))))))))

(defun native-extended-compare (mask operator left right type count m-offset l-offset r-offset)
  (let ((foreign (third (numeric-type type)))
        (zero (coerce 0 (second (numeric-type type)))))
    (with-native-bulk-output (output mask m-offset :uint8 count)
      (with-native-bulk-input (a left l-offset foreign count)
        (with-native-bulk-input (b right r-offset foreign count)
          (check-native-extended-status
           (funcall (native-extended-function "COMPARE" type)
                    (position operator '(:eq :ne :lt :le :gt :ge))
                    output a b (if l-offset zero left) (if r-offset zero right) count)))))))

(defun native-extended-select (destination mask on-true on-false count
                               d-offset m-offset t-offset f-offset)
  (let* ((type (vector-type destination))
         (foreign (third (numeric-type type)))
         (zero (coerce 0 (second (numeric-type type)))))
    (with-native-bulk-output (output destination d-offset foreign count)
      (with-native-bulk-input (bits mask m-offset :uint8 count)
        (with-native-bulk-input (yes on-true t-offset foreign count)
          (with-native-bulk-input (no on-false f-offset foreign count)
            (check-native-extended-status
             (funcall (native-extended-function "SELECT" type)
                      output bits yes no
                      (if t-offset zero on-true) (if f-offset zero on-false) count))))))))

(defun native-extended-reduction (operation mask count offset)
  (with-native-bulk-input (bits mask offset :uint8 count)
    (cffi:with-foreign-object (result :size)
      (check-native-extended-status
       (%native-extended-mask-reduce (position operation '(:count :any :all))
                                     bits count result))
      (let ((value (cffi:mem-ref result :size)))
        (if (eq operation :count) value (not (zerop value)))))))

(defun native-extended-float-convert (destination input count d-offset i-offset)
  (let ((d-type (foreign-type destination)) (i-type (foreign-type input)))
    (with-native-bulk-output (output destination d-offset d-type count)
      (with-native-bulk-input (source input i-offset i-type count)
        (if (eq d-type :float)
            (%native-extended-f64-to-f32 output source count)
            (%native-extended-f32-to-f64 output source count))))))

;; Typed Lisp loops. An operand is a vector read at OFFSET or a scalar; the
;; typed array and value are bound outside the loop so the loop body is typed.
(defmacro with-lisp-operand ((reference operand offset element) &body body)
  (let ((array (gensym "ARRAY")) (value (gensym "VALUE"))
        (base (gensym "BASE")) (vector-p (gensym "VECTOR-P")))
    `(let* ((,vector-p (vectorp ,operand))
            (,array (if ,vector-p ,operand (load-time-value (make-array 0 :element-type ',element))))
            (,value (if ,vector-p (coerce 0 ',element) ,operand))
            (,base (or ,offset 0)))
       (declare (type (simple-array ,element (*)) ,array) (type ,element ,value)
                (type fixnum ,base))
       (flet ((,reference (i)
                (declare (type fixnum i))
                (if ,vector-p (aref ,array (+ ,base i)) ,value)))
         (declare (inline ,reference))
         ,@body))))

(defmacro define-lisp-extended-loops ()
  `(progn
     ,@(loop for (key element) in *numeric-types*
             append
             (append
              `((defun ,(lisp-bulk-symbol "%LISP-MINMAX-" key)
                    (operation destination left right count d-offset l-offset r-offset)
                  (declare (type (simple-array ,element (*)) destination)
                           (type fixnum count d-offset))
                  (with-lisp-operand (left-ref left l-offset ,element)
                    (with-lisp-operand (right-ref right r-offset ,element)
                      (if (eq operation :min)
                          (dotimes (i count)
                            (let ((a (left-ref i)) (b (right-ref i)))
                              (setf (aref destination (+ d-offset i)) (if (< b a) b a))))
                          (dotimes (i count)
                            (let ((a (left-ref i)) (b (right-ref i)))
                              (setf (aref destination (+ d-offset i)) (if (> b a) b a)))))))
                  destination)
                (defun ,(lisp-bulk-symbol "%LISP-CLAMP-" key)
                    (destination input lower upper count d-offset i-offset l-offset u-offset)
                  (declare (type (simple-array ,element (*)) destination)
                           (type fixnum count d-offset))
                  (with-lisp-operand (input-ref input i-offset ,element)
                    (with-lisp-operand (lower-ref lower l-offset ,element)
                      (with-lisp-operand (upper-ref upper u-offset ,element)
                        (dotimes (i count)
                          (let ((x (input-ref i)) (low (lower-ref i)) (high (upper-ref i)))
                            (when (> low high) (error "CLAMP! lower bound exceeds upper bound"))
                            (let ((bounded (if (> low x) low x)))
                              (setf (aref destination (+ d-offset i))
                                    (if (< high bounded) high bounded))))))))
                  destination)
                (defun ,(lisp-bulk-symbol "%LISP-COMPARE-" key)
                    (mask operator left right count m-offset l-offset r-offset)
                  (declare (type (simple-array (unsigned-byte 8) (*)) mask)
                           (type fixnum count m-offset))
                  (with-lisp-operand (left-ref left l-offset ,element)
                    (with-lisp-operand (right-ref right r-offset ,element)
                      (ecase operator
                        ,@(loop for (name function) in '((:eq =) (:ne /=) (:lt <)
                                                         (:le <=) (:gt >) (:ge >=))
                                collect `(,name
                                          (dotimes (i count)
                                            (setf (aref mask (+ m-offset i))
                                                  (if (,function (left-ref i) (right-ref i))
                                                      1 0))))))))
                  mask)
                (defun ,(lisp-bulk-symbol "%LISP-SELECT-" key)
                    (destination mask on-true on-false count d-offset m-offset t-offset f-offset)
                  (declare (type (simple-array ,element (*)) destination)
                           (type (simple-array (unsigned-byte 8) (*)) mask)
                           (type fixnum count d-offset m-offset))
                  (with-lisp-operand (true-ref on-true t-offset ,element)
                    (with-lisp-operand (false-ref on-false f-offset ,element)
                      (dotimes (i count)
                        (setf (aref destination (+ d-offset i))
                              (if (zerop (aref mask (+ m-offset i))) (false-ref i) (true-ref i))))))
                  destination))
              (when (member key '(:f32 :f64))
                `((defun ,(lisp-bulk-symbol "%LISP-UNARY-" key)
                      (operation destination input count d-offset i-offset)
                    (declare (type (simple-array ,element (*)) destination input)
                             (type fixnum count d-offset i-offset))
                    (ecase operation
                      (:negate (dotimes (i count)
                                 (setf (aref destination (+ d-offset i)) (- (aref input (+ i-offset i))))))
                      (:abs (dotimes (i count)
                              (setf (aref destination (+ d-offset i)) (abs (aref input (+ i-offset i))))))
                      (:sqrt (dotimes (i count)
                               (setf (aref destination (+ d-offset i))
                                     (kernel-sqrt (aref input (+ i-offset i))))))
                      (:reciprocal (dotimes (i count)
                                     (setf (aref destination (+ d-offset i))
                                           (/ (coerce 1 ',element) (aref input (+ i-offset i)))))))
                    destination)))))))

(define-lisp-extended-loops)

(defun extended-unary (operation destination input start end destination-start input-start
                       stride destination-stride input-stride)
  (unless (operand-vector-p input) (error "A unary operation needs a vector input"))
  (when (and (eq operation :abs) (complex-type-p (vector-type input)))
    (return-from extended-unary
      (complex-magnitude! destination input start end destination-start input-start
                          stride destination-stride input-stride)))
  (let ((type (vector-type destination)))
    (when (and (integer-type-p type) (member operation '(:sqrt :reciprocal)))
      (error "~A requires float vectors" operation)))
  (multiple-value-bind (type count offsets strides)
      (extended-slice destination (list input) (list input-start) start end destination-start
                      :operand-strides (list input-stride) :stride stride
                      :destination-stride destination-stride)
    (destructuring-bind (d-offset i-offset) offsets
     (with-bulk-staged ((destination d-offset (first strides) :out)
                        (input i-offset (second strides)))
         (count :direct (and (eq *backend* :native) (not (complex-type-p type))))
      (cond ((complex-type-p type)
             (complex-unary operation destination input count d-offset i-offset))
            ((eq *backend* :native)
             (native-extended-unary operation destination input count d-offset i-offset))
            ((and (eq *backend* :sbcl) (not (integer-type-p type)))
             (funcall (if (eq type :f32)
                          #+(and sbcl x86-64) #'%sbcl-extended-unary-f32
                          #-(and sbcl x86-64) (error "SBCL SIMD unavailable")
                          #+(and sbcl x86-64) #'%sbcl-extended-unary-f64
                          #-(and sbcl x86-64) (error "SBCL SIMD unavailable"))
                      operation destination input count d-offset i-offset))
            ((not (integer-type-p type))
             (funcall (lisp-bulk-function "%LISP-UNARY-" destination)
                      operation destination input count d-offset i-offset))
            (t
             (dotimes (i count)
               (let ((value (aref input (+ i-offset i))))
                 (setf (aref destination (+ d-offset i))
                       (ecase operation
                         ((:negate :abs) (wrapped-unary operation type value))
                         (:sqrt (kernel-sqrt value))
                         (:reciprocal (/ (coerce 1 (second (numeric-type type)))
                                         value)))))))))))
  destination)

(defmacro define-extended-unary (name operation)
  `(defun ,name (destination input &key start end destination-start input-start
                                     stride destination-stride input-stride)
     (extended-unary ,operation destination input start end destination-start input-start
                     stride destination-stride input-stride)))

(define-extended-unary negate! :negate)
(define-extended-unary abs! :abs)
(define-extended-unary sqrt! :sqrt)
(define-extended-unary reciprocal! :reciprocal)

(defun kernel-min-left (left right)
  (if (< right left) right left))

(defun kernel-max-left (left right)
  (if (> right left) right left))

(defun extended-minmax (operation destination left right start end d-start l-start r-start
                        stride d-stride l-stride r-stride)
  (when (complex-type-p (vector-type destination))
    (error "Complex vectors have no ordering for MIN!/MAX!"))
  (unless (or (operand-vector-p left) (operand-vector-p right))
    (error "MIN!/MAX! needs a vector operand"))
  (multiple-value-bind (type count offsets strides)
      (extended-slice destination (list left right) (list l-start r-start) start end d-start
                      :operand-strides (list l-stride r-stride) :stride stride
                      :destination-stride d-stride)
    (declare (ignore type))
    (destructuring-bind (d-offset l-offset r-offset) offsets
      (with-bulk-staged ((destination d-offset (first strides) :out)
                         (left l-offset (second strides))
                         (right r-offset (third strides)))
          (count :direct (eq *backend* :native))
        (if (eq *backend* :native)
            (native-extended-minmax operation destination left right count d-offset l-offset r-offset)
            (funcall (lisp-bulk-function "%LISP-MINMAX-" destination)
                     operation destination left right count d-offset l-offset r-offset)))))
  destination)

(defmacro define-extended-minmax (name operation)
  `(defun ,name (destination left right &key start end destination-start left-start right-start
                                          stride destination-stride left-stride right-stride)
     (extended-minmax ,operation destination left right start end
                      destination-start left-start right-start
                      stride destination-stride left-stride right-stride)))

(define-extended-minmax min! :min)
(define-extended-minmax max! :max)

(defun clamp! (destination input lower upper &key start end destination-start
                                             input-start lower-start upper-start
                                             stride destination-stride input-stride
                                             lower-stride upper-stride)
  (when (complex-type-p (vector-type destination))
    (error "Complex vectors have no ordering for CLAMP!"))
  (unless (operand-vector-p input) (error "CLAMP! needs a vector input"))
  (when (and (not (operand-vector-p lower)) (not (operand-vector-p upper)) (> lower upper))
    (error "CLAMP! lower bound exceeds upper bound"))
  (multiple-value-bind (type count offsets strides)
      (extended-slice destination (list input lower upper)
                      (list input-start lower-start upper-start) start end destination-start
                      :operand-strides (list input-stride lower-stride upper-stride)
                      :stride stride :destination-stride destination-stride)
    (declare (ignore type))
    (destructuring-bind (d-offset i-offset l-offset u-offset) offsets
      (with-bulk-staged ((destination d-offset (first strides) :out)
                         (input i-offset (second strides))
                         (lower l-offset (third strides))
                         (upper u-offset (fourth strides)))
          (count :direct (eq *backend* :native))
        (if (eq *backend* :native)
            (native-extended-clamp destination input lower upper count
                                   d-offset i-offset l-offset u-offset)
            (funcall (lisp-bulk-function "%LISP-CLAMP-" destination)
                     destination input lower upper count d-offset i-offset l-offset u-offset)))))
  destination)

(defun integer-limits (type)
  (destructuring-bind (key element foreign bits signed) (numeric-type type)
    (declare (ignore key element foreign))
    (if signed
        (values (- (ash 1 (1- bits))) (1- (ash 1 (1- bits))))
        (values 0 (1- (ash 1 bits))))))

(defun convert-element (value destination-type rounding)
  (cond ((complex-type-p destination-type)
         (let ((real-type (if (eq destination-type :c32) 'single-float 'double-float)))
           (complex (coerce (realpart value) real-type)
                    (coerce (imagpart value) real-type))))
        ((complexp value)
         (if (zerop (imagpart value))
             (convert-element (realpart value) destination-type rounding)
             (error "Complex values with nonzero imaginary part cannot become real")))
        ((integer-type-p destination-type)
         (multiple-value-bind (low high) (integer-limits destination-type)
           (when (and (floatp value) (not (= value value)))
             (error "Cannot convert NaN to an integer"))
           (cond ((< value low) low)
                 ((> value high) high)
                 ((integerp value) value)
                 (t (min high (max low
                                   (ecase rounding
                                     (:nearest-even (round value))
                                     (:truncate (truncate value))
                                     (:floor (floor value))
                                     (:ceiling (ceiling value)))))))))
        (t (coerce value (second (numeric-type destination-type))))))

(defun native-float-conversion-p (destination-type input-type)
  "True when CONVERT! between DESTINATION-TYPE and INPUT-TYPE calls the native library."
  (and (member *backend* '(:native :sbcl)) *native-available-p*
       (member destination-type '(:f32 :f64))
       (member input-type '(:f32 :f64))
       (not (eq destination-type input-type))))

(defun convert! (destination input &key start end destination-start input-start
                                    stride destination-stride input-stride
                                    (rounding :nearest-even) destination-encoding input-encoding)
  (unless (member rounding '(:nearest-even :truncate :floor :ceiling))
    (error "Unknown conversion rounding mode ~S" rounding))
  (let ((destination-type (vector-type destination)))
    (vector-type input)
    (multiple-value-bind (encoding encode)
        (when (or destination-encoding input-encoding)
          (check-float-encodings destination input destination-encoding input-encoding))
      (multiple-value-bind (count offsets strides)
          (resolve-mixed-slice (list destination input)
                               (list destination-start input-start) start end
                               (list destination-stride input-stride) stride)
        (destructuring-bind (d-offset i-offset) offsets
          (with-bulk-staged ((destination d-offset (first strides) :out)
                             (input i-offset (second strides)))
              (count :direct (if encoding
                                 (native-encoded-conversion-p
                                  encode rounding
                                  (extended-encoded-conversion-p destination input destination-encoding input-encoding))
                                 (native-float-conversion-p destination-type (vector-type input))))
            ;; TODO: scalar paths for other type pairs; add packed conversions where safe (#72).
            (cond (encoding
                   (convert-encoded encoding encode rounding destination input count d-offset i-offset
                                    destination-encoding input-encoding))
                  ((native-float-conversion-p destination-type (vector-type input))
                   (native-extended-float-convert destination input count d-offset i-offset))
                  (t
                   (dotimes (i count)
                     (setf (aref destination (+ d-offset i))
                           (convert-element (aref input (+ i-offset i))
                                            destination-type rounding))))))))))
  destination)

(defun compare-values (operator left right)
  (ecase operator
    (:eq (= left right)) (:ne (/= left right))
    (:lt (< left right)) (:le (<= left right))
    (:gt (> left right)) (:ge (>= left right))))

(defun compare! (mask operator left right &key start end mask-start left-start right-start
                                            stride mask-stride left-stride right-stride)
  (unless (eq (vector-type mask) :u8) (error "Comparison mask must be an unsigned-byte-8 vector"))
  (unless (member operator '(:eq :ne :lt :le :gt :ge))
    (error "Unknown comparison operator ~S" operator))
  (let ((type (cond ((operand-vector-p left) (vector-type left))
                    ((operand-vector-p right) (vector-type right))
                    (t (error "COMPARE! needs a vector operand")))))
    (when (and (complex-type-p type) (not (member operator '(:eq :ne))))
      (error "Complex vectors support only equality comparisons"))
    (multiple-value-bind (resolved count offsets strides)
        (extended-slice mask (list left right) (list left-start right-start)
                        start end mask-start :numeric-type type
                        :operand-strides (list left-stride right-stride)
                        :stride stride :destination-stride mask-stride)
      (declare (ignore resolved))
      (destructuring-bind (m-offset l-offset r-offset) offsets
        (with-bulk-staged ((mask m-offset (first strides) :out)
                           (left l-offset (second strides))
                           (right r-offset (third strides)))
            (count :direct (and (eq *backend* :native) (not (complex-type-p type))))
          (cond ((complex-type-p type)
                 (dotimes (i count)
                   (setf (aref mask (+ m-offset i))
                         (if (compare-values operator (operand-value left l-offset i)
                                             (operand-value right r-offset i)) 1 0))))
                ((eq *backend* :native)
                 (native-extended-compare mask operator left right type count
                                          m-offset l-offset r-offset))
                (t
                 (funcall (lisp-bulk-function "%LISP-COMPARE-" (if (vectorp left) left right))
                          mask operator left right count m-offset l-offset r-offset)))))))
  mask)

(defun select! (destination mask on-true on-false &key start end destination-start
                                                mask-start true-start false-start
                                                stride destination-stride mask-stride
                                                true-stride false-stride)
  (multiple-value-bind (type count offsets strides)
      (extended-slice destination (list mask on-true on-false)
                      (list mask-start true-start false-start) start end destination-start
                      :mask-operand 0
                      :operand-strides (list mask-stride true-stride false-stride)
                      :stride stride :destination-stride destination-stride)
    (destructuring-bind (d-offset m-offset t-offset f-offset) offsets
      (with-bulk-staged ((destination d-offset (first strides) :out)
                         (mask m-offset (second strides))
                         (on-true t-offset (third strides))
                         (on-false f-offset (fourth strides)))
          (count :direct (and (eq *backend* :native) (not (complex-type-p type))))
        (cond ((complex-type-p type)
               (dotimes (i count)
                 (setf (aref destination (+ d-offset i))
                       (if (zerop (aref mask (+ m-offset i)))
                           (operand-value on-false f-offset i)
                           (operand-value on-true t-offset i)))))
              ((eq *backend* :native)
               (native-extended-select destination mask on-true on-false count
                                       d-offset m-offset t-offset f-offset))
              (t
               (funcall (lisp-bulk-function "%LISP-SELECT-" destination)
                        destination mask on-true on-false count
                        d-offset m-offset t-offset f-offset))))))
  destination)

(defun mask-reduction (operation mask start end mask-start stride mask-stride)
  (unless (eq (vector-type mask) :u8) (error "Mask must be an unsigned-byte-8 vector"))
  (multiple-value-bind (length offsets strides)
      (resolve-mixed-slice (list mask) (list mask-start) start end (list mask-stride) stride)
    (let ((offset (first offsets)))
      (with-staged ((mask offset (first strides)))
          (length :direct (eq *backend* :native)
                  :combine (ecase operation
                             (:count #'count-combiner) (:any #'any-combiner)
                             (:all #'all-combiner)))
        (if (eq *backend* :native)
            (native-extended-reduction operation mask length offset)
            (ecase operation
              (:count (loop for i below length count (not (zerop (aref mask (+ offset i))))))
              (:any (loop for i below length thereis (not (zerop (aref mask (+ offset i))))))
              (:all (loop for i below length always (not (zerop (aref mask (+ offset i))))))))))))

(defun count (mask &key start end mask-start stride mask-stride)
  (mask-reduction :count mask start end mask-start stride mask-stride))

(defun any (mask &key start end mask-start stride mask-stride)
  (mask-reduction :any mask start end mask-start stride mask-stride))

(defun all (mask &key start end mask-start stride mask-stride)
  (mask-reduction :all mask start end mask-start stride mask-stride))

(defvar *kernel-operators*)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *mask-kernel-comparisons* '(= /= < <= > >=))
  (defparameter *kernel-reducers* '(sum asum nrm2 minimum maximum argmin argmax prod))

  (defun mask-kernel-expression-p (expression)
    (and (consp expression)
         (or (member (first expression)
                     (append *mask-kernel-comparisons* '(select count any all)))
             (some #'mask-kernel-expression-p (rest expression)))))

  (defun mask-kernel-kind (expression arguments)
    (cond ((member expression arguments) :numeric)
          ((or (realp expression) (kernel-scalar-p expression)) :numeric)
          ((not (consp expression)) (error "Unsupported kernel expression ~S" expression))
          ((member (first expression) *mask-kernel-comparisons*)
           (unless (and (= 2 (length (rest expression)))
                        (every (lambda (operand)
                                 (eq :numeric (mask-kernel-kind operand arguments)))
                               (rest expression)))
             (error "A comparison needs two numeric operands"))
           :mask)
          ((eq (first expression) 'select)
           (unless (and (= 3 (length (rest expression)))
                        (eq :mask (mask-kernel-kind (second expression) arguments))
                        (every (lambda (operand)
                                 (eq :numeric (mask-kernel-kind operand arguments)))
                               (cddr expression)))
             (error "SELECT needs a mask and two numeric operands"))
           :numeric)
          ((member (first expression) (list* 'count 'any 'all *kernel-reducers*))
           (error "Kernel reductions must be top-level"))
          ((member (first expression) '(+ - * / min max sqrt abs exp sin cos fma))
           (let ((arity (length (rest expression))))
             (unless (case (first expression)
                       ((sqrt abs exp sin cos) (= arity 1))
                       (fma (= arity 3))
                       (otherwise (plusp arity)))
               (error "Wrong number of kernel operands in ~S" expression))
             (unless (every (lambda (operand)
                              (eq :numeric (mask-kernel-kind operand arguments)))
                            (rest expression))
               (error "Arithmetic needs numeric operands")))
           :numeric)
          (t (error "Unsupported kernel expression ~S" expression))))

  (defun mask-kernel-scalar-form (expression arguments values type)
    (cond ((member expression arguments)
           (nth (position expression arguments) values))
          ((kernel-scalar-p expression) (kernel-constant-form expression type))
          ((realp expression)
           (if (integer-type-p type)
               (progn
                 (unless (typep expression (second (numeric-type type)))
                   (error "Constant ~S does not fit ~S" expression type))
                 expression)
               (coerce expression (second (numeric-type type)))))
          (t
           (let* ((operator (first expression))
                  (operands (mapcar (lambda (part)
                                      (mask-kernel-scalar-form part arguments values type))
                                    (rest expression)))
                  (integerp (integer-type-p type)))
             (cond ((member operator *mask-kernel-comparisons*)
                    (cons operator operands))
                   ((eq operator 'select)
                    `(if ,(first operands) ,(second operands) ,(third operands)))
                   ((member operator '(sqrt abs exp sin cos))
                    (when (or (and integerp (member operator '(sqrt exp sin cos)))
                              (and (complex-type-p type) (member operator '(abs exp sin cos))))
                      (error "~A requires real float vectors" operator))
                    `(,(if (eq operator 'sqrt) 'kernel-sqrt
                           (if integerp (integer-operation-symbol :abs type) operator))
                      ,(first operands)))
                   ((eq operator 'fma)
                    (when (or integerp (complex-type-p type)) (error "FMA requires real float vectors"))
                    `(fma ,@operands))
                   (t
                    (let* ((operation (case operator
                                        (+ :add) (- :subtract) (* :multiply) (/ :divide)
                                        (min :min) (max :max)))
                           (binary (if integerp (integer-operation-symbol operation type)
                                       (case operator
                                         (min 'kernel-min-left) (max 'kernel-max-left)
                                         (otherwise operator)))))
                      (if (and (= 1 (length operands)) (member operator '(- /)))
                          (if (eq operator '-)
                              `(,(if integerp (integer-operation-symbol :negate type) '-)
                                ,(first operands))
                              `(,binary ,(if integerp 1 (coerce 1 (second (numeric-type type))))
                                        ,(first operands)))
                          (reduce (lambda (left right) `(,binary ,left ,right))
                                  (rest operands) :initial-value (first operands))))))))))

  (defun mask-kernel-tree (expression arguments)
    (cond ((member expression arguments) (list :argument (position expression arguments)))
          ((or (realp expression) (kernel-scalar-p expression)) (list :constant expression))
          (t
           (let* ((operator (first expression))
                  (operands (mapcar (lambda (part) (mask-kernel-tree part arguments))
                                    (rest expression)))
                  (kind (cdr (assoc operator *kernel-operators*))))
             (cond ((member operator *mask-kernel-comparisons*)
                    (list :operation
                          (cdr (assoc operator '((= . :eq) (/= . :ne) (< . :lt)
                                                (<= . :le) (> . :gt) (>= . :ge))))
                          (first operands) (second operands)))
                   ((eq operator 'select) (cons :select operands))
                   ((member kind '(:sqrt :abs :exp :sin :cos)) (list :unary kind (first operands)))
                   ((eq kind :fma) (cons :fma operands))
                   ((and (eq kind :subtract) (= (length operands) 1))
                    (list :negate (first operands)))
                   ((and (eq kind :divide) (= (length operands) 1))
                    (list :operation :divide (list :constant 1) (first operands)))
                   ((= (length operands) 1) (first operands))
                   (t (reduce (lambda (left right) (list :operation kind left right))
                              (rest operands) :initial-value (first operands))))))))

  (defun native-mask-kernel-form (program foreign destination arguments
                                   d-offset input-offsets count reduction)
    (let ((pointers (loop for nil in arguments collect (gensym "INPUT")))
          (table (gensym "TABLE")) (output (gensym "MASK"))
          (result (gensym "RESULT")))
      `(with-native-vectors (,foreign (,@(mapcar #'list pointers arguments input-offsets))
                            :count ,count)
         (cffi:with-foreign-object (,table :pointer ,(max 1 (length arguments)))
           ,@(loop for pointer in pointers for i from 0
                   collect `(setf (cffi:mem-aref ,table :pointer ,i) ,pointer))
           ,(if destination
                `(with-native-bulk-output (,output ,destination ,d-offset :uint8 ,count)
                   (call-native-mask-kernel ,program ,foreign ,table ,(length arguments)
                                            ,output ,count 0 (cffi:null-pointer))
                   ,destination)
                `(cffi:with-foreign-object (,result :size)
                   (call-native-mask-kernel ,program ,foreign ,table ,(length arguments)
                                            (cffi:null-pointer) ,count
                                            ,(position reduction '(nil count any all)) ,result)
                   ,(if (eq reduction 'count)
                        `(cffi:mem-ref ,result :size)
                        `(not (zerop (cffi:mem-ref ,result :size))))))))))

  (defun mask-kernel-dispatch-form (scalar programs program slot bytes constants scratch-count
                                    key foreign kind reduction destination arguments
                                    d-offset input-offsets count)
    (if (member reduction *kernel-reducers*)
        scalar
        `(if (eq *backend* :native)
             (let ((,program (or (aref ,programs ,slot)
                                 (setf (aref ,programs ,slot)
                                       (make-native-program ',bytes ',constants
                                                            ,scratch-count ,key)))))
               ,(if (eq kind :mask)
                    (native-mask-kernel-form program foreign destination arguments
                                             d-offset input-offsets count reduction)
                    (native-kernel-form program foreign destination arguments
                                        d-offset input-offsets count)))
             ,scalar)))

  (defun argument-index-result (reduction result form)
    "Turn the slice-relative index returned by FORM into START + index for ARGMIN and ARGMAX."
    (if (member reduction '(argmin argmax))
        `(let ((,result ,form))
           (and ,result (+ (or start 0) ,result)))
        form))

  (defun unshifted-kernel-inputs-forms (destination d-offset arguments offsets count)
    "Forms replacing each of ARGUMENTS whose slice is shifted against DESTINATION's
by a copy; see UNSHIFTED-KERNEL-INPUT."
    (when destination
      (loop for argument in arguments for offset in offsets
            collect `(when (or (eq ,argument ,destination)
                               (and (vector-view-p ,argument) (vector-view-p ,destination)))
                       (multiple-value-setq (,argument ,offset)
                         (unshifted-kernel-input ,destination ,d-offset ,argument ,offset ,count))))))

  (defun mask-kernel-expansion (name arguments expression)
    ;; TODO: SBCL mask kernels still use scalar loops; add packed expressions (#72).
    (unless arguments (error "A mask kernel needs input vectors"))
    (let* ((outer (and (consp expression) (first expression)))
           (reduction (and (member outer (list* 'count 'any 'all *kernel-reducers*)) outer))
           (body (if reduction (second expression) expression))
           (kind (mask-kernel-kind body arguments))
           (destination (unless reduction (gensym "DESTINATION")))
           (type (gensym "TYPE")) (count (gensym "COUNT"))
           (offsets (gensym "OFFSETS")) (d-offset (gensym "D-OFFSET"))
           (input-offsets (loop for nil in arguments collect (gensym "OFFSET")))
           (starts (mapcar (lambda (argument)
                             (intern (format nil "~A-START" argument))) arguments))
           (index (gensym "INDEX")) (result (gensym "RESULT"))
           (values (loop for nil in arguments collect (gensym "VALUE")))
           (programs (gensym "PROGRAMS")) (program (gensym "PROGRAM"))
           #+ecl (runners (gensym "RUNNERS"))
           (complex-kernel (gensym "COMPLEX-KERNEL"))
           (lowered (multiple-value-list (lower-kernel (mask-kernel-tree body arguments))))
           (bytes (kernel-bytes (first lowered)))
           (constants (second lowered)) (scratch-count (third lowered)))
      (when (and reduction (/= 2 (length expression)))
        (error "A reduction needs one expression"))
      (when (and reduction
                 (not (eq kind (if (member reduction '(count any all)) :mask :numeric))))
        (error "Kernel result has the wrong kind"))
      `(let ((,programs (make-array ,(length *numeric-types*) :initial-element nil))
             #+ecl (,runners (make-array ,(length *numeric-types*) :initial-element nil))
             (,complex-kernel (make-complex-kernel ',arguments ',(mask-kernel-tree body arguments)
                                                  ',kind ',reduction)))
         (defun ,name (,@(when destination (list destination)) ,@arguments
                     &key start end ,@(when destination '(destination-start)) ,@starts)
         ,(argument-index-result reduction result
           `(let ((,type (vector-type ,(first arguments))))
           ,@(loop for argument in (rest arguments)
                   collect `(unless (eq ,type (vector-type ,argument))
                              (error "Kernel arguments must have the same element type")))
           ,@(when destination
               `((unless (eq (vector-type ,destination)
                             ,(if (eq kind :mask) :u8 type))
                   (error "Kernel destination has the wrong element type"))))
           (when (complex-type-p ,type)
             (validate-complex-kernel (complex-kernel-tree ,complex-kernel) ',reduction))
           (multiple-value-bind (,count ,offsets)
               (resolve-mixed-slice
                (list ,@(when destination (list destination)) ,@arguments)
                (list ,@(when destination '(destination-start)) ,@starts) start end)
             (destructuring-bind (,@(when destination (list d-offset)) ,@input-offsets)
                 ,offsets
              ,@(unshifted-kernel-inputs-forms destination d-offset arguments input-offsets count)
              ,(staged-kernel-form
                name reduction type destination d-offset arguments input-offsets
                (if destination (cons 'destination-start starts) starts) count
                `(if (complex-type-p ,type)
                     (and (native-complex-kernel-p) ,(not (eq reduction 'prod)))
                     ,(if (member reduction *kernel-reducers*) nil '(eq *backend* :native)))
               `(if (complex-type-p ,type)
                    (run-complex-expression ,complex-kernel ,type ,destination
                                            (list ,@arguments) (list ,@input-offsets) ,(if destination d-offset 0)
                                            ,count)
                    (ecase ,type
                 ,@(loop for (key element foreign) in *numeric-types*
                         for slot from 0
                         for form = (handler-case
                                (let* ((scalar (mask-kernel-scalar-form body arguments values key))
                                       (bindings (loop for value in values for argument in arguments
                                                       for offset in input-offsets
                                                       collect `(,value (aref ,argument (+ ,offset ,index)))))
                                       (test `(let ,bindings ,scalar)))
                                  (mask-kernel-dispatch-form
                                   (case reduction
                                    (count `(let ((,result 0))
                                              (dotimes (,index ,count ,result)
                                                (when ,test (incf ,result)))))
                                    (any `(loop for ,index below ,count thereis ,test))
                                    (all `(loop for ,index below ,count always ,test))
                                    ((sum asum nrm2 minimum maximum argmin argmax prod)
                                     (reduction-loop-form reduction key count index test))
                                    (otherwise
                                     `(progn
                                        (dotimes (,index ,count)
                                          (setf (aref ,destination (+ ,d-offset ,index))
                                                ,(if (eq kind :mask)
                                                     `(if ,test 1 0) test)))
                                        ,destination)))
                                   programs program slot bytes constants scratch-count
                                   key foreign kind reduction destination arguments
                                   d-offset input-offsets count))
                              (error (condition)
                                `(error ,(princ-to-string condition))))
                         collect
                         `(,key
                           ,#+ecl
                           (let ((parameters (append (list programs)
                                                     (when destination (list destination d-offset))
                                                     arguments input-offsets (list count))))
                             `(funcall (or (aref ,runners ,slot)
                                           (setf (aref ,runners ,slot)
                                                 (eval '(lambda ,parameters ,form))))
                                       ,@parameters))
                           #-ecl form))))
                (lambda (position)
                  `(kernel-tree-value (complex-kernel-tree ,complex-kernel) ,type (list ,@arguments)
                                      (list ,@input-offsets) ,position t))))))))))))
