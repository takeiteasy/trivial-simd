(in-package #:trivial-simd)

(defparameter *nd-operations*
  '(:add :subtract :multiply :divide :negate :abs :sqrt :reciprocal
    :min :max :clamp :compare :select :convert :log :tanh :sigmoid :exp :sin :cos :silu :gelu))

(defparameter *nd-types* (mapcar #'first *numeric-types*))

(defun nd-integer (value &optional (minimum (max most-negative-fixnum (- (ash 1 63)))))
  (unless (and (integerp value) (<= minimum value (min most-positive-fixnum (1- (ash 1 63)))))
    (error "N-D layout integer is out of range: ~S" value))
  value)

(defun nd-shape (shape)
  (unless (typep shape '(or list vector)) (error "Shape must be a list or vector"))
  (map nil (lambda (dimension) (nd-integer dimension 0)) shape)
  (if (vectorp shape) shape (coerce shape 'vector)))

(defun nd-contiguous-strides (shape)
  (let ((strides (make-array (length shape))) (step 1))
    (loop for axis downfrom (1- (length shape)) to 0 do
      (setf (aref strides axis) (nd-integer step 0))
      (setf step (* step (aref shape axis))))
    strides))

(defun nd-layout (shape strides start storage empty)
  (let* ((strides (if strides
                      (progn
                        (unless (and (typep strides '(or list vector))
                                     (= (length strides) (length shape)))
                          (error "Stride rank must match shape"))
                        (map nil #'nd-integer strides)
                        (if (vectorp strides) strides (coerce strides 'vector)))
                      (nd-contiguous-strides shape)))
         (start (nd-integer start 0))
         (low start) (high start)
         (bytes (vector-element-size (vector-type storage))))
    (loop for dimension across shape for stride across strides do
      (nd-integer (* bytes stride))
      (let ((extent (nd-integer (* (max 0 (1- dimension)) stride))))
        (incf low (min 0 extent))
        (incf high (max 0 extent))))
    (nd-integer (* bytes low))
    (nd-integer (* bytes high))
    (unless (or empty (<= 0 low high (1- (vector-length storage))))
      (error "N-D layout reaches outside storage"))
    (values strides start low high)))

(defun nd-walk (shape layouts starts function)
  (unless (find 0 shape)
    (let* ((rank (length shape)) (indices (make-array rank :initial-element 0))
           (offsets (copy-seq starts)))
      (loop
        (funcall function offsets)
        (loop for axis downfrom (1- rank) to 0 do
          (incf (aref indices axis))
          (if (< (aref indices axis) (aref shape axis))
              (progn
                (loop for strides across layouts for index from 0 do
                  (incf (aref offsets index) (aref strides axis)))
                (return))
              (progn
                (setf (aref indices axis) 0)
                (loop for strides across layouts for index from 0 do
                  (decf (aref offsets index) (* (1- (aref shape axis)) (aref strides axis))))))
              finally (return-from nd-walk nil))))))

(defun nd-unique-output-p (shape strides start)
  (let ((axes (sort (loop for dimension across shape for stride across strides
                         when (> dimension 1) collect (cons (abs stride) dimension))
                    #'< :key #'car))
        (extent 0))
    (when (loop for (stride . dimension) in axes always (> stride extent)
               do (incf extent (* stride (1- dimension))))
      (return-from nd-unique-output-p t)))
  ;; TODO: O(size) workspace; extend arithmetic uniqueness proofs (#63).
  (let ((seen (make-hash-table :test #'eql)))
    (nd-walk shape (vector strides) (vector start)
             (lambda (offsets)
               (let ((offset (aref offsets 0)))
                 (when (gethash offset seen) (return-from nd-unique-output-p nil))
                 (setf (gethash offset seen) t))))
    t))

(defun nd-span (storage low high)
  (let ((bytes (vector-element-size (vector-type storage)))
        (base (if (vector-view-p storage)
                  (cffi:pointer-address (vector-view-pointer storage)) 0)))
    (values (+ base (* bytes low)) (+ base (* bytes (1+ high))))))

(defun nd-alias-p (destination input shape d-strides i-strides d-start i-start d-low d-high i-low i-high)
  (let ((foreign-d (vector-view-p destination)) (foreign-i (vector-view-p input)))
    ;; TODO: mixed storage snapshots; compare ranges while arrays are pinned (#64).
    (when (not (eq foreign-d foreign-i)) (return-from nd-alias-p t))
    (unless (or (eq destination input) (and foreign-d foreign-i))
      (return-from nd-alias-p nil))
    (multiple-value-bind (d-base d-end) (nd-span destination d-low d-high)
      (multiple-value-bind (i-base i-end) (nd-span input i-low i-high)
        (and (< d-base i-end) (< i-base d-end)
             (not (and (eq (vector-type destination) (vector-type input))
                       (= (nd-span destination d-start d-start) (nd-span input i-start i-start))
                       (loop for dimension across shape for ds across d-strides for is across i-strides
                             always (or (<= dimension 1) (= ds is))))))))))

(defun nd-check-types (operation destination operands operator rounding d-encoding i-encoding)
  (let* ((destination-type (vector-type destination))
         (numeric (if (eq operation :select) (rest operands) operands))
         (type (cond ((member operation '(:compare :abs :convert))
                      (if (find-if #'operand-vector-p numeric) (vector-type (find-if #'operand-vector-p numeric))
                          (if (eq operation :compare) :u8 destination-type)))
                     (t destination-type))))
    (unless (eq destination-type
                (cond ((eq operation :compare) :u8)
                      ((and (eq operation :abs) (complex-type-p type))
                       (if (eq type :c32) :f32 :f64))
                      (t destination-type)))
      (error "N-D destination has the wrong element type"))
    (when (and (eq operation :select)
               (not (and (operand-vector-p (first operands))
                         (eq (vector-type (first operands)) :u8))))
      (error "Selection needs a byte mask"))
    (when (eq operation :convert)
      (unless (operand-vector-p (first operands)) (error "Operation needs a vector input")))
    (when (and (member operation '(:log :tanh :sigmoid :exp :sin :cos :silu :gelu))
               (not (member type '(:f32 :f64))))
      (error "Operation requires real floating-point input"))
    (when (and (integer-type-p type) (member operation '(:sqrt :reciprocal)))
      (error "Operation requires floating-point input"))
    (when (and (complex-type-p type) (member operation '(:min :max :clamp)))
      (error "Complex values have no ordering"))
    (when (eq operation :compare)
      (unless (member operator '(:eq :ne :lt :le :gt :ge)) (error "Invalid comparison"))
      (when (and (complex-type-p type) (not (member operator '(:eq :ne))))
        (error "Complex values support only equality")))
    (if (eq operation :convert)
        (progn
          (unless (member rounding '(:nearest-even :truncate :floor :ceiling))
            (error "Invalid conversion rounding"))
          (when (or d-encoding i-encoding)
            (check-float-encodings destination (first operands) d-encoding i-encoding)))
        (dolist (operand numeric)
          (unless (if (operand-vector-p operand) (eq type (vector-type operand))
                      (typep operand (second (numeric-type type))))
            (error "N-D operand has the wrong element type"))))
    (when (and (eq operation :abs) (not (complex-type-p type)) (not (eq type destination-type)))
      (error "Absolute value destination has the wrong type"))
    (when (and (eq operation :clamp) (numberp (second operands)) (numberp (third operands))
               (> (second operands) (third operands)))
      (error "CLAMP! lower bound exceeds upper bound"))
    type))

(defun nd-convert-value (value destination-type input-type rounding d-encoding i-encoding)
  (if (or d-encoding i-encoding)
      (with-single-float-bits (to-single-bits from-single-bits)
        (with-double-float-bits (to-double-bits from-double-bits)
          (cond ((eq d-encoding i-encoding) value)
                (i-encoding
                 (let ((bits (if (eq i-encoding :bf16) (bf16-to-f32-bits value) (f16-to-f32-bits value))))
                   (cond (d-encoding
                          (if (eq d-encoding :bf16) (f32-bits-to-bf16 bits rounding)
                              (f32-bits-to-f16 bits rounding)))
                         ((eq destination-type :f64) (from-double-bits (f32-bits-to-f64-bits bits)))
                         (t (from-single-bits bits)))))
                ((eq input-type :f64) (f64-bits-to-encoded (to-double-bits value) d-encoding rounding))
                (t (if (eq d-encoding :bf16) (f32-bits-to-bf16 (to-single-bits value) rounding)
                       (f32-bits-to-f16 (to-single-bits value) rounding))))))
      (convert-element value destination-type rounding)))

(declaim (inline nd-silu-value nd-gelu-value))

(defun nd-silu-value (x)
  (* x (sigmoid x)))

(defun nd-gelu-value (x)
  (* (* (float 0.5 x) x)
     (+ (float 1 x)
        (tanh (* (float 0.7978845608028654d0 x)
                 (+ x (* (float 0.044715d0 x) (* x (* x x)))))))))

(defun nd-lisp (operation destination operands shape layouts starts type operator rounding d-encoding i-encoding)
  (let* ((destination-type (vector-type destination))
         (integer-function (and (integer-type-p type) (member operation '(:add :subtract :multiply :divide))
                                (symbol-function (integer-operation-symbol operation type))))
         (inputs (coerce operands 'vector)))
    (nd-walk shape layouts starts
             (lambda (offsets)
               (flet ((input (index)
                        (let ((value (aref inputs index)))
                          (if (operand-vector-p value) (vector-ref value (aref offsets (1+ index))) value))))
                 (let* ((a (input 0))
                        (value
                          (case operation
                            ((:add :subtract :multiply :divide)
                             (let ((b (input 1)))
                               (if integer-function (funcall integer-function a b)
                                   (ecase operation (:add (+ a b)) (:subtract (- a b))
                                          (:multiply (* a b)) (:divide (/ a b))))))
                            ((:negate :abs) (wrapped-unary operation type a))
                            (:log (log a)) (:tanh (tanh a)) (:sigmoid (sigmoid a))
                            (:exp (exp a)) (:sin (sin a)) (:cos (cos a))
                            (:silu (nd-silu-value a)) (:gelu (nd-gelu-value a))
                            (:sqrt (if (complex-type-p type) (sqrt a) (kernel-sqrt a)))
                            (:reciprocal (/ (coerce 1 (if (eq type :c32) 'single-float
                                                        (if (eq type :c64) 'double-float
                                                            (second (numeric-type type))))) a))
                            (:min (kernel-min-left a (input 1)))
                            (:max (kernel-max-left a (input 1)))
                            (:clamp (let ((low (input 1)) (high (input 2)))
                                      (when (> low high) (error "CLAMP! lower bound exceeds upper bound"))
                                      (let ((bounded (if (> low a) low a)))
                                        (if (< high bounded) high bounded))))
                            (:compare (if (compare-values operator a (input 1)) 1 0))
                            (:select (if (zerop a) (input 2) (input 1)))
                            (:convert (nd-convert-value a destination-type type rounding d-encoding i-encoding)))))
                   (setf (vector-ref destination (aref offsets 0)) value)))))))

(define-native ("ts_nd_execute" %native-nd-execute) :int
  (operation :uint) (type :uint) (destination-type :uint) (input-type :uint)
  (comparison :uint) (rounding :int) (rank :size) (shape :pointer)
  (strides :pointer) (pointers :pointer) (indices :pointer))

(defvar *native-nd-available-p*
  (and *native-available-p* (null (missing-native-symbols '("ts_nd_execute")))))

(defvar *native-nd-activation-math-p*
  (and *native-available-p*
       (not (null (ignore-errors (cffi:foreign-symbol-pointer "ts_nd_activation_math"))))))

(defvar *native-nd-inference-math-p*
  (and *native-available-p*
       (not (null (ignore-errors (cffi:foreign-symbol-pointer "ts_nd_inference_math"))))))

(defun nd-with-pointers (values starts types function &optional (pointers nil))
  (if (null values)
      (funcall function (nreverse pointers))
      (let* ((value (first values)) (foreign (third (numeric-type (first types)))))
        (if (operand-vector-p value)
            (with-vector-pointer (pointer value)
              (nd-with-pointers (rest values) (rest starts) (rest types) function
                                (cons (element-pointer pointer foreign (first starts)) pointers)))
            (cffi:with-foreign-object (scalar foreign)
              (setf (cffi:mem-ref scalar foreign) value)
              (nd-with-pointers (rest values) (rest starts) (rest types) function (cons scalar pointers)))))))

(defun nd-coalesce (shape layouts)
  (let ((dimensions nil) (strides (make-array (length layouts) :initial-element nil)))
    (loop for axis downfrom (1- (length shape)) to 0
          for dimension = (aref shape axis)
          unless (= dimension 1) do
      (if (and dimensions
               (loop for layout across layouts for steps across strides
                     always (= (aref layout axis) (* (first dimensions) (first steps)))))
          (setf (first dimensions) (* dimension (first dimensions)))
          (progn
            (push dimension dimensions)
            (loop for layout across layouts for index from 0 do
              (push (aref layout axis) (aref strides index))))))
    (values (coerce dimensions 'vector) (map 'vector (lambda (steps) (coerce steps 'vector)) strides))))

(defun nd-flat-lisp (operation destination operands shape layouts starts type operator rounding d-encoding i-encoding)
  (when (member operation '(:log :tanh :sigmoid :exp :sin :cos :silu :gelu)) (return-from nd-flat-lisp nil))
  (unless (and (<= (length shape) 1) (vectorp destination) (not (complex-type-p type))
               (or (zerop (length shape)) (= (aref (aref layouts 0) 0) 1))
               (every (lambda (input) (or (numberp input) (vectorp input))) operands)
               (loop for index from 1 below (length layouts)
                     always (or (zerop (length shape)) (member (aref (aref layouts index) 0) '(0 1)))))
    (return-from nd-flat-lisp nil))
  (let* ((count (if (zerop (length shape)) 1 (aref shape 0)))
         (offsets (coerce starts 'list))
         (values (loop for input in operands for index from 1
                       collect (if (and (vectorp input) (plusp (length shape))
                                        (zerop (aref (aref layouts index) 0)))
                                   (aref input (aref starts index)) input)))
         (offsets (cons (first offsets)
                        (loop for value in values for offset in (rest offsets)
                              collect (and (vectorp value) offset))))
         (*backend* (if (eq *backend* :sbcl) :sbcl :lisp)))
    (when (and (member operation '(:negate :abs :sqrt :reciprocal :clamp :select :convert))
               (not (vectorp (first values))))
      (return-from nd-flat-lisp nil))
    (destructuring-bind (d-start &optional a-start b-start c-start) offsets
      (destructuring-bind (a &optional b c) values
        (case operation
          ((:add :subtract :multiply :divide)
           (dispatch-binary type operation destination a b count d-start a-start b-start))
          ((:negate :abs :sqrt :reciprocal)
           (cond ((and (eq *backend* :sbcl) (member type '(:f32 :f64)))
                  #+(and sbcl x86-64)
                  (funcall (if (eq type :f32) #'%sbcl-nd-unary-f32 #'%sbcl-nd-unary-f64)
                           operation destination a count d-start a-start))
                 ((member type '(:f32 :f64))
                  (funcall (lisp-bulk-function "%LISP-UNARY-" destination)
                           operation destination a count d-start a-start))
                 (t (dotimes (i count)
                      (setf (aref destination (+ d-start i))
                            (wrapped-unary operation type (aref a (+ a-start i))))))))
          ((:min :max)
           (funcall (lisp-bulk-function "%LISP-MINMAX-" destination)
                    operation destination a b count d-start a-start b-start))
          (:clamp (funcall (lisp-bulk-function "%LISP-CLAMP-" destination)
                           destination a b c count d-start a-start b-start c-start))
          (:compare (funcall (lisp-bulk-symbol "%LISP-COMPARE-" type)
                             destination operator a b count d-start a-start b-start))
          (:select (funcall (lisp-bulk-function "%LISP-SELECT-" destination)
                            destination a b c count d-start a-start b-start c-start))
          (:convert (dotimes (i count)
                      (setf (aref destination (+ d-start i))
                            (nd-convert-value (aref a (+ a-start i)) (vector-type destination) type rounding
                                              d-encoding i-encoding)))))))
    t))

(defun nd-native (operation destination operands shape layouts starts type operator rounding d-encoding i-encoding)
  (let* ((rank (length shape))
         (values (cons destination operands))
         (types (cons (vector-type destination)
                      (mapcar (lambda (value) (if (operand-vector-p value) (vector-type value) type)) operands))))
    (cffi:with-foreign-objects ((dimensions :int64 (max 1 rank))
                                (strides :int64 (max 1 (* 4 rank)))
                                (indices :int64 (max 1 rank)) (pointers :pointer 4))
      (dotimes (axis rank)
        (setf (cffi:mem-aref dimensions :int64 axis) (aref shape axis)
              (cffi:mem-aref indices :int64 axis) 0)
        (dotimes (operand 4)
          (setf (cffi:mem-aref strides :int64 (+ (* operand rank) axis))
                (if (< operand (length layouts)) (aref (aref layouts operand) axis) 0))))
      (dotimes (operand 4) (setf (cffi:mem-aref pointers :pointer operand) (cffi:null-pointer)))
      (nd-with-pointers values (coerce starts 'list) types
        (lambda (addresses)
          (loop for address in addresses for index from 0 do
            (setf (cffi:mem-aref pointers :pointer index) address))
          (let ((status (%native-nd-execute
                         (position operation *nd-operations*) (position type *nd-types*)
                         (if (eq operation :convert) (encoded-type-code destination d-encoding) 0)
                         (if (eq operation :convert) (encoded-type-code (first operands) i-encoding) 0)
                         (or (position operator '(:eq :ne :lt :le :gt :ge)) 0)
                         (if rounding (encoding-rounding-code rounding) 0)
                         rank dimensions strides pointers indices)))
            (if (= status -3) (error 'division-by-zero) (check-native-extended-status status))))))))

(defun nd-operation (operation destination operands shape starts stride-specs
                      &key operator rounding destination-encoding input-encoding)
  ;; TODO: per-call layout setup caps short spans; reuse validated descriptors (#65).
  (let* ((shape (nd-shape shape)) (empty (find 0 shape))
         (type (nd-check-types operation destination operands operator rounding destination-encoding input-encoding))
         (values (coerce (cons destination operands) 'vector))
         (layouts (make-array (length values)))
         (starts (coerce starts 'vector))
         (lows (make-array (length values) :initial-element 0))
         (highs (make-array (length values) :initial-element 0)))
    (unless empty (nd-integer (reduce #'* shape :initial-value 1) 0))
    (loop for value across values for spec in stride-specs for index from 0 do
      (if (operand-vector-p value)
          (multiple-value-bind (strides start low high)
              (nd-layout shape spec (or (aref starts index) 0) value empty)
            (setf (aref layouts index) strides (aref starts index) start
                  (aref lows index) low (aref highs index) high))
          (progn
            (when (or spec (aref starts index)) (error "A scalar has no start or strides"))
            (setf (aref layouts index) (make-array (length shape) :initial-element 0)
                  (aref starts index) 0))))
    (unless (nd-unique-output-p shape (aref layouts 0) (aref starts 0))
      (error "N-D output maps multiple elements to the same storage"))
    (when empty (return-from nd-operation destination))
    (loop for index from 1 below (length values) for input = (aref values index)
          when (and (operand-vector-p input)
                    (nd-alias-p destination input shape (aref layouts 0) (aref layouts index)
                                (aref starts 0) (aref starts index) (aref lows 0) (aref highs 0)
                                (aref lows index) (aref highs index))) do
      (let* ((snapshot-shape (map 'vector (lambda (dimension stride) (if (zerop stride) 1 dimension))
                                  shape (aref layouts index)))
             (snapshot (make-array (reduce #'* snapshot-shape :initial-value 1)
                                   :element-type (second (numeric-type (vector-type input)))))
             (snapshot-strides (nd-contiguous-strides snapshot-shape))
             (position 0))
        (nd-walk snapshot-shape (vector (aref layouts index)) (vector (aref starts index))
                 (lambda (offsets) (setf (aref snapshot position) (vector-ref input (aref offsets 0)))
                   (incf position)))
        (dotimes (axis (length shape))
          (when (zerop (aref (aref layouts index) axis)) (setf (aref snapshot-strides axis) 0)))
        (setf (aref values index) snapshot (aref starts index) 0
              (aref layouts index) snapshot-strides)))
    (let* ((operands (coerce (subseq values 1) 'list))
           (native (and (eq *backend* :native) *native-nd-available-p*
                        (eq *native-array-access* :pointer) (not (complex-type-p type))
                        (or (not (member operation '(:log :tanh :sigmoid)))
                            *native-nd-activation-math-p*)
                        (or (not (member operation '(:exp :sin :cos :silu :gelu)))
                            *native-nd-inference-math-p*)
                        (or (not (eq operation :convert))
                            (and (member type '(:f32 :f64 :u16))
                                 (member (vector-type destination) '(:f32 :f64 :u16))
                                 (or destination-encoding input-encoding
                                     (and (member type '(:f32 :f64))
                                          (member (vector-type destination) '(:f32 :f64)))))))))
      (multiple-value-bind (shape layouts) (nd-coalesce shape layouts)
        (unless (and (not native)
                     (nd-flat-lisp operation destination operands shape layouts starts type operator rounding
                                   destination-encoding input-encoding))
          (funcall (if native #'nd-native #'nd-lisp)
                   operation destination operands shape layouts starts type operator rounding
                   destination-encoding input-encoding))))
    destination))

(defmacro define-nd-operation (name operation arguments layout-names &optional options)
  (let ((starts (mapcar (lambda (name) (intern (format nil "~A-START" name))) layout-names))
        (strides (mapcar (lambda (name) (intern (format nil "~A-STRIDES" name))) layout-names)))
    `(defun ,name (destination ,@arguments shape &key ,@starts ,@strides ,@options)
       (nd-operation ,operation destination (list ,@arguments) shape
                     (list ,@starts) (list ,@strides)
                     ,@(when options '(:rounding rounding :destination-encoding destination-encoding
                                       :input-encoding input-encoding))))))

(define-nd-operation nd-add! :add (left right) (destination left right))
(define-nd-operation nd-subtract! :subtract (left right) (destination left right))
(define-nd-operation nd-multiply! :multiply (left right) (destination left right))
(define-nd-operation nd-divide! :divide (left right) (destination left right))
(define-nd-operation nd-min! :min (left right) (destination left right))
(define-nd-operation nd-max! :max (left right) (destination left right))
(define-nd-operation nd-negate! :negate (input) (destination input))
(define-nd-operation nd-abs! :abs (input) (destination input))
(define-nd-operation nd-sqrt! :sqrt (input) (destination input))
(define-nd-operation nd-reciprocal! :reciprocal (input) (destination input))
(define-nd-operation nd-clamp! :clamp (input lower upper) (destination input lower upper))
(define-nd-operation nd-select! :select (mask on-true on-false) (destination mask true false))
(define-nd-operation nd-convert! :convert (input) (destination input)
  ((rounding :nearest-even) destination-encoding input-encoding))

(defun nd-compare! (mask operator left right shape &key mask-start left-start right-start
                                                   mask-strides left-strides right-strides)
  (nd-operation :compare mask (list left right) shape (list mask-start left-start right-start)
                (list mask-strides left-strides right-strides) :operator operator))

(define-nd-operation nd-log! :log (input) (destination input))
(define-nd-operation nd-tanh! :tanh (input) (destination input))
(define-nd-operation nd-sigmoid! :sigmoid (input) (destination input))

(define-nd-operation nd-exp! :exp (input) (destination input))
(define-nd-operation nd-sin! :sin (input) (destination input))
(define-nd-operation nd-cos! :cos (input) (destination input))
(define-nd-operation nd-silu! :silu (input) (destination input))
(define-nd-operation nd-gelu! :gelu (input) (destination input))
