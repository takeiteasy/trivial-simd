(in-package #:trivial-simd)

(defmacro with-single-float-bits ((to-bits from-bits) &body body)
  "Run BODY with local functions between a single-float and its 32 IEEE bits."
  #+sbcl
  `(flet ((,to-bits (x)
            (declare (type single-float x))
            (ldb (byte 32 0) (sb-kernel:single-float-bits x)))
          (,from-bits (bits)
            (declare (type (unsigned-byte 32) bits))
            (sb-kernel:make-single-float
             (if (logbitp 31 bits) (- bits (ash 1 32)) bits))))
     (declare (inline ,to-bits ,from-bits) (ignorable #',to-bits #',from-bits))
     ,@body)
  #-sbcl
  (let ((cell (gensym "CELL")))
    `(cffi:with-foreign-object (,cell :uint32)
       (flet ((,to-bits (x)
                (setf (cffi:mem-ref ,cell :float) x)
                (the (unsigned-byte 32) (cffi:mem-ref ,cell :uint32)))
              (,from-bits (bits)
                (setf (cffi:mem-ref ,cell :uint32) bits)
                (the single-float (cffi:mem-ref ,cell :float))))
         (declare (inline ,to-bits ,from-bits) (ignorable #',to-bits #',from-bits))
         ,@body))))

(defmacro with-double-float-bits ((to-bits from-bits) &body body)
  #+sbcl
  `(flet ((,to-bits (x)
            (declare (type double-float x))
            (logior (ash (ldb (byte 32 0) (sb-kernel:double-float-high-bits x)) 32)
                    (sb-kernel:double-float-low-bits x)))
          (,from-bits (bits)
            (let ((high (ldb (byte 32 32) bits)))
              (sb-kernel:make-double-float
               (if (logbitp 31 high) (- high (ash 1 32)) high)
               (ldb (byte 32 0) bits)))))
     (declare (inline ,to-bits ,from-bits) (ignorable #',to-bits #',from-bits))
     ,@body)
  #-sbcl
  (let ((cell (gensym "CELL")))
    `(cffi:with-foreign-object (,cell :uint64)
       (flet ((,to-bits (x)
                (setf (cffi:mem-ref ,cell :double) x)
                (cffi:mem-ref ,cell :uint64))
              (,from-bits (bits)
                (setf (cffi:mem-ref ,cell :uint64) bits)
                (cffi:mem-ref ,cell :double)))
         (declare (inline ,to-bits ,from-bits) (ignorable #',to-bits #',from-bits))
         ,@body))))

(defun f64-bits-to-encoded (bits encoding rounding)
  (let* ((precision (if (eq encoding :f16) 10 7))
         (bias (if (eq encoding :f16) 15 127))
         (infinity (ash (if (eq encoding :f16) 31 255) precision))
         (sign (if (logbitp 63 bits) #x8000 0))
         (exponent (ldb (byte 11 52) bits))
         (fraction (ldb (byte 52 0) bits))
         (away (or (and (eq rounding :floor) (plusp sign))
                   (and (eq rounding :ceiling) (zerop sign)))))
    (logior sign
      (cond ((= exponent 2047)
             (logior infinity (if (zerop fraction) 0
                                  (logior (ash 1 (1- precision))
                                          (ash fraction (- precision 52))))))
            (t
             (let* ((power (if (zerop exponent) -1022 (- exponent 1023)))
                    (mantissa (logior fraction (if (zerop exponent) 0 (ash 1 52)))))
               (if (> power bias)
                   (if (or (eq rounding :nearest-even) away) infinity (1- infinity))
                   (let* ((normal (>= power (- 1 bias)))
                          (shift (+ (- 52 precision) (if normal 0 (- 1 bias power))))
                          (quotient (ash mantissa (- shift)))
                          (remainder (ldb (byte shift 0) mantissa))
                          (half (ash 1 (1- shift)))
                          (increment (if (eq rounding :nearest-even)
                                         (or (> remainder half)
                                             (and (= remainder half) (oddp quotient)))
                                         (and away (plusp remainder))))
                          (result (+ (if normal (ash (+ power bias -1) precision) 0)
                                     quotient (if increment 1 0))))
                     (if (and (>= result infinity) (not (eq rounding :nearest-even)) (not away))
                         (1- infinity) result)))))))))

(defun f32-bits-to-f64-bits (bits)
  (let ((sign (ash (logand bits #x80000000) 32))
        (exponent (ldb (byte 8 23) bits))
        (fraction (ldb (byte 23 0) bits)))
    (logior sign
      (cond ((= exponent 255) (logior #x7ff0000000000000 (ash fraction 29)))
            ((plusp exponent) (logior (ash (+ exponent 896) 52) (ash fraction 29)))
            ((zerop fraction) 0)
            (t (let ((shift (- 24 (integer-length fraction))))
                 (logior (ash (- 897 shift) 52)
                         (ash (ldb (byte 23 0) (ash fraction shift)) 29))))))))

(declaim (inline bf16-to-f32-bits f32-bits-to-bf16 f32-bits-to-f16 f16-to-f32-bits))

(defun bf16-to-f32-bits (half)
  (declare (type (unsigned-byte 16) half))
  (let ((bits (ash half 16)))
    (if (> (logand bits #x7fffffff) #x7f800000) (logior bits #x400000) bits)))

(declaim (inline encoding-round-away-p encoding-round-bias))

(defun encoding-round-away-p (bits rounding)
  (or (and (eq rounding :floor) (logbitp 31 bits))
      (and (eq rounding :ceiling) (not (logbitp 31 bits)))))

(defun encoding-round-bias (bits rounding shift)
  (cond ((eq rounding :nearest-even)
         (+ (1- (ash 1 (1- shift))) (ldb (byte 1 shift) bits)))
        ((encoding-round-away-p bits rounding) (1- (ash 1 shift)))
        (t 0)))

(defun f32-bits-to-bf16 (bits rounding)
  (declare (type (unsigned-byte 32) bits))
  (if (> (logand bits #x7fffffff) #x7f800000)
      (logior (ash bits -16) #x40)
      (ash (+ bits (encoding-round-bias bits rounding 16)) -16)))

(defun f32-bits-to-f16 (bits rounding)
  (declare (type (unsigned-byte 32) bits))
  (let* ((sign (logand (ash bits -16) #x8000))
         (magnitude (logand bits #x7fffffff))
         (away (encoding-round-away-p bits rounding)))
    (logior
     sign
     (cond ((> magnitude #x7f800000) (logior #x7e00 (ldb (byte 10 13) magnitude)))
           ((>= magnitude #x477ff000) ; 65520 and above
            (if (and (< magnitude #x7f800000)
                     (not (eq rounding :nearest-even)) (not away))
                #x7bff #x7c00))
           ((>= magnitude #x38800000) ; f16 normal range
            (let ((rebiased (- magnitude (ash 112 23))))
              (ash (+ rebiased (encoding-round-bias bits rounding 13)) -13)))
           ((< (ash magnitude -23) 102) ; below 2^-25
            (if (and away (plusp magnitude)) 1 0))
           (t (let* ((shift (- 126 (ash magnitude -23)))
                     (mantissa (logior (ldb (byte 23 0) magnitude) #x800000))
                     (quotient (ash mantissa (- shift)))
                     (remainder (ldb (byte shift 0) mantissa))
                     (half (ash 1 (1- shift))))
                (if (if (eq rounding :nearest-even)
                        (or (> remainder half) (and (= remainder half) (oddp quotient)))
                        (and away (plusp remainder)))
                    (1+ quotient)
                    quotient)))))))

(defun f16-to-f32-bits (half)
  (declare (type (unsigned-byte 16) half))
  (let ((sign (ash (logand half #x8000) 16))
        (exponent (ldb (byte 5 10) half))
        (mantissa (ldb (byte 10 0) half)))
    (logior
     sign
     (cond ((= exponent 31)
            (logior #x7f800000 (ash mantissa 13) (if (zerop mantissa) 0 #x400000)))
           ((and (zerop exponent) (zerop mantissa)) 0)
           ((zerop exponent)
            ;; Normalize the subnormal: its leading bit becomes the implicit bit.
            (let ((shift (- 11 (integer-length mantissa))))
              (logior (ash (- 113 shift) 23)
                      (ash (ldb (byte 10 0) (ash mantissa shift)) 13))))
           (t (logior (ash (+ exponent 112) 23) (ash mantissa 13)))))))

(defun lisp-decode-floats (encoding destination input count d-offset i-offset)
  (declare (type (simple-array single-float (*)) destination)
           (type (simple-array (unsigned-byte 16) (*)) input)
           (type fixnum count d-offset i-offset))
  (with-single-float-bits (float-bits bits-float)
    (if (eq encoding :bf16)
        (dotimes (i count)
          (setf (aref destination (+ d-offset i))
                (bits-float (bf16-to-f32-bits (aref input (+ i-offset i))))))
        (dotimes (i count)
          (setf (aref destination (+ d-offset i))
                (bits-float (f16-to-f32-bits (aref input (+ i-offset i)))))))))

(defun lisp-encode-floats (encoding rounding destination input count d-offset i-offset)
  (declare (type (simple-array (unsigned-byte 16) (*)) destination)
           (type (simple-array single-float (*)) input)
           (type fixnum count d-offset i-offset))
  (with-single-float-bits (float-bits bits-float)
    (if (eq encoding :bf16)
        (dotimes (i count)
          (setf (aref destination (+ d-offset i))
                (f32-bits-to-bf16 (float-bits (aref input (+ i-offset i))) rounding)))
        (dotimes (i count)
          (setf (aref destination (+ d-offset i))
                (f32-bits-to-f16 (float-bits (aref input (+ i-offset i))) rounding))))))

(defun native-convert-encoded (encoding encode rounding destination input count d-offset i-offset)
  (with-native-bulk-output (output destination d-offset (if encode :uint16 :float) count)
    (with-native-bulk-input (source input i-offset (if encode :float :uint16) count)
      (if encode
          (funcall (if (eq encoding :bf16) #'%native-extended-f32-to-bf16 #'%native-extended-f32-to-f16)
                   output source count (encoding-rounding-code rounding))
          (funcall (if (eq encoding :bf16) #'%native-extended-bf16-to-f32 #'%native-extended-f16-to-f32)
                   output source count)))))

(defun check-float-encodings (destination input destination-encoding input-encoding)
  (loop for vector in (list destination input)
        for encoding in (list destination-encoding input-encoding) do
    (unless (member encoding '(nil :bf16 :f16))
      (error "Unknown float encoding ~S; expected :BF16 or :F16" encoding))
    (unless (if encoding (eq (vector-type vector) :u16)
                (member (vector-type vector) '(:f32 :f64)))
      (error "Encoded conversions require unsigned-byte-16 storage or a float vector")))
  (values (or destination-encoding input-encoding) (and destination-encoding t)))

(defun extended-encoded-conversion-p (destination input destination-encoding input-encoding)
  (or (and destination-encoding input-encoding)
      (eq (vector-type destination) :f64) (eq (vector-type input) :f64)))

(defun encoded-type-code (vector encoding)
  (if encoding (ecase encoding (:bf16 2) (:f16 3))
      (ecase (vector-type vector) (:f32 0) (:f64 1))))

(defun lisp-convert-extended-encoded (destination input destination-encoding input-encoding
                                    rounding count d-offset i-offset)
  (with-double-float-bits (float-bits bits-float)
    (dotimes (i count)
      (let ((value (aref input (+ i-offset i))))
        (setf (aref destination (+ d-offset i))
              (cond ((eq destination-encoding input-encoding) value)
                    ((null input-encoding)
                     (f64-bits-to-encoded (float-bits value) destination-encoding rounding))
                    (t (let ((bits (if (eq input-encoding :bf16)
                                       (bf16-to-f32-bits value) (f16-to-f32-bits value))))
                         (if destination-encoding
                             (if (eq destination-encoding :bf16)
                                 (f32-bits-to-bf16 bits rounding) (f32-bits-to-f16 bits rounding))
                             (bits-float (f32-bits-to-f64-bits bits)))))))))))

(defun native-convert-extended-encoded (destination input destination-encoding input-encoding
                                      rounding count d-offset i-offset)
  (with-native-bulk-output (output destination d-offset (foreign-type destination) count)
    (with-native-bulk-input (source input i-offset (foreign-type input) count)
      (%native-convert-encoded output source count
                              (encoded-type-code destination destination-encoding)
                              (encoded-type-code input input-encoding)
                              (encoding-rounding-code rounding)))))

(defun encoding-rounding-code (rounding)
  (ecase rounding (:nearest-even 0) (:truncate 1) (:floor 2) (:ceiling 3)))

(defun native-encoded-conversion-p (encode rounding &optional extended)
  "True when this bf16 or f16 conversion uses the native library."
  (and (member *backend* '(:native :sbcl)) *native-available-p*
       (if extended
           (logbitp (encoding-rounding-code rounding) *native-encoded-conversion-rounding-modes*)
           (and *native-float-encoding-available-p*
                (or (not encode)
                    (logbitp (encoding-rounding-code rounding) *native-float-encoding-rounding-modes*))))))

(defun convert-encoded (encoding encode rounding destination input count d-offset i-offset
                        destination-encoding input-encoding)
  (cond ((extended-encoded-conversion-p destination input destination-encoding input-encoding)
         (funcall (if (native-encoded-conversion-p encode rounding t)
                      #'native-convert-extended-encoded #'lisp-convert-extended-encoded)
                  destination input destination-encoding input-encoding rounding count d-offset i-offset))
        ((native-encoded-conversion-p encode rounding)
         (native-convert-encoded encoding encode rounding destination input count d-offset i-offset))
        (encode
         (lisp-encode-floats encoding rounding destination input count d-offset i-offset))
        (t
         (lisp-decode-floats encoding destination input count d-offset i-offset))))
