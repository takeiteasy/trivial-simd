(in-package #:trivial-simd)

;; bf16 and f16 values are stored in (unsigned-byte 16) vectors as IEEE bit
;; patterns. The Lisp conversions work on bits, like the scalar helpers in
;; native/extended.c: narrowing rounds to nearest even and overflows to
;; infinity, or with TRUNCATE rounds toward zero and overflows to the largest
;; finite value. NaN keeps its sign and leading payload bits and becomes quiet.

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

(declaim (inline bf16-to-f32-bits f32-bits-to-bf16 f32-bits-to-f16 f16-to-f32-bits))

(defun bf16-to-f32-bits (half)
  (declare (type (unsigned-byte 16) half))
  (let ((bits (ash half 16)))
    (if (> (logand bits #x7fffffff) #x7f800000) (logior bits #x400000) bits)))

(defun f32-bits-to-bf16 (bits truncate)
  (declare (type (unsigned-byte 32) bits))
  (cond ((> (logand bits #x7fffffff) #x7f800000) (logior (ash bits -16) #x40))
        (truncate (ash bits -16))
        (t (ash (+ bits #x7fff (ldb (byte 1 16) bits)) -16))))

(defun f32-bits-to-f16 (bits truncate)
  (declare (type (unsigned-byte 32) bits))
  (let ((sign (logand (ash bits -16) #x8000))
        (magnitude (logand bits #x7fffffff)))
    (logior
     sign
     (cond ((> magnitude #x7f800000) (logior #x7e00 (ldb (byte 10 13) magnitude)))
           ((>= magnitude #x477ff000)         ; 65520 and above
            (if (and truncate (< magnitude #x7f800000)) #x7bff #x7c00))
           ((>= magnitude #x38800000)         ; f16 normal range
            (let ((rebiased (- magnitude (ash 112 23))))
              (ash (+ rebiased (if truncate 0 (+ #xfff (ldb (byte 1 13) rebiased)))) -13)))
           ((< (ash magnitude -23) 102) 0)    ; at most 2^-25
           (t (let* ((shift (- 126 (ash magnitude -23)))
                     (mantissa (logior (ldb (byte 23 0) magnitude) #x800000))
                     (quotient (ash mantissa (- shift)))
                     (remainder (ldb (byte shift 0) mantissa))
                     (half (ash 1 (1- shift))))
                (if (and (not truncate)
                         (or (> remainder half) (and (= remainder half) (oddp quotient))))
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

(defun lisp-encode-floats (encoding truncate destination input count d-offset i-offset)
  (declare (type (simple-array (unsigned-byte 16) (*)) destination)
           (type (simple-array single-float (*)) input)
           (type fixnum count d-offset i-offset))
  (with-single-float-bits (float-bits bits-float)
    (if (eq encoding :bf16)
        (dotimes (i count)
          (setf (aref destination (+ d-offset i))
                (f32-bits-to-bf16 (float-bits (aref input (+ i-offset i))) truncate)))
        (dotimes (i count)
          (setf (aref destination (+ d-offset i))
                (f32-bits-to-f16 (float-bits (aref input (+ i-offset i))) truncate))))))

(defun native-convert-encoded (encoding encode truncate destination input count d-offset i-offset)
  (with-native-bulk-output (output destination d-offset (if encode :uint16 :float) count)
    (with-native-bulk-input (source input i-offset (if encode :float :uint16) count)
      (if encode
          (funcall (if (eq encoding :bf16) #'%native-extended-f32-to-bf16 #'%native-extended-f32-to-f16)
                   output source count (if truncate 1 0))
          (funcall (if (eq encoding :bf16) #'%native-extended-bf16-to-f32 #'%native-extended-f16-to-f32)
                   output source count)))))

(defun check-float-encodings (destination input destination-encoding input-encoding rounding)
  "Validate CONVERT! encodings; return the encoding and whether it is the destination's."
  (dolist (encoding (list destination-encoding input-encoding))
    (unless (member encoding '(nil :bf16 :f16))
      (error "Unknown float encoding ~S; expected :BF16 or :F16" encoding)))
  (when (and destination-encoding input-encoding)
    (error "Only one side of a conversion can have a float encoding"))
  (when (and destination-encoding (member rounding '(:floor :ceiling)))
    (error "Narrowing to ~S supports only :NEAREST-EVEN and :TRUNCATE rounding, not ~S"
           destination-encoding rounding))
  (let ((encoded (if destination-encoding destination input))
        (other (if destination-encoding input destination)))
    (unless (eq (vector-type encoded) :u16)
      (error "A ~S vector must have element type (unsigned-byte 16)"
             (or destination-encoding input-encoding)))
    (unless (eq (vector-type other) :f32)
      (error "~S converts only to and from single-float vectors"
             (or destination-encoding input-encoding)))
    (values (or destination-encoding input-encoding) (and destination-encoding t))))

(defun native-encoded-conversion-p ()
  "True when bf16 and f16 CONVERT! calls the native library."
  (and (member *backend* '(:native :sbcl)) *native-float-encoding-available-p*))

(defun convert-encoded (encoding encode rounding destination input count d-offset i-offset)
  (let ((truncate (eq rounding :truncate)))
    (cond ((native-encoded-conversion-p)
           (native-convert-encoded encoding encode truncate destination input count d-offset i-offset))
          (encode
           (lisp-encode-floats encoding truncate destination input count d-offset i-offset))
          (t
           (lisp-decode-floats encoding destination input count d-offset i-offset)))))
