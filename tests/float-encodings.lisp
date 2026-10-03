(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

;; The reference decodes bf16 and f16 from the format definition with rational
;; arithmetic and rounds to nearest even by searching the table of encoded
;; values, so it shares no bit tricks with the conversions under test.

(defun encoding-format (encoding)
  "Return the exponent bits, mantissa bits and exponent bias of ENCODING."
  (ecase encoding
    (:bf16 (values 8 7 127))
    (:f16 (values 5 10 15))))

(defun f32-bits (x)
  (simd::with-single-float-bits (to-bits from-bits) (to-bits x)))

(defun bits-f32 (bits)
  (simd::with-single-float-bits (to-bits from-bits) (from-bits bits)))

(defun encoded-magnitude (encoding bits)
  "Rational magnitude of finite BITS, ignoring the sign."
  (multiple-value-bind (exponent-bits mantissa-bits bias) (encoding-format encoding)
    (let ((exponent (ldb (byte exponent-bits mantissa-bits) bits))
          (mantissa (ldb (byte mantissa-bits 0) bits)))
      (if (zerop exponent)
          (* mantissa (expt 2 (- 1 bias mantissa-bits)))
          (* (+ (ash 1 mantissa-bits) mantissa) (expt 2 (- exponent bias mantissa-bits)))))))

(defun reference-decode (encoding bits)
  "Expected single-float bits for the encoded BITS."
  (multiple-value-bind (exponent-bits mantissa-bits) (encoding-format encoding)
    (let ((negative (logbitp 15 bits))
          (exponent (ldb (byte exponent-bits mantissa-bits) bits))
          (mantissa (ldb (byte mantissa-bits 0) bits)))
      (if (= exponent (1- (ash 1 exponent-bits)))
          (logior (if negative #x80000000 0) #x7f800000
                  (if (zerop mantissa) 0 #x400000)
                  (ash mantissa (- 23 mantissa-bits)))
          (let ((value (coerce (encoded-magnitude encoding bits) 'single-float)))
            (f32-bits (if negative (- value) value)))))))

(defparameter *signaling-nans-p*
  (ignore-errors (bits-f32 #x7f800001) t)
  "True when this Lisp can hold a signaling NaN single-float; ECL traps on one.")

(defun usable-input-p (f32-bits)
  (or *signaling-nans-p*
      (<= (ldb (byte 31 0) f32-bits) #x7f800000)
      (logbitp 22 f32-bits)))

(defvar *encoded-values* (make-hash-table))

(defun encoded-values (encoding)
  "Magnitudes of the non-negative finite encodings, then the overflow threshold
2^(emax+1) standing for the infinity encoding."
  (or (gethash encoding *encoded-values*)
      (setf (gethash encoding *encoded-values*)
            (multiple-value-bind (exponent-bits mantissa-bits) (encoding-format encoding)
              (let* ((infinity (ash (1- (ash 1 exponent-bits)) mantissa-bits))
                     (values (make-array (1+ infinity))))
                (dotimes (bits infinity)
                  (setf (aref values bits) (encoded-magnitude encoding bits)))
                (setf (aref values infinity)
                      (* 2 (encoded-magnitude encoding (ash (- (ash 1 exponent-bits) 2)
                                                           mantissa-bits))))
                values)))))

(defun reference-encode (encoding f32-bits &optional (rounding :nearest-even))
  "Expected encoded bits for single-float F32-BITS, rounding to nearest even or
toward zero, where finite overflow gives the largest finite value."
  (multiple-value-bind (exponent-bits mantissa-bits) (encoding-format encoding)
    (let* ((sign (if (logbitp 31 f32-bits) #x8000 0))
           (magnitude (ldb (byte 31 0) f32-bits))
           (infinity (ash (1- (ash 1 exponent-bits)) mantissa-bits)))
      (logior
       sign
       (cond ((> magnitude #x7f800000)
              (logior infinity (ash 1 (1- mantissa-bits))
                      (ldb (byte mantissa-bits (- 23 mantissa-bits)) magnitude)))
             ((= magnitude #x7f800000) infinity)
             (t
              (let* ((value (rational (bits-f32 magnitude)))
                     (table (encoded-values encoding))
                     (low 0) (high infinity))
                ;; Largest encoding whose magnitude does not exceed VALUE.
                (loop while (< low high)
                      do (let ((middle (ceiling (+ low high) 2)))
                           (if (<= (aref table middle) value)
                               (setf low middle)
                               (setf high (1- middle)))))
                (cond
                  ((eq rounding :truncate) (min low (1- infinity)))
                  ((or (= low infinity) (= (aref table low) value)) low)
                  (t
                    (let ((midpoint (/ (+ (aref table low) (aref table (1+ low))) 2)))
                      (cond ((< value midpoint) low)
                            ((> value midpoint) (1+ low))
                            ((evenp low) low)
                            (t (1+ low)))))))))))))

(defun u16-vector (contents)
  (make-array (length contents) :element-type '(unsigned-byte 16) :initial-contents contents))

(defun f32-vector-from-bits (bits)
  (map '(simple-array single-float (*)) #'bits-f32 bits))

(defun encode-inputs (encoding)
  "Single-float bit patterns for narrowing tests: every encoded value, the
midpoints between neighbours and one unit either side, edge cases and a
fixed pseudo-random sample."
  (multiple-value-bind (exponent-bits mantissa-bits) (encoding-format encoding)
    (let* ((infinity (ash (1- (ash 1 exponent-bits)) mantissa-bits))
           (table (encoded-values encoding))
           (inputs '()))
      (dotimes (bits #x10000)
        (push (reference-decode encoding bits) inputs))
      (dotimes (bits infinity)
        ;; Every midpoint, including the one below the overflow threshold, is
        ;; exactly representable as a single-float.
        (let ((midpoint (f32-bits (coerce (/ (+ (aref table bits) (aref table (1+ bits))) 2)
                                          'single-float))))
          (dolist (delta '(-1 0 1))
            (dolist (sign '(0 #x80000000))
              (push (logior sign (+ midpoint delta)) inputs)))))
      (dolist (bits '(#x00000000 #x80000000 #x00000001 #x80000001 #x007fffff #x00400000
                      #x00008000 #x00018000 #x7f7fffff #xff7fffff #x7f7f8000 #x7f7f7fff
                      #x7f800000 #xff800000 #x7f800001 #xff800001 #x7fc00000 #xffc00000
                      #x7fa00000 #x7f802000 #x7fffffff #x477fe000 #x477ff000 #x477fefff
                      #x47800000 #x33000000 #x33000001 #x33400000 #x387fe000 #x387ff000
                      #x38800000))
        (push bits inputs))
      (let ((state 72))
        (dotimes (i 20000)
          (setf state (ldb (byte 64 0) (+ (* state 6364136223846793005) 1442695040888963407)))
          (push (ash state -32) inputs)))
      (coerce (remove-if-not #'usable-input-p (nreverse inputs))
              '(simple-array (unsigned-byte 32) (*))))))

(defun first-mismatch (expected actual)
  (let ((index (mismatch expected actual)))
    (when index
      (list :index index
            :expected (and (< index (length expected)) (aref expected index))
            :actual (and (< index (length actual)) (aref actual index))))))

(test float-encoding-native-library-current
  (when simd::*native-available-p*
    (is-true simd::*native-float-encoding-available-p*
             "build/ lacks the bf16/f16 symbols, so the native path is untested; rebuild with cmake")))

(test float-encoding-decode-exhaustive
  (dolist (encoding '(:bf16 :f16))
    (let* ((input (u16-vector (loop for bits below #x10000 collect bits)))
           (expected (map 'vector (lambda (bits) (reference-decode encoding bits)) input))
           ;; An odd length leaves a tail after the packed blocks.
           (odd-input (subseq input 0 #xfffd)))
      (dolist (backend (available-backends))
        (with-backend (backend)
          (dolist (source (list input odd-input))
            (let ((destination (make-array (length source) :element-type 'single-float)))
              (is (eq destination (simd:convert! destination source :input-encoding encoding)))
              (let ((problem (first-mismatch (subseq expected 0 (length source))
                                             (map 'vector #'f32-bits destination))))
                (is (null problem) "~S ~S decode: ~S" backend encoding problem)))))))))

(test float-encoding-encode-rounding
  (dolist (encoding '(:bf16 :f16))
    (let* ((bits (encode-inputs encoding))
           (input (f32-vector-from-bits bits)))
      (dolist (rounding '(:nearest-even :truncate))
        (let ((expected (map 'vector (lambda (x) (reference-encode encoding x rounding)) bits)))
          (dolist (backend (available-backends))
            (with-backend (backend)
              (let ((destination (make-array (length input) :element-type '(unsigned-byte 16))))
                (is (eq destination (simd:convert! destination input :destination-encoding encoding
                                                                     :rounding rounding)))
                (let ((problem (first-mismatch expected destination)))
                  (is (null problem) "~S ~S ~S encode: ~S" backend encoding rounding
                      (and problem
                           (append problem (list :input (aref bits (getf problem :index))))))))))))))) 

(defun encode-bits (encoding f32-bits &optional (rounding :nearest-even))
  (let ((destination (make-array 1 :element-type '(unsigned-byte 16))))
    (simd:convert! destination (f32-vector-from-bits (list f32-bits))
                   :destination-encoding encoding :rounding rounding)
    (aref destination 0)))

(defun decode-bits (encoding bits)
  (let ((destination (make-array 1 :element-type 'single-float)))
    (simd:convert! destination (u16-vector (list bits)) :input-encoding encoding)
    (f32-bits (aref destination 0))))

(test float-encoding-special-values
  (dolist (backend (available-backends))
    (with-backend (backend)
      ;; f16: normal, overflow and the round-to-even boundary at 65520.
      (is (= #x3c00 (encode-bits :f16 #x3f800000)))         ; 1.0
      (is (= #x7bff (encode-bits :f16 #x477fe000)))         ; 65504, largest finite
      (is (= #x7bff (encode-bits :f16 #x477fefff)))         ; just below 65520
      (is (= #x7c00 (encode-bits :f16 #x477ff000)))         ; 65520 rounds to infinity
      (is (= #xfc00 (encode-bits :f16 #xd01502f9)))         ; -1e10
      ;; f16 subnormals, including ties to even and f32 subnormal inputs.
      (is (= #x0001 (encode-bits :f16 #x33800000)))         ; 2^-24, smallest subnormal
      (is (= #x0000 (encode-bits :f16 #x33000000)))         ; 2^-25 ties to zero
      (is (= #x0001 (encode-bits :f16 #x33000001)))         ; just above 2^-25
      (is (= #x0002 (encode-bits :f16 #x33c00000)))         ; 3*2^-25 ties to 2
      (is (= #x03ff (encode-bits :f16 #x387fc000)))         ; largest subnormal
      (is (= #x0400 (encode-bits :f16 #x387ff000)))         ; rounds up to smallest normal
      (is (= #x8000 (encode-bits :f16 #x80000001)))         ; f32 subnormal to -0
      ;; f16 infinities and NaN: sign kept, payload truncated, always quiet.
      (is (= #x7c00 (encode-bits :f16 #x7f800000)))
      (is (= #xfc00 (encode-bits :f16 #xff800000)))
      (is (= #x7e00 (encode-bits :f16 #x7fc00000)))
      (is (= #xfe00 (encode-bits :f16 #xffc00000)))
      (when *signaling-nans-p*
        (is (= #x7e00 (encode-bits :f16 #x7f800001)))       ; payload below f16 width
        (is (= #x7f00 (encode-bits :f16 #x7fa00000))))
      (is (= #x7f00 (encode-bits :f16 #x7fe00000)))
      (is (= #x7f800000 (decode-bits :f16 #x7c00)))
      (is (= #xff800000 (decode-bits :f16 #xfc00)))
      (is (= #x7fc00000 (decode-bits :f16 #x7e00)))
      (is (= #x7fc02000 (decode-bits :f16 #x7c01)))         ; signaling NaN becomes quiet
      (is (= #x33800000 (decode-bits :f16 #x0001)))         ; 2^-24
      (is (= #x387fc000 (decode-bits :f16 #x03ff)))
      (is (= #x80000000 (decode-bits :f16 #x8000)))         ; -0
      ;; bf16: ties to even, overflow, subnormals and NaN.
      (is (= #x3f80 (encode-bits :bf16 #x3f800000)))
      (is (= #x3f80 (encode-bits :bf16 #x3f808000)))        ; tie to even below
      (is (= #x3f82 (encode-bits :bf16 #x3f818000)))        ; tie to even above
      (is (= #x3f81 (encode-bits :bf16 #x3f808001)))
      (is (= #x7f80 (encode-bits :bf16 #x7f7fffff)))        ; largest f32 overflows
      (is (= #x7f7f (encode-bits :bf16 #x7f7f7fff)))
      (is (= #x0000 (encode-bits :bf16 #x00008000)))        ; subnormal tie to even
      (is (= #x0002 (encode-bits :bf16 #x00018000)))
      (is (= #x8001 (encode-bits :bf16 #x80010000)))
      (is (= #x7f80 (encode-bits :bf16 #x7f800000)))
      (is (= #xff80 (encode-bits :bf16 #xff800000)))
      (when *signaling-nans-p*
        (is (= #x7fc0 (encode-bits :bf16 #x7f800001)))      ; NaN never rounds to infinity
        (is (= #xffc0 (encode-bits :bf16 #xff800001))))
      (is (= #x7fc0 (encode-bits :bf16 #x7fc0ffff)))        ; quiet NaN payload never carries
      (is (= #x7fff (encode-bits :bf16 #x7fffffff)))
      (is (= #x00010000 (decode-bits :bf16 #x0001)))        ; bf16 subnormal stays subnormal
      (is (= #xff800000 (decode-bits :bf16 #xff80)))
      (is (= #x7fc10000 (decode-bits :bf16 #x7f81)))        ; signaling NaN becomes quiet
      (is (= #xffc00000 (decode-bits :bf16 #xffc0))))))

(test float-encoding-truncate-special-values
  (dolist (backend (available-backends))
    (with-backend (backend)
      (is (= #x7bff (encode-bits :f16 #x477ff000 :truncate)))    ; 65520 stays finite
      (is (= #x7bff (encode-bits :f16 #x47800000 :truncate)))    ; 65536
      (is (= #xfbff (encode-bits :f16 #xd01502f9 :truncate)))    ; -1e10
      (is (= #x7c00 (encode-bits :f16 #x7f800000 :truncate)))    ; infinity stays
      (is (= #x7e00 (encode-bits :f16 #x7fc00000 :truncate)))
      (is (= #x0000 (encode-bits :f16 #x337fffff :truncate)))    ; just below 2^-24 to zero
      (is (= #x0001 (encode-bits :f16 #x33ffffff :truncate)))    ; almost 2^-23 truncates down
      (is (= #x03ff (encode-bits :f16 #x387ff000 :truncate)))    ; not rounded up to normal
      (is (= #x7f7f (encode-bits :bf16 #x7f7fffff :truncate)))   ; no overflow to infinity
      (is (= #x3f81 (encode-bits :bf16 #x3f81ffff :truncate)))
      (is (= #xbf81 (encode-bits :bf16 #xbf81ffff :truncate)))   ; toward zero, not floor
      (is (= #x7f80 (encode-bits :bf16 #x7f800000 :truncate))))))

(test float-encoding-slices-and-tails
  (dolist (backend (available-backends))
    (with-backend (backend)
      (dolist (encoding '(:bf16 :f16))
        (let* ((bits (loop for i below 40 collect (reference-decode :f16 (* 811 (1+ i)))))
               (input (f32-vector-from-bits bits))
               (expected (map 'vector (lambda (x) (reference-encode encoding x)) bits)))
          (loop for count from 0 to 19 do
            (let ((destination (make-array 45 :element-type '(unsigned-byte 16)
                                              :initial-element 7)))
              (simd:convert! destination input :destination-encoding encoding
                                               :start 3 :end (+ 3 count) :destination-start 5)
              (is (equalp (subseq destination 5 (+ 5 count)) (subseq expected 3 (+ 3 count))))
              (is (every (lambda (x) (= x 7)) (subseq destination 0 5)))
              (is (every (lambda (x) (= x 7)) (subseq destination (+ 5 count))))
              (let ((round-trip (make-array 45 :element-type 'single-float :initial-element 9.0)))
                (simd:convert! round-trip destination :input-encoding encoding
                                                      :start 5 :end (+ 5 count))
                (is (equalp (map 'vector #'f32-bits (subseq round-trip 5 (+ 5 count)))
                            (map 'vector (lambda (x) (reference-decode encoding x))
                                 (subseq expected 3 (+ 3 count)))))
                (is (= 9.0 (aref round-trip 4)))))))))))

(test float-encoding-errors
  (let ((halves (make-array 4 :element-type '(unsigned-byte 16)))
        (singles (make-array 4 :element-type 'single-float))
        (doubles (make-array 4 :element-type 'double-float))
        (bytes (make-array 4 :element-type '(unsigned-byte 8))))
    (dolist (backend (available-backends))
      (with-backend (backend)
        (signals error (simd:convert! halves singles :destination-encoding :f8))
        (signals error (simd:convert! singles halves :input-encoding :bfloat16))
        (signals error (simd:convert! bytes singles :destination-encoding :f16))
        (signals error (simd:convert! singles bytes :input-encoding :bf16))
        (signals error (simd:convert! halves singles :destination-encoding :bf16 :rounding :floor))
        (signals error (simd:convert! halves singles :destination-encoding :f16 :rounding :ceiling))
        (simd:convert! singles halves :input-encoding :f16 :rounding :floor)
        (signals error (simd:convert! halves doubles :destination-encoding :bf16))
        (signals error (simd:convert! doubles halves :input-encoding :f16))
        (signals error (simd:convert! halves halves :input-encoding :f16
                                                    :destination-encoding :bf16))
        (signals error (simd:convert! halves (make-array 3 :element-type 'single-float)
                                      :destination-encoding :f16))
        (signals error (simd:convert! singles halves :input-encoding :f16 :end 5))
        ;; Without an encoding an (unsigned-byte 16) vector holds integers.
        (simd:convert! singles (u16-vector '(1 2 3 #x3c00)))
        (is (= #x3c00 (aref singles 3)))))))
