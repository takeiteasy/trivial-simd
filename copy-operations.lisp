(in-package #:trivial-simd)

(define-native ("ts_swap_bytes" %native-swap-bytes) :void
  (a :pointer) (b :pointer) (bytes :size))

(defmacro define-native-fills ()
  `(progn
     ,@(loop for (key nil foreign) in *numeric-types*
             collect
             `(define-native (,(format nil "ts_fill_~(~A~)" key)
                              ,(intern (format nil "%NATIVE-FILL-~A" key)))
                  :void
                (output :pointer)
                (value ,(if (integer-type-p key)
                            (intern (format nil "UINT~D" (fourth (numeric-type key))) :keyword)
                            foreign))
                (count :size)))))

(define-native-fills)

(defvar *native-copy-available-p*
  (and *native-available-p*
       (every (lambda (name) (ignore-errors (cffi:foreign-symbol-pointer name)))
              '("ts_swap_bytes" "ts_fill_f32" "ts_fill_f64" "ts_fill_u8" "ts_fill_u64"))))

;; Measured crossover on SBCL ARM64: the pinned native call costs about 0.2 us.
(defconstant +native-fill-minimum-bytes+ 4096)
(defconstant +native-swap-minimum-bytes+ 64)

(defun native-copy-p (type count minimum-bytes)
  (and (eq *backend* :native) *native-copy-available-p*
       (eq *native-array-access* :pointer)
       (not (complex-type-p type))
       (>= (* count (floor (fourth (numeric-type type)) 8)) minimum-bytes)))

(defmacro define-lisp-swaps ()
  `(progn
     ,@(loop for (key element) in (append *numeric-types*
                                          '((:c32 (complex single-float))
                                            (:c64 (complex double-float))))
             collect
             `(defun ,(intern (format nil "%LISP-SWAP-~A" key)) (x y count x-offset y-offset)
                (declare (type (simple-array ,element (*)) x y)
                         (type fixnum count x-offset y-offset))
                (dotimes (i count)
                  (rotatef (aref x (+ x-offset i)) (aref y (+ y-offset i))))))))

(define-lisp-swaps)

(defun native-swap (x y count x-offset y-offset)
  (let ((type (foreign-type x)))
    (with-pinned-pointers ((a x) (b y))
      (%native-swap-bytes (element-pointer a type x-offset)
                          (element-pointer b type y-offset)
                          (* count (cffi:foreign-type-size type))))))

(defun native-fill (destination value count offset)
  (let ((type (vector-type destination)))
    (with-pinned-pointers ((output destination))
      (funcall (native-bulk-function "%NATIVE-FILL-" type)
               (element-pointer output (foreign-type destination) offset)
               (native-scalar-bits type value)
               count))))

(defun copy! (destination source &key start end destination-start source-start
                                    stride destination-stride source-stride)
  "Copy the SOURCE slice into DESTINATION and return DESTINATION.
Overlapping slices of one vector copy as if SOURCE were read first."
  (multiple-value-bind (type count offsets strides)
      (resolve-slice (list destination source) (list destination-start source-start)
                     start end (list destination-stride source-stride) stride)
    (declare (ignore type))
    (destructuring-bind (destination-offset source-offset) offsets
      (with-bulk-staged ((destination destination-offset (first strides) :out)
                         (source source-offset (second strides)))
          (count)
        (replace destination source
                 :start1 destination-offset :end1 (+ destination-offset count)
                 :start2 source-offset :end2 (+ source-offset count)))))
  destination)

(defun fill! (destination value &key start end stride)
  "Set the slice of DESTINATION to VALUE and return DESTINATION."
  (multiple-value-bind (type count offsets strides)
      (resolve-slice (list destination) (list nil) start end nil stride)
    (unless (typep value (second (numeric-type type)))
      (error 'type-error :datum value :expected-type (second (numeric-type type))))
    (let ((offset (first offsets)))
      (with-staged ((destination offset (first strides) :out))
          (count :direct (native-copy-p type count +native-fill-minimum-bytes+))
        (if (native-copy-p type count +native-fill-minimum-bytes+)
            (native-fill destination value count offset)
            (fill destination value :start offset :end (+ offset count))))))
  destination)

(defun swap! (x y &key start end x-start y-start stride x-stride y-stride)
  "Exchange the slices of X and Y and return both vectors as two values."
  (multiple-value-bind (type count offsets strides)
      (resolve-slice (list x y) (list x-start y-start) start end
                     (list x-stride y-stride) stride)
    (destructuring-bind (x-offset y-offset) offsets
      (case (slice-overlap x x-offset (first strides) y y-offset (second strides) count)
        (:same (return-from swap! (values x y)))
        (:overlap (error "SWAP! output slices overlap")))
      (with-staged ((x x-offset (first strides) :in-out)
                    (y y-offset (second strides) :in-out))
          (count :direct (native-copy-p type count +native-swap-minimum-bytes+))
        (cond ((zerop count))
              ((native-copy-p type count +native-swap-minimum-bytes+)
               (native-swap x y count x-offset y-offset))
              (t (funcall (lisp-bulk-function "%LISP-SWAP-" x)
                          x y count x-offset y-offset))))))
  (values x y))
