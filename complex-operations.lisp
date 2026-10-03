(in-package #:trivial-simd)

(define-native ("ts_complex_binary_c32" %native-complex-binary-c32) :void
  (operation :uint) (output :pointer) (left :pointer) (right :pointer) (count :size))
(define-native ("ts_complex_binary_c64" %native-complex-binary-c64) :void
  (operation :uint) (output :pointer) (left :pointer) (right :pointer) (count :size))

(defvar *native-complex-available-p*
  (and *native-available-p*
       (ignore-errors (cffi:foreign-symbol-pointer "ts_complex_binary_c32"))
       (ignore-errors (cffi:foreign-symbol-pointer "ts_complex_binary_c64"))))

(defmacro define-native-complex-transfer (name foreign native-call)
  `(defun ,name (operation destination left right count d-offset l-offset r-offset)
     (cffi:with-foreign-objects ((output ,foreign (* 2 count))
                                 (a ,foreign (* 2 count))
                                 (b ,foreign (* 2 count)))
       (dotimes (i count)
         (let ((x (aref left (+ l-offset i)))
               (y (aref right (+ r-offset i))))
           (setf (cffi:mem-aref a ,foreign (* 2 i)) (realpart x)
                 (cffi:mem-aref a ,foreign (1+ (* 2 i))) (imagpart x)
                 (cffi:mem-aref b ,foreign (* 2 i)) (realpart y)
                 (cffi:mem-aref b ,foreign (1+ (* 2 i))) (imagpart y))))
       (,native-call (position operation '(:add :subtract :multiply))
                     output a b count)
       (dotimes (i count)
         (setf (aref destination (+ d-offset i))
               (complex (cffi:mem-aref output ,foreign (* 2 i))
                        (cffi:mem-aref output ,foreign (1+ (* 2 i)))))))
     destination))

(define-native-complex-transfer native-complex-binary-c32 :float
  %native-complex-binary-c32)
(define-native-complex-transfer native-complex-binary-c64 :double
  %native-complex-binary-c64)

(defun complex-zero (type)
  (if (eq type :c32) #C(0.0f0 0.0f0) #C(0.0d0 0.0d0)))

(defun complex-binary (operation destination left right count d-offset l-offset r-offset)
  ;; TODO: Per-call copies cap short-span speed; evaluate pinned or reused buffers (#74).
  (when (and (eq *backend* :native) *native-complex-available-p*
             (vectorp left) (vectorp right) (>= count 32)
             (member operation '(:add :subtract :multiply)))
    (return-from complex-binary
      (funcall (if (eq (vector-type destination) :c32)
                   #'native-complex-binary-c32 #'native-complex-binary-c64)
               operation destination left right count d-offset l-offset r-offset)))
  (dotimes (i count destination)
    (let ((a (operand-value left l-offset i))
          (b (operand-value right r-offset i)))
      (setf (aref destination (+ d-offset i))
            (ecase operation
              (:add (+ a b)) (:subtract (- a b))
              (:multiply (* a b)) (:divide (/ a b)))))))

(defun complex-scale (x a count offset)
  (dotimes (i count x)
    (setf (aref x (+ offset i)) (* a (aref x (+ offset i))))))

(defun complex-axpy (y a x count y-offset x-offset)
  (dotimes (i count y)
    (incf (aref y (+ y-offset i)) (* a (aref x (+ x-offset i))))))

(defun complex-sum (input count offset &optional wide)
  (let ((result (if wide #C(0d0 0d0) (complex-zero (vector-type input)))))
    (dotimes (i count result)
      (incf result (aref input (+ offset i))))))

(defun widen-complex (value)
  (complex (coerce (realpart value) 'double-float)
           (coerce (imagpart value) 'double-float)))

(defun complex-dot (left right count left-offset right-offset conjugate-left &optional wide)
  (let ((result (if wide #C(0d0 0d0) (complex-zero (vector-type left)))))
    (dotimes (i count result)
      (let ((a (aref left (+ left-offset i)))
            (b (aref right (+ right-offset i))))
        (when wide (setf a (widen-complex a) b (widen-complex b)))
        (incf result (* (if conjugate-left (conjugate a) a) b))))))

(defun complex-asum (input count offset wide)
  (let ((result (if (and (eq (vector-type input) :c32) (not wide)) 0.0f0 0.0d0)))
    (dotimes (i count result)
      (let ((value (aref input (+ offset i))))
        (incf result (abs (if wide (widen-complex value) value)))))))

(defun complex-nrm2 (input count offset)
  "Euclidean norm of complex single-float elements, accumulated in double precision."
  (let ((sum 0d0))
    (dotimes (i count (sqrt sum))
      (let ((value (widen-complex (aref input (+ offset i)))))
        (incf sum (+ (expt (realpart value) 2) (expt (imagpart value) 2)))))))

(defun complex-unary (operation destination input count d-offset i-offset)
  (dotimes (i count destination)
    (let ((value (aref input (+ i-offset i))))
      (setf (aref destination (+ d-offset i))
            (ecase operation
              (:negate (- value))
              (:sqrt (sqrt value)) (:reciprocal (/ value)))))))

(defun complex-magnitude! (destination input start end destination-start input-start
                           stride destination-stride input-stride)
  (let* ((source-type (vector-type input))
         (target-type (vector-type destination)))
    (unless (and (complex-type-p source-type)
                 (eq target-type (if (eq source-type :c32) :f32 :f64)))
      (error "Complex magnitude requires matching real destination precision"))
    (multiple-value-bind (count offsets strides)
        (resolve-mixed-slice (list destination input)
                             (list destination-start input-start) start end
                             (list destination-stride input-stride) stride)
      (destructuring-bind (d-offset i-offset) offsets
        (with-staged ((destination d-offset (first strides) :out)
                      (input i-offset (second strides)))
            (count)
          (dotimes (i count)
            (setf (aref destination (+ d-offset i))
                  (abs (aref input (+ i-offset i))))))
        destination))))
