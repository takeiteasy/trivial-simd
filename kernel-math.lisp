(in-package #:trivial-simd)

(defun kernel-sqrt (value)
  (when (and (realp value) (minusp value))
    (error "Negative kernel square root operand: ~S" value))
  (sqrt value))

(defun kernel-min (left right)
  (if (<= left right) left right))

(defun kernel-max (left right)
  (if (>= left right) left right))

(defun portable-fma (a b c)
  "Return A*B+C with one nearest-even rounding for same-type finite floats."
  (let* ((single (typep a 'single-float))
         (type (if single 'single-float 'double-float)))
    (unless (and (typep a type) (typep b type) (typep c type))
      (error "FMA needs three floats of the same type"))
    (multiple-value-bind (am ae as) (integer-decode-float a)
      (multiple-value-bind (bm be bs) (integer-decode-float b)
        (multiple-value-bind (cm ce cs) (integer-decode-float c)
          (let* ((pe (+ ae be))
                 (exponent (min pe ce))
                 (exact (+ (ash (* am bm as bs) (- pe exponent))
                           (ash (* cm cs) (- ce exponent))))
                 (zero (coerce 0 type)))
            (if (zerop exact)
                (if (and (or (zerop am) (zerop bm)) (zerop cm)
                         (minusp (* as bs)) (minusp cs))
                    (- zero) zero)
                (let* ((minimum (nth-value 1 (integer-decode-float
                                              (if single
                                                  least-positive-normalized-single-float
                                                  least-positive-normalized-double-float))))
                       (shift (max 0 (- (integer-length (abs exact)) (float-digits a))
                                   (- minimum exponent)))
                       (rounded (round (abs exact) (ash 1 shift)))
                       (result (scale-float (float rounded a) (+ exponent shift))))
                  (if (minusp exact) (- result) result)))))))))

(defvar *fma-mode* :auto)
(defvar *arm64-fma-available-p* nil)
(defvar *sbcl-fma-available-p* nil)
(defvar *sbcl-fma-f32* nil)
(defvar *sbcl-fma-f64* nil)

(defun refresh-sbcl-fma ()
  (setf *sbcl-fma-available-p*
        #+(and sbcl x86-64)
        (and *sbcl-fma-f32* *sbcl-fma-f64*
             (ignore-errors
               (and (plusp (sb-alien:extern-alien "avx_supported" sb-alien:int))
                    (logbitp 12 (nth-value 2 (sb-vm::%cpu-identification 1 0))))))
        #-(and sbcl x86-64) nil))

#+(and sbcl x86-64)
(defun initialize-sbcl-fma ()
  (let ((package (find-package "SB-SIMD-FMA")))
    (when (and package
               (every (lambda (name) (let ((symbol (find-symbol name package)))
                                      (and symbol (fboundp symbol))))
                      '("F32-FMADD" "F64-FMADD" "F32.4-FMADD" "F64.2-FMADD")))
      (flet ((helper (name type)
               (compile nil `(lambda (a b c)
                               (declare (type ,type a b c) (optimize (speed 3)))
                               (,(find-symbol name package) a b c)))))
        (setf *sbcl-fma-f32* (helper "F32-FMADD" 'single-float)
              *sbcl-fma-f64* (helper "F64-FMADD" 'double-float)))))
  (refresh-sbcl-fma)
  (pushnew 'refresh-sbcl-fma sb-ext:*init-hooks*))

(defun hardware-fma-available-p ()
  (or *arm64-fma-available-p* *sbcl-fma-available-p*
      (and *native-fma-available-p* (plusp (%native-fma-supported)))))

(defun sbcl-fma-enabled-p ()
  (and (eq *fma-mode* :auto) *sbcl-fma-available-p*))

(defun fma (a b c)
  "Return A*B+C with one nearest-even rounding for same-type finite floats."
  (let* ((single (typep a 'single-float))
         (type (if single 'single-float 'double-float)))
    (unless (and (typep a type) (typep b type) (typep c type))
      (error "FMA needs three floats of the same type"))
    (cond
          #+(or (and sbcl arm64) (and ccl arm64-target) (and ecl aarch64))
          ((and *arm64-fma-available-p* (arm64-fma-enabled-p type))
           (if single (arm64-fma-f32 a b c) (arm64-fma-f64 a b c)))
          ((sbcl-fma-enabled-p)
           (funcall (if single *sbcl-fma-f32* *sbcl-fma-f64*) a b c))
          ((and (member *fma-mode* '(:auto :native)) *native-fma-available-p*)
           (if single (%native-fma-f32 a b c) (%native-fma-f64 a b c)))
          (t (portable-fma a b c)))))

(declaim (inline sigmoid))

(defun sigmoid (value)
  (check-type value (or single-float double-float))
  (let* ((one (float 1 value))
         (z (exp (- (abs value)))))
    (if (minusp value) (/ z (+ one z)) (/ one (+ one z)))))
