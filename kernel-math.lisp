(in-package #:trivial-simd)

(defun kernel-sqrt (value)
  (when (minusp value) (error "Negative kernel square root operand: ~S" value))
  (sqrt value))

(defun kernel-min (left right)
  (if (<= left right) left right))

(defun kernel-max (left right)
  (if (>= left right) left right))

;; TODO: exact integer FMA is costly; select correctly rounded primitives (#54).
(defun fma (a b c)
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
