(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defparameter *real-types*
  '(single-float double-float (signed-byte 8) (unsigned-byte 8) (signed-byte 16)
    (unsigned-byte 16) (signed-byte 32) (unsigned-byte 32) (signed-byte 64)
    (unsigned-byte 64)))

(defparameter *reduction-lengths* '(0 1 2 3 4 5 7 31 32 33 34 63 64 65 100 257))

(defun small-values (length type)
  "Deterministic values drawn from a small range so ties are common."
  (let ((state 12345))
    (make-array length :element-type type
                :initial-contents
                (loop repeat length
                      do (setf state (mod (+ (* state 1103515245) 12345) 2147483648))
                      collect (let ((value (mod (ash state -8) 9)))
                                (coerce (if (subtypep type 'unsigned-byte)
                                            value
                                            (- value 4))
                                        type))))))

(defun reference-argext (vector start end maximum-p)
  (when (< start end)
    (let ((best start))
      (loop for i from (1+ start) below end
            do (when (if maximum-p
                         (> (aref vector i) (aref vector best))
                         (< (aref vector i) (aref vector best)))
                 (setf best i)))
      best)))

(test arg-reductions-match-reference
  (dolist (type *real-types*)
    (dolist (length *reduction-lengths*)
      (let ((vector (small-values length type)))
        (dolist (backend (available-backends))
          (with-backend (backend)
            (is (eql (reference-argext vector 0 length nil) (simd:argmin vector)))
            (is (eql (reference-argext vector 0 length t) (simd:argmax vector)))
            (let ((index (reference-argext vector 0 length nil)))
              (is (eql (and index (aref vector index)) (simd:minimum vector))))
            (let ((index (reference-argext vector 0 length t)))
              (is (eql (and index (aref vector index)) (simd:maximum vector))))))))))

(test arg-reductions-use-absolute-indexes
  (dolist (type '(single-float double-float (signed-byte 16)))
    (let ((vector (small-values 120 type)))
      (dolist (backend (available-backends))
        (with-backend (backend)
          (is (eql (reference-argext vector 5 70 nil) (simd:argmin vector :start 5 :end 70)))
          (is (eql (reference-argext vector 5 70 t) (simd:argmax vector :start 5 :end 70)))
          (is (eql (reference-argext vector 40 100 t)
                   (simd:argmax vector :start 2 :end 62 :input-start 40)))
          (is (null (simd:argmin vector :start 7 :end 7)))
          (is (null (simd:maximum vector :start 7 :end 7 :input-start 90))))))))

(defun signed-zero-vector (type first second)
  (let ((vector (make-array 80 :element-type type :initial-element (coerce -1 type))))
    (setf (aref vector 40) (coerce first type)
          (aref vector 41) (coerce second type))
    vector))

(test arg-reductions-keep-the-first-signed-zero
  (dolist (type '(single-float double-float))
    (dolist (backend (available-backends))
      (with-backend (backend)
        (let ((negative-first (signed-zero-vector type -0d0 0d0))
              (positive-first (signed-zero-vector type 0d0 -0d0)))
          (is (= 40 (simd:argmax negative-first)))
          (is (minusp (float-sign (simd:maximum negative-first))))
          (is (= 40 (simd:argmax positive-first)))
          (is (plusp (float-sign (simd:maximum positive-first)))))
        (let ((negative-first (signed-zero-vector type -0d0 0d0))
              (positive-first (signed-zero-vector type 0d0 -0d0)))
          (map-into negative-first #'- negative-first)
          (map-into positive-first #'- positive-first)
          (is (= 40 (simd:argmin positive-first)))
          (is (minusp (float-sign (simd:minimum positive-first))))
          (is (= 40 (simd:argmin negative-first)))
          (is (plusp (float-sign (simd:minimum negative-first)))))))))

(defun wrap-to (type value)
  (let* ((bits (second type)) (modulus (ash 1 bits)) (value (mod value modulus)))
    (if (and (eq (first type) 'signed-byte) (>= value (ash modulus -1)))
        (- value modulus)
        value)))

(test asum-matches-reference
  (dolist (type *real-types*)
    (dolist (length *reduction-lengths*)
      (let* ((vector (small-values length type))
             (expected (let ((sum (coerce 0 type)))
                         (loop for value across vector
                               do (setf sum (if (integerp value)
                                                (wrap-to type (+ sum (abs value)))
                                                (+ sum (abs value)))))
                         sum)))
        (dolist (backend (available-backends))
          (with-backend (backend)
            (is (eql expected (simd:asum vector)))))))))

(test asum-wraps-integers
  (dolist (backend (available-backends))
    (with-backend (backend)
      (is (= 5 (simd:asum (make-array 3 :element-type '(signed-byte 8)
                                        :initial-contents '(-128 -128 5)))))
      (is (= 44 (simd:asum (make-array 3 :element-type '(unsigned-byte 8)
                                         :initial-contents '(200 50 50))))))))

(test complex-asum-uses-the-modulus
  (dolist (type '((complex single-float) (complex double-float)))
    (let ((vector (make-array 3 :element-type type
                                :initial-contents
                                (list (complex (coerce 3 (second type)) (coerce 4 (second type)))
                                      (complex (coerce -6 (second type)) (coerce 8 (second type)))
                                      (complex (coerce 0 (second type)) (coerce 0 (second type)))))))
      (dolist (backend (available-backends))
        (with-backend (backend)
          (is (close-enough-p (simd:asum vector) 15 (second type))))))))

(test nrm2-matches-reference
  (dolist (type '(single-float double-float))
    (dolist (length *reduction-lengths*)
      (let* ((vector (small-values length type))
             (expected (sqrt (loop for value across vector sum (* (coerce value 'double-float)
                                                                    (coerce value 'double-float))))))
        (dolist (backend (available-backends))
          (with-backend (backend)
            (is (close-enough-p (simd:nrm2 vector) expected type))))))))

(test nrm2-avoids-overflow-and-underflow
  (dolist (backend (available-backends))
    (with-backend (backend)
      (let ((large (make-array 4 :element-type 'double-float :initial-element 1d200))
            (small (make-array 4 :element-type 'double-float :initial-element 1d-200))
            (single (make-array 4 :element-type 'single-float :initial-element 1f30)))
        (is (close-enough-p (simd:nrm2 large) 2d200 'double-float))
        (is (close-enough-p (simd:nrm2 small) 2d-200 'double-float))
        (is (close-enough-p (simd:nrm2 single) 2f30 'single-float))
        (is (zerop (simd:nrm2 (make-array 5 :element-type 'double-float :initial-element 0d0))))
        (is (zerop (simd:nrm2 (make-array 0 :element-type 'single-float))))))))

(test nrm2-mixed-magnitudes
  (let ((vector (make-array 3 :element-type 'double-float
                             :initial-contents '(1d200 1d-200 0d0))))
    (dolist (backend (available-backends))
      (with-backend (backend)
        (is (= 1d200 (simd:nrm2 vector)))))))

(test complex-nrm2-matches-reference
  (dolist (type '((complex single-float) (complex double-float)))
    (let* ((real (second type))
           (vector (make-array 3 :element-type type
                                 :initial-contents
                                 (list (complex (coerce 3 real) (coerce 4 real))
                                       (complex (coerce 0 real) (coerce 12 real))
                                       (complex (coerce 0 real) (coerce 0 real))))))
      (dolist (backend (available-backends))
        (with-backend (backend)
          (is (close-enough-p (simd:nrm2 vector) 13 real))))
      (when (eq real 'double-float)
        (let ((large (make-array 2 :element-type type :initial-element (complex 1d200 1d200))))
          (is (close-enough-p (simd:nrm2 large) (* 2d200 (sqrt 1d0)) real)))))))

(test accumulate-widens-single-float-reductions
  (let ((left (make-array 1001 :element-type 'single-float :initial-element 1f0))
        (right (make-array 1001 :element-type 'single-float :initial-element 1f0))
        (c32 (make-array 1001 :element-type '(complex single-float)
                              :initial-element #C(1f0 0f0))))
    ;; CCL x86-64 drops a constant-index SETF AREF on complex arrays (#123).
    (setf (aref left 0) 16777216f0
          (row-major-aref c32 0) #C(16777216f0 0f0))
    (dolist (backend (available-backends))
      (with-backend (backend)
        (is (eql 16778216d0 (simd:sum left :accumulate :f64)))
        (is (eql 16778216d0 (simd:dot left right :accumulate :f64)))
        (is (eql 16778216d0 (simd:asum left :accumulate :f64)))
        (is (eql #C(16778216d0 0d0) (simd:sum c32 :accumulate :f64)))
        (is (typep (simd:nrm2 left :accumulate :f64) 'double-float))
        (is (typep (simd:nrm2 left) 'single-float))))))

(test accumulate-leaves-double-float-and-rejects-bad-requests
  (let ((doubles (small-values 10 'double-float))
        (integers (small-values 10 '(signed-byte 32)))
        (singles (small-values 10 'single-float)))
    (dolist (backend (available-backends))
      (with-backend (backend)
        (is (eql (simd:sum doubles) (simd:sum doubles :accumulate :f64)))
        (is (eql (simd:asum doubles) (simd:asum doubles :accumulate :f64)))
        (signals error (simd:sum integers :accumulate :f64))
        (signals error (simd:asum integers :accumulate :f64))
        (signals error (simd:sum singles :accumulate :f32))
        (signals error (simd:nrm2 integers))
        (signals error (simd:argmin (make-array 2 :element-type '(complex single-float)
                                                  :initial-element #C(1f0 1f0))))
        (signals error (simd:maximum #(1 2 3)))))))

(test blas-unit-stride-reductions-match-the-strided-path
  (dolist (backend (available-backends))
    (with-backend (backend)
      (let ((x (small-values 100 'single-float))
            (y (small-values 100 'double-float)))
        (setf (aref x 50) -9f0 (aref y 50) -9d0 (aref y 51) 9d0)
        (is (= 45 (trivial-simd/blas:isamax 90 x 1 :x-offset 5)))
        (is (= 45 (trivial-simd/blas:idamax 90 y 1 :x-offset 5)))
        (is (= 0 (trivial-simd/blas:idamax 0 y 1)))
        (is (close-enough-p (trivial-simd/blas:sasum 90 x 1 :x-offset 5)
                            (loop for i from 5 below 95 sum (abs (aref x i)))
                            'single-float))
        (is (close-enough-p (trivial-simd/blas:dnrm2 90 y 1 :x-offset 5)
                            (sqrt (loop for i from 5 below 95 sum (expt (aref y i) 2)))
                            'double-float))
        (is (close-enough-p (trivial-simd/blas:snrm2 45 x 2 :x-offset 5)
                            (sqrt (loop for i from 5 below 95 by 2 sum (expt (aref x i) 2)))
                            'single-float))))))
