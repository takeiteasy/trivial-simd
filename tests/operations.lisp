(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defun available-backends ()
  (append '(:lisp)
          (when simd::*native-available-p* '(:native))
          (when simd::*sbcl-simd-available-p* '(:sbcl))))

(defun values-for (length type offset)
  (make-array length :element-type type
              :initial-contents
              (loop for i below length collect (coerce (+ i offset) type))))

(defun close-enough-p (actual expected type)
  (let ((tolerance (if (eq type 'single-float) 1.0e-5 1.0d-12)))
    (<= (abs (- actual expected)) (* tolerance (max 1 (abs expected))))))

(test backend-is-reported
  (is (member (simd:backend) (available-backends))))

(test arithmetic-across-backends
  (dolist (type '(single-float double-float))
    (dolist (length '(0 1 2 3 4 5 7 8 9 17 257))
      (let ((left (values-for length type 1))
            (right (values-for length type 2)))
        (dolist (operation '(simd:add! simd:subtract! simd:multiply! simd:divide!))
          (let* ((reference (make-array length :element-type type))
                 (function (symbol-function operation)))
            (let ((simd::*backend* :lisp))
              (funcall function reference left right))
            (dolist (backend (available-backends))
              (let ((output (make-array length :element-type type)))
                (let ((simd::*backend* backend))
                  (is (eq output (funcall function output left right))))
                (dotimes (i length)
                  (is (close-enough-p (aref output i) (aref reference i) type)))))))))))

(test destination-may-alias-an-input
  (dolist (backend (available-backends))
    (dolist (type '(single-float double-float))
      (dolist (operation '(simd:add! simd:subtract! simd:multiply! simd:divide!))
        (let* ((left (values-for 9 type 1))
               (right (values-for 9 type 2))
               (reference (make-array 9 :element-type type)))
          (let ((simd::*backend* :lisp))
            (funcall operation reference left right))
          (let ((simd::*backend* backend))
            (funcall operation left left right))
          (dotimes (i 9)
            (is (close-enough-p (aref left i) (aref reference i) type)))
          (let ((left (values-for 9 type 1))
                (right (values-for 9 type 2)))
            (let ((simd::*backend* backend))
              (funcall operation right left right))
            (dotimes (i 9)
              (is (close-enough-p (aref right i) (aref reference i) type)))))))))

(test reductions-across-backends
  (dolist (type '(single-float double-float))
    (dolist (length '(0 1 2 3 4 5 8 9 17 1001))
      (let* ((left (values-for length type 1))
             (right (values-for length type 2))
             (reference-sum (let ((simd::*backend* :lisp)) (simd:sum left)))
             (reference-dot (let ((simd::*backend* :lisp)) (simd:dot left right))))
        (dolist (backend (available-backends))
          (let ((simd::*backend* backend))
            (is (close-enough-p (simd:sum left) reference-sum type))
            (is (close-enough-p (simd:dot left right) reference-dot type))))))))

(test invalid-vectors
  (let ((single (values-for 4 'single-float 1))
        (double (values-for 4 'double-float 1))
        (short (values-for 3 'single-float 1)))
    (signals error (simd:add! single single short))
    (signals error (simd:add! single single double))
    (signals type-error (simd:sum #(1 2 3)))
    (signals error (simd:dot single short))))
