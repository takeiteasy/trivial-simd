(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defun available-backends ()
  (append '(:lisp)
          (when simd::*native-available-p* '(:native :native-copy))
          (when simd::*sbcl-simd-available-p* '(:sbcl))))

(defmacro with-backend ((backend) &body body)
  `(let* ((backend ,backend)
          (simd::*backend* (if (eq backend :native-copy) :native backend))
          (simd::*native-array-access*
            (if (eq backend :native-copy) :copy simd::*native-array-access*)))
     ,@body))

(defun values-for (length type offset)
  (make-array length :element-type type
              :initial-contents
              (loop for i below length collect (coerce (+ i offset) type))))

(defun close-enough-p (actual expected type)
  (let ((tolerance (if (eq type 'single-float) 1.0e-5 1.0d-12)))
    (<= (abs (- actual expected)) (* tolerance (max 1 (abs expected))))))

(test backend-is-reported
  (is (member (simd:backend) (remove :native-copy (available-backends)))))

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
                (with-backend (backend)
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
          (with-backend (backend)
            (funcall operation left left right))
          (dotimes (i 9)
            (is (close-enough-p (aref left i) (aref reference i) type)))
          (let ((left (values-for 9 type 1))
                (right (values-for 9 type 2)))
            (with-backend (backend)
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
          (with-backend (backend)
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

(defun expected-slice (operation type count left-offset right-offset left right)
  (let ((function (ecase operation
                    (simd:add! #'+) (simd:subtract! #'-)
                    (simd:multiply! #'*) (simd:divide! #'/))))
    (loop for i below count
          collect (funcall function
                           (aref left (+ left-offset i))
                           (aref right (+ right-offset i))))))

(test slices-across-backends
  (dolist (backend (available-backends))
    (dolist (type '(single-float double-float))
      (dolist (operation '(simd:add! simd:subtract! simd:multiply! simd:divide!))
        (dolist (case '((3 20 nil nil nil) (0 9 5 0 11) (1 6 7 3 2) (0 0 4 4 4)))
          (destructuring-bind (start end destination-start left-start right-start) case
            (let* ((left (values-for 40 type 1))
                   (right (values-for 40 type 2))
                   (output (values-for 40 type 100))
                   (before (copy-seq output))
                   (count (- end start))
                   (d (or destination-start start))
                   (l (or left-start start))
                   (r (or right-start start))
                   (expected (expected-slice operation type count l r left right)))
              (with-backend (backend)
                (is (eq output (funcall operation output left right
                                        :start start :end end
                                        :destination-start destination-start
                                        :left-start left-start
                                        :right-start right-start))))
              (dotimes (i 40)
                (if (<= d i (+ d count -1))
                    (is (close-enough-p (aref output i) (elt expected (- i d)) type))
                    (is (= (aref output i) (aref before i))))))))))))

(test slice-reductions-across-backends
  (dolist (backend (available-backends))
    (dolist (type '(single-float double-float))
      (let ((left (values-for 40 type 1))
            (right (values-for 40 type 2)))
        (with-backend (backend)
          (is (close-enough-p (simd:sum left :start 3 :end 20)
                              (loop for i from 3 below 20 sum (aref left i)) type))
          (is (close-enough-p (simd:sum left :start 2 :end 11 :input-start 20)
                              (loop for i from 20 below 29 sum (aref left i)) type))
          (is (close-enough-p (simd:dot left right :start 1 :end 10 :right-start 30)
                              (loop for i from 1 below 10
                                    sum (* (aref left i) (aref right (+ 29 i)))) type))
          (is (zerop (simd:sum left :start 5 :end 5))))))))

(test invalid-slices
  (let ((a (values-for 8 'single-float 1))
        (b (values-for 8 'single-float 1)))
    (signals error (simd:add! a a b :start 4 :end 2))
    (signals error (simd:add! a a b :start -1 :end 4))
    (signals error (simd:add! a a b :start 0 :end 9))
    (signals error (simd:add! a a b :start 0 :end 4 :right-start 5))
    (signals error (simd:add! a a b :start 0 :end 4 :left-start -1))
    (signals error (simd:sum a :start 0 :end 4 :input-start 5))
    (signals error (simd:add! a a (values-for 9 'single-float 1)))))
