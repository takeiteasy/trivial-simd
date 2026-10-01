(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defparameter *copy-types*
  '(single-float double-float (complex single-float) (complex double-float)
    (signed-byte 8) (unsigned-byte 8) (signed-byte 16) (unsigned-byte 16)
    (signed-byte 32) (unsigned-byte 32) (signed-byte 64) (unsigned-byte 64)))

(defparameter *copy-lengths* '(0 1 3 4 5 17 257 1030 4101))

(defun copy-element (type index)
  (cond ((eq type 'single-float) (coerce (1+ index) 'single-float))
        ((eq type 'double-float) (coerce (1+ index) 'double-float))
        ((equal type '(complex single-float))
         (complex (coerce (1+ index) 'single-float) (coerce (- index) 'single-float)))
        ((equal type '(complex double-float))
         (complex (coerce (1+ index) 'double-float) (coerce (- index) 'double-float)))
        (t (coerce (1+ (mod index 100)) type))))

(defun copy-vector (type length seed)
  (let ((vector (make-array length :element-type type)))
    (dotimes (i length vector)
      (setf (aref vector i) (copy-element type (+ i seed))))))

(test copy-matches-replace-across-backends
  (dolist (type *copy-types*)
    (dolist (length *copy-lengths*)
      (dolist (backend (available-backends))
        (with-backend (backend)
          (let ((source (copy-vector type length 0))
                (destination (copy-vector type length 1000)))
            (is (eq destination (simd:copy! destination source)))
            (is (equalp destination source))))))))

(test copy-slices-and-per-vector-starts
  (dolist (type *copy-types*)
    (dolist (backend (available-backends))
      (with-backend (backend)
        (let* ((source (copy-vector type 20 0))
               (destination (copy-vector type 20 500))
               (expected (copy-seq destination)))
          (replace expected source :start1 3 :end1 8 :start2 3 :end2 8)
          (simd:copy! destination source :start 3 :end 8)
          (is (equalp destination expected))
          (replace expected source :start1 12 :end1 17 :start2 2 :end2 7)
          (simd:copy! destination source :start 0 :end 5
                                         :destination-start 12 :source-start 2)
          (is (equalp destination expected)))))))

(test copy-overlap-reads-source-first
  (dolist (type '(single-float (complex double-float) (unsigned-byte 8)))
    (dolist (shift '(-3 3))
      (let* ((vector (copy-vector type 16 0))
             (original (copy-seq vector))
             (source-start (max 0 (- shift)))
             (destination-start (max 0 shift)))
        (simd:copy! vector vector :start 0 :end 10
                                  :destination-start destination-start
                                  :source-start source-start)
        (replace original (subseq original source-start (+ source-start 10))
                 :start1 destination-start)
        (is (equalp vector original))))))

(test copy-errors
  (let ((floats (make-array 4 :element-type 'single-float))
        (doubles (make-array 4 :element-type 'double-float))
        (short (make-array 3 :element-type 'single-float)))
    (signals error (simd:copy! floats doubles))
    (signals error (simd:copy! floats short))
    (signals error (simd:copy! floats floats :start 2 :end 6))
    (signals error (simd:copy! floats floats :start 3 :end 2))
    (signals error (simd:copy! floats floats :end 2 :source-start 3))))

(test fill-across-backends
  (dolist (type *copy-types*)
    (dolist (length *copy-lengths*)
      (dolist (backend (available-backends))
        (with-backend (backend)
          (let ((vector (copy-vector type length 0))
                (value (copy-element type 41)))
            (is (eq vector (simd:fill! vector value)))
            (is (every (lambda (element) (eql element value)) vector))))))))

(test fill-slice-leaves-the-rest
  (dolist (type *copy-types*)
    (let* ((vector (copy-vector type 20 0))
           (expected (copy-seq vector))
           (value (copy-element type 77)))
      (fill expected value :start 4 :end 11)
      (simd:fill! vector value :start 4 :end 11)
      (is (equalp vector expected)))))

(test fill-rejects-wrong-values
  (signals type-error (simd:fill! (make-array 4 :element-type 'single-float) 1.0d0))
  (signals type-error (simd:fill! (make-array 4 :element-type 'single-float) 1))
  (signals type-error (simd:fill! (make-array 4 :element-type '(unsigned-byte 8)) 256))
  (signals type-error (simd:fill! (make-array 4 :element-type '(signed-byte 8)) -129))
  (signals type-error (simd:fill! (make-array 4 :element-type '(complex single-float)) 1.0f0))
  (signals error (simd:fill! (make-array 4 :element-type 'single-float) 1.0f0 :end 5)))

(test fill-keeps-signed-zero
  (let ((vector (make-array 3 :element-type 'double-float :initial-element 1d0)))
    (simd:fill! vector -0d0)
    (is (every (lambda (element) (minusp (float-sign element))) vector))))

(test swap-across-backends
  (dolist (type *copy-types*)
    (dolist (length *copy-lengths*)
      (dolist (backend (available-backends))
        (with-backend (backend)
          (let* ((x (copy-vector type length 0))
                 (y (copy-vector type length 1000))
                 (x-original (copy-seq x))
                 (y-original (copy-seq y)))
            (multiple-value-bind (first second) (simd:swap! x y)
              (is (eq first x))
              (is (eq second y)))
            (is (equalp x y-original))
            (is (equalp y x-original))))))))

(test swap-slices-and-per-vector-starts
  (dolist (type *copy-types*)
    (dolist (backend (available-backends))
      (with-backend (backend)
        (let* ((x (copy-vector type 40 0))
               (y (copy-vector type 40 500))
               (x-expected (copy-seq x))
               (y-expected (copy-seq y)))
          (replace x-expected y :start1 5 :end1 22 :start2 9)
          (replace y-expected x :start1 9 :end1 26 :start2 5)
          (simd:swap! x y :start 0 :end 17 :x-start 5 :y-start 9)
          (is (equalp x x-expected))
          (is (equalp y y-expected)))))))

(test swap-of-one-vector-with-itself
  (dolist (backend (available-backends))
    (with-backend (backend)
      (let* ((vector (copy-vector 'double-float 9 0))
             (original (copy-seq vector)))
        (simd:swap! vector vector)
        (is (equalp vector original))
        (simd:swap! vector vector :start 2 :end 6)
        (is (equalp vector original))))))

(test swap-errors
  (let ((floats (make-array 4 :element-type 'single-float))
        (doubles (make-array 4 :element-type 'double-float)))
    (signals error (simd:swap! floats doubles))
    (signals error (simd:swap! floats floats :start 2 :end 6))
    (signals error (simd:swap! floats floats :end 2 :y-start 3))))

(test fill-and-swap-large-unaligned-slices
  (dolist (type *copy-types*)
    (dolist (backend (available-backends))
      (with-backend (backend)
        (let* ((x (copy-vector type 5000 0))
               (y (copy-vector type 5000 3000))
               (x-expected (copy-seq x))
               (y-expected (copy-seq y))
               (value (copy-element type 9)))
          (fill x-expected value :start 3 :end 4500)
          (simd:fill! x value :start 3 :end 4500)
          (is (equalp x x-expected))
          (replace y-expected x :start1 7 :end1 4400 :start2 11)
          (replace x-expected y :start1 11 :end1 4404 :start2 7)
          (simd:swap! x y :start 0 :end 4393 :x-start 11 :y-start 7)
          (is (equalp x x-expected))
          (is (equalp y y-expected)))))))
