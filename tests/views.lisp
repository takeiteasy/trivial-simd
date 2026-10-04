(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

;; Every check calls an operation on Lisp vectors and again on vector views of
;; foreign copies of them (all views, then alternating views and vectors), on each
;; backend, and compares the results and the final memory. Small integer data keeps
;; float sums exact whatever the block order.

(defvar *view-memory* nil)

(defmacro with-view-memory (&body body)
  `(let ((*view-memory* nil))
     (unwind-protect (progn ,@body)
       (mapc #'cffi:foreign-free *view-memory*))))

(defun view-memory (bytes)
  (let ((pointer (cffi:foreign-alloc :uint8 :count (max 1 bytes))))
    (push pointer *view-memory*)
    pointer))

(defun mirror (vector)
  "Return a vector view of a foreign copy of VECTOR."
  (let* ((type (simd::vector-type vector))
         (view (simd:make-vector-view
                (view-memory (* (simd::vector-element-size type) (length vector)))
                type (length vector))))
    (simd::stage-out vector view 0 1 (length vector))
    view))

(defun contents (vector)
  "A Lisp copy of a view's elements, or VECTOR itself."
  (if (simd:vector-view-p vector)
      (let* ((length (simd:vector-view-length vector))
             (copy (simd::staging-buffer vector length)))
        (simd::stage-in vector 0 1 length copy)
        copy)
      vector))

(defun lisp-vector-p (value)
  (and (typep value '(simple-array * (*))) (not (stringp value))))

(defun same-results-p (actual expected)
  (equalp (mapcar #'contents actual) (mapcar #'contents expected)))

(defun floats-close-p (a b)
  "True when floats A and B agree to a relative tolerance. The ratio avoids a
subnormal difference, which traps on Lisps that enable underflow traps."
  (let ((tolerance (if (typep b 'single-float) 1f-5 1d-12)))
    (cond ((= a b) t)
          ((or (zerop a) (zerop b)) (< (max (abs a) (abs b)) tolerance))
          (t (handler-case (<= (abs (- (/ a b) 1)) tolerance)
               (arithmetic-error () nil))))))

(defun numerically-close-p (actual expected)
  "Like SAME-RESULTS-P, but floats need only agree to a few units in the last place."
  (labels ((close-p (a b)
             (cond ((and (floatp a) (floatp b)) (floats-close-p a b))
                   ((and (complexp a) (complexp b))
                    (and (close-p (realpart a) (realpart b))
                         (close-p (imagpart a) (imagpart b))))
                   ((and (lisp-vector-p a) (lisp-vector-p b))
                    (and (= (length a) (length b)) (every #'close-p a b)))
                   (t (equalp a b)))))
    (every #'close-p (mapcar #'contents actual) (mapcar #'contents expected))))

(defvar *view-case* nil "Description of the current check, for failure messages.")

(defun check-views (function arguments &key (test #'same-results-p))
  "Compare FUNCTION on ARGUMENTS with FUNCTION on views of copies of their vectors."
  (let* ((reference (mapcar (lambda (value) (if (lisp-vector-p value) (copy-seq value) value))
                            arguments))
         (expected (multiple-value-list (apply function reference))))
    (dolist (mode '(:views :odd :even))
      (with-view-memory
        (let* ((index -1)
               (actual-arguments
                 (mapcar (lambda (value)
                           (cond ((not (lisp-vector-p value)) value)
                                 ((or (eq mode :views)
                                      (eq (oddp (incf index)) (eq mode :odd)))
                                  (mirror value))
                                 (t (copy-seq value))))
                         arguments))
               (actual (multiple-value-list (apply function actual-arguments))))
          (is (funcall test actual expected)
              "~A (~A, ~A, block ~D): results ~S, expected ~S"
              *view-case* mode simd::*backend* simd::*view-block-size*
              (mapcar #'contents actual) expected)
          (is (funcall test (mapcar #'contents actual-arguments) reference)
              "~A (~A, ~A, block ~D): memory differs"
              *view-case* mode simd::*backend* simd::*view-block-size*))))))

(defmacro do-view-backends (&body body)
  "Run BODY on every backend with a block size that splits slices and the default."
  `(dolist (backend (available-backends))
     (with-backend (backend)
       (dolist (simd::*view-block-size* '(5 4096))
         ,@body))))

(defun lisp-type (key) (second (simd::numeric-type key)))

(defun blank (type length)
  "A zeroed vector, so that elements an operation skips compare equal."
  (make-array length :element-type type :initial-element (coerce 0 type)))

(defun view-element (key index &key (signed t))
  "A small value of type KEY; negative for odd INDEX when SIGNED and the type allows."
  (let* ((magnitude (1+ (mod (* 7 index) 11)))
         (value (if (and signed (oddp index)
                         (not (member key '(:u8 :u16 :u32 :u64))))
                    (- magnitude)
                    magnitude)))
    (case key
      (:c32 (complex (coerce value 'single-float) (coerce (- (mod index 5) 2) 'single-float)))
      (:c64 (complex (coerce value 'double-float) (coerce (- (mod index 5) 2) 'double-float)))
      (t (coerce value (lisp-type key))))))

(defun sample-vector (key length &key (seed 0) (signed t))
  (let ((vector (make-array length :element-type (lisp-type key))))
    (dotimes (i length vector)
      (setf (aref vector i) (view-element key (+ i seed) :signed signed)))))

(defun scalar-for (key value)
  "VALUE as an element of type KEY; negative values wrap for unsigned types."
  (if (member key '(:u8 :u16 :u32 :u64))
      (mod value (ash 1 (fourth (simd::numeric-type key))))
      (coerce value (lisp-type key))))

(defparameter *view-lengths* '(0 1 13 64))
(defparameter *view-real-types* '(:f32 :f64 :s8 :u8 :s16 :u32 :s64))
(defparameter *view-types* (append *view-real-types* '(:c32 :c64)))

(test vector-view-construction
  (with-view-memory
    (let* ((pointer (view-memory 64))
           (view (simd:make-vector-view pointer :f64 4 :offset 2)))
      (is (simd:vector-view-p view))
      (is (eq :f64 (simd:vector-view-type view)))
      (is (= 4 (simd:vector-view-length view)))
      (is (= (+ 16 (cffi:pointer-address pointer))
             (cffi:pointer-address (simd:vector-view-pointer view))))
      (is (not (simd:vector-view-p (make-array 3 :element-type 'double-float))))
      (is (search "VECTOR-VIEW :F64 4" (string-upcase (prin1-to-string view))))
      (is (simd:vector-view-p (simd:make-vector-view (cffi:null-pointer) :u8 0)))
      (signals type-error (simd:make-vector-view 12 :f32 1))
      (signals error (simd:make-vector-view pointer :f16 1))
      (signals error (simd:make-vector-view pointer :f32 -1))
      (signals error (simd:make-vector-view pointer :f32 1 :offset -1))
      (signals error (simd:make-vector-view (cffi:null-pointer) :f32 1)))))

(test vector-view-binary-operations
  (do-view-backends
    (dolist (key *view-types*)
      (dolist (length *view-lengths*)
        (let ((left (sample-vector key length :seed 1))
              (right (sample-vector key length :seed 2 :signed nil))
              (scalar (scalar-for key 3)))
          (dolist (operation '(simd:add! simd:subtract! simd:multiply! simd:divide!))
            (let ((*view-case* (list operation key length)))
              (check-views operation (list (sample-vector key length) left right))
              (check-views operation (list (sample-vector key length) left scalar))
              (check-views operation (list (sample-vector key length) scalar right))
              (unless (< length 13)
                (check-views (lambda (d l r) (funcall operation d l r :start 2 :end 11))
                             (list (sample-vector key length) left right))
                (check-views (lambda (d l r)
                               (funcall operation d l r :end 4 :left-start 3 :right-stride 3
                                        :destination-stride -2 :destination-start 9))
                             (list (sample-vector key length) left right))))))))))

(test vector-view-scale-axpy-fma
  (do-view-backends
    (dolist (key *view-types*)
      (dolist (length *view-lengths*)
        (let ((x (sample-vector key length :seed 3))
              (*view-case* (list 'scale-axpy key length)))
          (check-views (lambda (x) (simd:scale! x (scalar-for key 2))) (list x))
          (check-views (lambda (y x) (simd:axpy! y (scalar-for key -2) x))
                       (list (sample-vector key length) x))
          (unless (< length 13)
            (check-views (lambda (x) (simd:scale! x (scalar-for key 3) :start 1 :end 4 :stride 4))
                         (list (sample-vector key length)))
            (check-views (lambda (y x) (simd:axpy! y (scalar-for key 3) x :end 6
                                                   :x-stride 2 :y-stride -1 :y-start 7))
                         (list (sample-vector key length) x)))
          (when (member key '(:f32 :f64))
            (check-views #'simd:fma!
                         (list (sample-vector key length) x (sample-vector key length :seed 4)
                               (sample-vector key length :seed 5)))
            (check-views (lambda (d x z) (simd:fma! d x (scalar-for key 2) z))
                         (list (sample-vector key length) x (sample-vector key length :seed 5)))))))))

(test vector-view-extended-operations
  (do-view-backends
    (dolist (key *view-types*)
      (dolist (length *view-lengths*)
        (let* ((input (sample-vector key length :seed 6))
               (positive (sample-vector key length :seed 7 :signed nil))
               (realp (not (simd::complex-type-p key)))
               (floatp (member key '(:f32 :f64 :c32 :c64)))
               (*view-case* (list 'extended key length)))
          (check-views #'simd:negate! (list (sample-vector key length) input))
          (when realp (check-views #'simd:abs! (list (sample-vector key length) input)))
          (when floatp
            (check-views #'simd:sqrt! (list (sample-vector key length) positive))
            (check-views #'simd:reciprocal! (list (sample-vector key length) positive)))
          (when realp
            (check-views #'simd:min! (list (sample-vector key length) input positive))
            (check-views #'simd:max! (list (sample-vector key length) input (scalar-for key 2)))
            (check-views (lambda (d i) (simd:clamp! d i (scalar-for key 2) (scalar-for key 9)))
                         (list (sample-vector key length) input))
            (check-views #'simd:clamp!
                         (list (sample-vector key length) input
                               (simd:fill! (sample-vector key length) (scalar-for key 1)) positive)))
          (dolist (operator (if realp '(:eq :ne :lt :le :gt :ge) '(:eq :ne)))
            (check-views (lambda (mask left right) (simd:compare! mask operator left right))
                         (list (blank '(unsigned-byte 8) length)
                               input positive)))
          (let ((mask (blank '(unsigned-byte 8) length)))
            (dotimes (i length) (setf (aref mask i) (if (zerop (mod i 3)) 0 (mod i 7))))
            (check-views #'simd:select! (list (sample-vector key length) mask input positive))
            (check-views (lambda (d m y) (simd:select! d m y (scalar-for key 9)))
                         (list (sample-vector key length) mask input))
            (check-views #'simd:count (list mask))
            (check-views #'simd:any (list mask))
            (check-views #'simd:all (list mask))
            (when (>= length 13)
              (check-views (lambda (m) (simd:count m :start 1 :end 5 :stride 2)) (list mask)))))))))

(test vector-view-conversions
  (do-view-backends
    (dolist (length *view-lengths*)
      (dolist (pair '((:f64 :f32) (:f32 :f64) (:s16 :f32) (:f32 :s16) (:u8 :s64)
                      (:s64 :f64) (:f64 :u8) (:c32 :c64) (:c32 :f32)))
        (destructuring-bind (to from) pair
          (let ((*view-case* (list 'convert! pair length))
                (input (sample-vector from length :seed 8)))
            (when (member from '(:f32 :f64))
              (dotimes (i length) (setf (aref input i) (/ (aref input i) (coerce 4 (lisp-type from))))))
            (dolist (rounding '(:nearest-even :truncate :floor :ceiling))
              (check-views (lambda (d i) (simd:convert! d i :rounding rounding))
                           (list (blank (lisp-type to) length) input)))
            (unless (< length 13)
              (check-views (lambda (d i) (simd:convert! d i :end 5 :destination-stride 2
                                                        :input-start 12 :input-stride -2))
                           (list (blank (lisp-type to) length) input))))))
      (let ((floats (sample-vector :f32 length :seed 9))
            (*view-case* (list 'convert-encoded length)))
        (dotimes (i length) (setf (aref floats i) (/ (aref floats i) 3)))
        (dolist (encoding '(:bf16 :f16))
          (dolist (modes (remove-duplicates (list simd::*native-float-encoding-rounding-modes* #b0011)))
            (let ((simd::*native-float-encoding-rounding-modes* modes))
              (dolist (rounding '(:nearest-even :truncate :floor :ceiling))
                (check-views (lambda (d i) (simd:convert! d i :destination-encoding encoding
                                                              :rounding rounding))
                             (list (blank '(unsigned-byte 16) length) floats)))))
          (let ((encoded (blank '(unsigned-byte 16) length)))
            (simd:convert! encoded floats :destination-encoding encoding)
            (check-views (lambda (d i) (simd:convert! d i :input-encoding encoding))
                         (list (blank 'single-float length) encoded))))))))

(test vector-view-reductions
  (do-view-backends
    (dolist (key *view-types*)
      (dolist (length *view-lengths*)
        (let* ((left (sample-vector key length :seed 10))
               (right (sample-vector key length :seed 11))
               (realp (not (simd::complex-type-p key)))
               (*view-case* (list 'reductions key length)))
          (check-views #'simd:sum (list left))
          (check-views #'simd:dot (list left right))
          ;; Complex moduli are inexact, so their sums depend on the block order.
          (check-views #'simd:asum (list left)
                       :test (if realp #'same-results-p #'numerically-close-p))
          (when (simd::complex-type-p key) (check-views #'simd:dotc (list left right)))
          (when (member key '(:f32 :c32))
            (check-views (lambda (x) (simd:sum x :accumulate :f64)) (list left))
            (check-views (lambda (x y) (simd:dot x y :accumulate :f64)) (list left right))
            (check-views (lambda (x) (simd:asum x :accumulate :f64)) (list left)
                         :test #'numerically-close-p))
          (unless (simd::integer-type-p key)
            (check-views #'simd:nrm2 (list left) :test #'numerically-close-p))
          (when realp
            (dolist (function (list #'simd:minimum #'simd:maximum #'simd:argmin #'simd:argmax))
              (check-views function (list left))
              (when (>= length 13)
                (check-views (lambda (x) (funcall function x :start 2 :end 6 :stride 2))
                             (list left))
                (check-views (lambda (x) (funcall function x :start 1 :end 4 :input-start 12
                                                  :stride -3))
                             (list left)))))
          (when (>= length 13)
            (check-views (lambda (x y) (simd:dot x y :end 4 :left-stride 3 :right-start 12
                                                 :right-stride -1))
                         (list left right))
            (check-views (lambda (x) (simd:sum x :start 3 :end 7 :stride 2)) (list left))))))))

(test vector-view-ties-and-extremes
  ;; The first of equal extremes wins across block boundaries too.
  (do-view-backends
    (let ((data (make-array 23 :element-type 'double-float :initial-element 1d0)))
      (setf (aref data 4) 7d0 (aref data 9) 7d0 (aref data 17) 7d0
            (aref data 6) -3d0 (aref data 21) -3d0)
      (let ((*view-case* 'ties))
        (dolist (function (list #'simd:argmax #'simd:argmin #'simd:maximum #'simd:minimum))
          (check-views function (list data))
          (check-views (lambda (x) (funcall function x :start 5)) (list data))))
      (with-view-memory
        (let ((view (mirror data)))
          (is (= 4 (simd:argmax view)))
          (is (= 9 (simd:argmax view :start 5)))
          (is (= 6 (simd:argmin view)))
          (is (null (simd:argmax view :start 23)))
          (is (null (simd:maximum view :start 23))))))))

(test vector-view-nrm2-scaling
  (do-view-backends
    (dolist (scale '(1d-300 1d300))
      (let ((data (make-array 17 :element-type 'double-float)))
        (dotimes (i 17) (setf (aref data i) (* scale (1+ (mod i 4)))))
        (let ((*view-case* (list 'nrm2 scale)))
          (check-views #'simd:nrm2 (list data) :test #'numerically-close-p))))))

(test vector-view-copies
  (do-view-backends
    (dolist (key *view-types*)
      (dolist (length *view-lengths*)
        (let ((source (sample-vector key length :seed 12))
              (*view-case* (list 'copies key length)))
          (check-views #'simd:copy! (list (sample-vector key length) source))
          (check-views (lambda (d) (simd:fill! d (scalar-for key 5))) (list (sample-vector key length)))
          (check-views #'simd:swap! (list (sample-vector key length) source))
          (when (>= length 13)
            (check-views (lambda (d s) (simd:copy! d s :end 6 :source-stride 2
                                                   :destination-start 12 :destination-stride -1))
                         (list (sample-vector key length) source))
            (check-views (lambda (d) (simd:fill! d (scalar-for key 2) :start 1 :end 5 :stride 3))
                         (list (sample-vector key length)))
            (check-views (lambda (x y) (simd:swap! x y :end 4 :x-stride 3 :y-start 9))
                         (list (sample-vector key length) source))))))))

(defun check-overlapping-copy (key length destination-start source-start count
                               &key (destination-stride 1) (source-stride 1))
  "Copy within one foreign buffer through two views and compare with a Lisp vector."
  (let ((expected (sample-vector key length :seed 13)))
    (with-view-memory
      (let* ((first (mirror expected))
             (second (simd:make-vector-view (simd:vector-view-pointer first) key length)))
        (simd:copy! expected (copy-seq expected) :end count
                    :destination-start destination-start :source-start source-start
                    :destination-stride destination-stride :source-stride source-stride)
        (simd:copy! second first :end count
                    :destination-start destination-start :source-start source-start
                    :destination-stride destination-stride :source-stride source-stride)
        (is (equalp expected (contents first))
            "Overlapping ~A copy ~A <- ~A (~A, block ~D)"
            key destination-start source-start simd::*backend* simd::*view-block-size*)))))

(test vector-view-overlapping-copies
  (do-view-backends
    (dolist (key '(:f32 :c64 :u8))
      (check-overlapping-copy key 40 3 0 30)
      (check-overlapping-copy key 40 0 3 30)
      (check-overlapping-copy key 40 5 5 30)
      (check-overlapping-copy key 40 25 30 10 :destination-stride -2 :source-stride -2)
      (check-overlapping-copy key 40 30 25 10 :destination-stride -2 :source-stride -2)
      (check-overlapping-copy key 40 0 1 13 :destination-stride 3 :source-stride 2)
      (check-overlapping-copy key 40 39 0 20 :destination-stride -1))))

(test vector-view-mixed-types-and-errors
  (with-view-memory
    (let* ((f32 (mirror (sample-vector :f32 8)))
           (f64 (mirror (sample-vector :f64 8)))
           (short (mirror (sample-vector :f32 5)))
           (array (sample-vector :f32 8)))
      (signals error (simd:add! f32 f32 f64))
      (signals error (simd:add! f32 f32 short))
      (signals error (simd:dot f32 short))
      (signals error (simd:dot f32 array :end 9))
      (signals error (simd:sum f32 :start 3 :end 2))
      (signals error (simd:sum f32 :stride 2))
      (signals error (simd:axpy! f32 f32 f32))
      (signals error (simd:scale! f32 f32))
      (signals error (simd:compare! f32 :lt f32 f32))
      (signals error (simd:convert! f32 short))
      (signals error (simd:copy! f32 short :end 6))
      (signals type-error (simd:add! f32 f32 1d0))
      (is (= (simd:dot array array) (simd:dot f32 array) (simd:dot array f32)))
      (is (eq f32 (simd:add! f32 array f32)))
      (is (eq f32 (simd:convert! f32 f64))))))

(test vector-view-direct-native-access
  ;; A NIL block size makes any staging fail, so these calls must use the view's
  ;; memory directly.
  (when simd::*native-available-p*
    (with-backend (:native)
      (with-view-memory
        (let* ((x (mirror (sample-vector :f32 2000)))
               (y (mirror (sample-vector :f32 2000 :seed 1)))
               (d (mirror (sample-vector :f64 2000)))
               (expected (simd:dot (contents x) (contents y)))
               (simd::*view-block-size* nil))
          (is (= expected (simd:dot x y)))
          (is (= (simd:sum (contents x)) (simd:sum x)))
          (finishes (simd:axpy! y 2.0 x))
          (finishes (simd:add! y x y :start 3 :end 90))
          (finishes (simd:convert! d x))
          (is (= (simd:sum (contents d)) (simd:sum (contents x))))
          (finishes (simd:fill! y 0.0)))))))

(test vector-view-native-copy-access
  ;; In :COPY mode a view still binds its own memory, not a copy of it.
  (when simd::*native-available-p*
    (with-backend (:native)
      (with-view-memory
        (let* ((simd::*native-array-access* :copy)
               (x (mirror (sample-vector :f32 2000)))
               (y (mirror (sample-vector :f32 2000 :seed 1)))
               (array (sample-vector :f32 2000 :seed 2))
               (expected (simd:dot (contents x) array))
               (simd::*view-block-size* nil))
          (is (= expected (simd:dot x array)))
          (finishes (simd:axpy! y 2.0 x))
          (is (equalp (contents y)
                      (let ((reference (sample-vector :f32 2000 :seed 1)))
                        (simd:axpy! reference 2.0 (contents x))
                        reference))))))))

#+sbcl
(test vector-view-bounded-staging
  ;; The Lisp backend reads a large view through a small buffer, not a full copy.
  (with-backend (:lisp)
    (with-view-memory
      (let* ((count 2000000)
             (view (simd:make-vector-view (view-memory (* 4 count)) :f32 count)))
        (simd:fill! view 1.0)
        (simd:dot view view)
        (let ((before (sb-ext:get-bytes-consed)))
          (is (= count (simd:dot view view)))
          (is (= (floor count 2) (simd:sum view :stride 2 :end (floor count 2) :accumulate :f64)
                 (simd:sum view :stride 2 :end (floor count 2))))
          (simd:axpy! view 1.0 view)
          (is (< (- (sb-ext:get-bytes-consed) before) (* 1024 1024))))
        (is (= (* 2 count) (simd:sum view :accumulate :f64)))))))

(simd:define-kernel view-kernel-combine (a b) (+ (* 2 a) (- b 1)))
(simd:define-kernel view-kernel-sum (a b) (simd:sum (* a b)))
(simd:define-kernel view-kernel-asum (a b) (simd:asum (- a b)))
(simd:define-kernel view-kernel-nrm2 (a b) (simd:nrm2 (+ a b)))
(simd:define-kernel view-kernel-minimum (a b) (simd:minimum (- a b)))
(simd:define-kernel view-kernel-maximum (a b) (simd:maximum (- a b)))
(simd:define-kernel view-kernel-argmin (a b) (simd:argmin (- a b)))
(simd:define-kernel view-kernel-argmax (a b) (simd:argmax (* a b)))
(simd:define-kernel view-kernel-less (a b) (< a b))
(simd:define-kernel view-kernel-select (a b) (simd:select (> a b) a (* 2 b)))
(simd:define-kernel view-kernel-count (a b) (simd:count (>= a b)))
(simd:define-kernel view-kernel-any (a b) (simd:any (> a (+ b 20))))
(simd:define-kernel view-kernel-masked-sum (a b) (simd:sum (simd:select (< a b) a 0)))
(simd:define-kernel view-kernel-masked-argmax (a b) (simd:argmax (simd:select (< a b) b a)))

(test vector-view-kernels
  (do-view-backends
    (dolist (key '(:f32 :f64 :s32 :s8 :u16 :c64))
      (dolist (length *view-lengths*)
        (let ((a (sample-vector key length :seed 14))
              (b (sample-vector key length :seed 15))
              (realp (not (simd::complex-type-p key)))
              (*view-case* (list 'kernels key length)))
          (check-views #'view-kernel-combine (list (sample-vector key length) a b))
          (check-views #'view-kernel-sum (list a b))
          (when (and (>= length 13) realp)
            (check-views (lambda (d a b) (view-kernel-combine d a b :start 2 :end 9
                                                              :destination-start 4))
                         (list (sample-vector key length) a b)))
          (when realp
            (check-views #'view-kernel-asum (list a b))
            (dolist (kernel (list #'view-kernel-minimum #'view-kernel-maximum
                                  #'view-kernel-argmin #'view-kernel-argmax))
              (check-views kernel (list a b))
              (when (>= length 13)
                (check-views (lambda (a b) (funcall kernel a b :start 3 :end 12 :b-start 1))
                             (list a b))))
            (check-views #'view-kernel-less
                         (list (blank '(unsigned-byte 8) length) a b))
            (check-views #'view-kernel-select (list (sample-vector key length) a b))
            (check-views #'view-kernel-count (list a b))
            (check-views #'view-kernel-any (list a b))
            (check-views #'view-kernel-masked-sum (list a b))
            (check-views #'view-kernel-masked-argmax (list a b))
            (when (>= length 13)
              (check-views (lambda (a b) (view-kernel-masked-argmax a b :start 2 :end 11))
                           (list a b))))
          (when (member key '(:f32 :f64 :c64))
            (check-views #'view-kernel-nrm2 (list a b) :test #'numerically-close-p)))))))

(defmacro with-blas-thresholds ((native) &body body)
  "Run BODY with the native BLAS kernels taken for all sizes when NATIVE, else never."
  `(let ((trivial-simd/blas::*native-blas-threshold* (if ,native 1 most-positive-fixnum))
         (trivial-simd/blas::*native-blas-level2-threshold* (if ,native 1 most-positive-fixnum)))
     ,@body))

(defun blas-key (type)
  (cond ((eq type 'single-float) :f32) ((eq type 'double-float) :f64)
        ((equal type '(complex single-float)) :c32) (t :c64)))

(defun blas-data (type length &key (seed 0) (signed t))
  (sample-vector (blas-key type) length :seed seed :signed signed))

(defun blas-scalar-of (type value)
  (coerce value type))

(test vector-view-blas-level1
  (do-view-backends
    (dolist (type '(single-float double-float (complex single-float) (complex double-float)))
      (let* ((prefix (char (string-downcase (blas-prefix type)) 0))
             (*view-case* (list 'blas-level1 type)))
        (flet ((routine (name)
                 (symbol-function (find-symbol (string-upcase (format nil "~A~A" prefix name))
                                               :trivial-simd/blas))))
          (let ((x (blas-data type 23 :seed 1)) (y (blas-data type 23 :seed 2)))
            (check-views (lambda (x y) (funcall (routine "axpy") 7 (blas-scalar-of type 2) x 3 y -2
                                                :x-offset 1 :y-offset 2))
                         (list x y))
            (check-views (lambda (x y) (funcall (routine "copy") 11 x 2 y -1 :y-offset 1)) (list x y))
            (check-views (lambda (x y) (funcall (routine "swap") 6 x -3 y 2)) (list x y))
            (check-views (lambda (x) (funcall (routine "scal") 8 (blas-scalar-of type -3) x 2)) (list x))
            (if (consp type)
                (progn
                  (check-views (lambda (x y) (funcall (routine "dotu") 9 x 2 y -1)) (list x y))
                  (check-views (lambda (x y) (funcall (routine "dotc") 9 x 2 y -1)) (list x y))
                  (let ((real-prefix (if (eq (second type) 'single-float) "cs" "zd")))
                    (check-views (lambda (x)
                                   (funcall (find-symbol (string-upcase (format nil "~ASCAL" real-prefix))
                                                         :trivial-simd/blas)
                                            8 (coerce 2 (second type)) x -2))
                                 (list x))))
                (progn
                  (check-views (lambda (x y) (funcall (routine "dot") 9 x 2 y -1)) (list x y))
                  (check-views (lambda (x y) (funcall (routine "rot") 9 x 2 y -1
                                                      (coerce 1/2 type) (coerce 3/4 type)))
                               (list x y))
                  (let ((parameters (make-array 5 :element-type type
                                                  :initial-contents (mapcar (lambda (v) (coerce v type))
                                                                            '(-1 2 1/2 -1/4 3)))))
                    (check-views (lambda (x y) (funcall (routine "rotm") 9 x 2 y -1 parameters))
                                 (list x y)))))
            (dolist (name (if (consp type)
                              (list (if (eq (second type) 'single-float) "scnrm2" "dznrm2")
                                    (if (eq (second type) 'single-float) "scasum" "dzasum")
                                    (format nil "i~Aamax" prefix))
                              (list (format nil "~Anrm2" prefix) (format nil "~Aasum" prefix)
                                    (format nil "i~Aamax" prefix))))
              (let ((function (symbol-function (find-symbol (string-upcase name) :trivial-simd/blas))))
                (check-views (lambda (x) (funcall function 11 x 2)) (list x)
                             :test #'numerically-close-p)
                (check-views (lambda (x) (funcall function 7 x -3 :x-offset 1)) (list x)
                             :test #'numerically-close-p)))))))))

(test vector-view-blas-convenience
  (do-view-backends
    (let ((x (blas-data 'double-float 17 :seed 3))
          (y (blas-data 'double-float 17 :seed 4))
          (*view-case* 'blas-convenience))
      (check-views (lambda (y x) (trivial-simd/blas/convenience:axpy! y 2d0 x :start 2)) (list y x))
      (check-views #'trivial-simd/blas/convenience:dot (list x y))
      (check-views #'trivial-simd/blas/convenience:norm2 (list x) :test #'numerically-close-p)
      (check-views (lambda (a x y)
                     (trivial-simd/blas/convenience:gemv!
                      y 2d0 (trivial-simd/blas:make-matrix-view a 3 5) x :beta 1d0))
                   (list (blas-data 'double-float 15) (blas-data 'double-float 5)
                         (blas-data 'double-float 3))))))

(defun check-blas-views (function arguments)
  "CHECK-VIEWS with the native BLAS kernels on and then off."
  (dolist (native (if trivial-simd::*native-blas-available-p* '(t nil) '(nil)))
    (with-blas-thresholds (native)
      (let ((*view-case* (list *view-case* :native-blas native)))
        (check-views function arguments)))))

(test vector-view-blas-level2
  (do-view-backends
    (dolist (type '(single-float double-float (complex double-float)))
      (dolist (layout '(:row-major :column-major))
        (let* ((m 7) (n 5)
               (realp (not (consp type)))
               (prefix (char (string-downcase (blas-prefix type)) 0))
               (*view-case* (list 'blas-level2 type layout)))
          (flet ((routine (name)
                   (symbol-function (find-symbol (string-upcase (format nil "~A~A" prefix name))
                                                 :trivial-simd/blas)))
                 (matrix (data rows cols &rest keys)
                   (apply #'trivial-simd/blas:make-matrix-view data rows cols :layout layout keys)))
            (dolist (transpose '(:no-transpose :transpose :conjugate-transpose))
              (check-blas-views
               (lambda (a x y)
                 (funcall (routine "gemv") transpose (blas-scalar-of type 2) (matrix a m n :offset 1)
                          x 2 (blas-scalar-of type 1/2) y -1 :x-offset 1)
                 y)
               (list (blas-data type (1+ (* m n)) :seed 5) (blas-data type 20 :seed 6)
                     (blas-data type 9 :seed 7))))
            (check-blas-views
             (lambda (x y a)
               (funcall (routine (if realp "ger" "geru")) (blas-scalar-of type -1) x 1 y 2
                        (matrix a m n))
               a)
             (list (blas-data type m :seed 8) (blas-data type 10 :seed 9)
                   (blas-data type (* m n) :seed 10)))
            ;; A dominant diagonal keeps the triangular solves well conditioned.
            (let ((triangle (blas-data type 36 :seed 11 :signed nil)))
              (dotimes (i 6) (setf (aref triangle (* 7 i)) (blas-scalar-of type 40)))
              (dolist (uplo '(:upper :lower))
                (dolist (transpose '(:no-transpose :transpose))
                  (check-blas-views
                   (lambda (a x)
                     (funcall (routine "trsv") uplo transpose :non-unit (matrix a 6 6) x 1)
                     x)
                   (list triangle (blas-data type 6 :seed 12)))
                  (check-blas-views
                   (lambda (a x)
                     (funcall (routine "trmv") uplo transpose :unit (matrix a 6 6) x -1)
                     x)
                   (list triangle (blas-data type 6 :seed 13))))
                (check-blas-views
                 (lambda (a x y)
                   (funcall (routine (if realp "symv" "hemv")) uplo (blas-scalar-of type 1)
                            (matrix a 6 6) x 1 (blas-scalar-of type 2) y 1)
                   y)
                 (list triangle (blas-data type 6 :seed 14) (blas-data type 6 :seed 15)))
                (check-blas-views
                 (lambda (x a)
                   (if realp
                       (funcall (routine "syr") uplo (blas-scalar-of type 2) x 1 (matrix a 6 6))
                       (funcall (routine "her") uplo 2d0 x 1 (matrix a 6 6)))
                   a)
                 (list (blas-data type 6 :seed 16) (copy-seq triangle)))
                (check-blas-views
                 (lambda (a x)
                   (funcall (routine "tpmv") uplo :no-transpose :non-unit
                            (trivial-simd/blas:make-packed-matrix-view a 6 :layout layout) x 1)
                   x)
                 (list (blas-data type 21 :seed 17) (blas-data type 6 :seed 18)))
                (check-blas-views
                 (lambda (a x)
                   (funcall (routine "tbmv") uplo :transpose :non-unit
                            (trivial-simd/blas:make-band-matrix-view
                             a 6 6 :kind :triangular :bandwidth 2 :layout layout)
                            x 1)
                   x)
                 (list (blas-data type 18 :seed 19) (blas-data type 6 :seed 20)))))
            (check-blas-views
             (lambda (a x y)
               (funcall (routine "gbmv") :no-transpose (blas-scalar-of type 1)
                        (trivial-simd/blas:make-band-matrix-view a m n :kl 2 :ku 1 :layout layout)
                        x 1 (blas-scalar-of type 1) y 1)
               y)
             (list (blas-data type 40 :seed 21) (blas-data type n :seed 22)
                   (blas-data type m :seed 23)))))))))

(test vector-view-blas-level3
  (do-view-backends
    (dolist (type '(single-float double-float (complex single-float)))
      (dolist (layout '(:row-major :column-major))
        (let* ((prefix (char (string-downcase (blas-prefix type)) 0))
               (*view-case* (list 'blas-level3 type layout)))
          (flet ((routine (name)
                   (symbol-function (find-symbol (string-upcase (format nil "~A~A" prefix name))
                                                 :trivial-simd/blas)))
                 (matrix (data rows cols &rest keys)
                   (apply #'trivial-simd/blas:make-matrix-view data rows cols :layout layout keys)))
            (dolist (transposes '((:no-transpose :no-transpose) (:transpose :no-transpose)
                                  (:no-transpose :conjugate-transpose)))
              (check-blas-views
               (lambda (a b c)
                 (funcall (routine "gemm") (first transposes) (second transposes)
                          (blas-scalar-of type 2) (matrix a 6 6 :leading-dimension 7)
                          (matrix b 6 6 :offset 2) (blas-scalar-of type 1/2) (matrix c 6 6))
                 c)
               (list (blas-data type 42 :seed 1) (blas-data type 40 :seed 2)
                     (blas-data type 36 :seed 3))))
            (dolist (side '(:left :right))
              (check-blas-views
               (lambda (a b c)
                 (funcall (routine "symm") side :upper (blas-scalar-of type 1)
                          (matrix a 5 5) (matrix b 5 5) (blas-scalar-of type 1) (matrix c 5 5))
                 c)
               (list (blas-data type 25 :seed 4) (blas-data type 25 :seed 5)
                     (blas-data type 25 :seed 6)))
              (let ((triangle (blas-data type 25 :seed 7 :signed nil)))
                (dotimes (i 5) (setf (aref triangle (* 6 i)) (blas-scalar-of type 30)))
                (dolist (name '("trmm" "trsm"))
                  (check-blas-views
                   (lambda (a b)
                     (funcall (routine name) side :lower :no-transpose :non-unit
                              (blas-scalar-of type 2) (matrix a 5 5) (matrix b 5 5))
                     b)
                   (list triangle (blas-data type 25 :seed 8))))))
            (check-blas-views
             (lambda (a c)
               (funcall (routine "syrk") :lower :no-transpose (blas-scalar-of type 1)
                        (matrix a 5 4) (blas-scalar-of type 1) (matrix c 5 5))
               c)
             (list (blas-data type 20 :seed 9) (blas-data type 25 :seed 10)))
            (check-blas-views
             (lambda (a b c)
               (funcall (routine "syr2k") :upper :transpose (blas-scalar-of type 1)
                        (matrix a 4 5) (matrix b 4 5) (blas-scalar-of type 1) (matrix c 5 5))
               c)
             (list (blas-data type 20 :seed 11) (blas-data type 20 :seed 12)
                   (blas-data type 25 :seed 13)))
            (with-view-memory
              ;; Views of one memory overlap even as distinct objects.
              (let* ((data (mirror (blas-data type 100)))
                     (alias (simd:make-vector-view (simd:vector-view-pointer data)
                                                   (blas-key type) 100 :offset 10))
                     (apart (simd:make-vector-view (simd:vector-view-pointer data)
                                                   (blas-key type) 50 :offset 50)))
                (signals error
                  (funcall (routine "gemm") :no-transpose :no-transpose (blas-scalar-of type 1)
                           (matrix data 5 5) (matrix data 5 5 :offset 50)
                           (blas-scalar-of type 0) (matrix alias 5 5)))
                (finishes
                  (funcall (routine "gemm") :no-transpose :no-transpose (blas-scalar-of type 1)
                           (matrix data 5 5) (matrix alias 5 5)
                           (blas-scalar-of type 0) (matrix apart 5 5)))))))))))

(test vector-view-blas-errors
  (with-view-memory
    (let ((f32 (mirror (blas-data 'single-float 8)))
          (s32 (mirror (sample-vector :s32 8))))
      (signals type-error (trivial-simd/blas:ddot 4 f32 1 f32 1))
      (signals type-error (trivial-simd/blas:saxpy 4 1.0 s32 1 f32 1))
      (signals error (trivial-simd/blas:sdot 9 f32 1 f32 1))
      (signals error (trivial-simd/blas:make-matrix-view f32 3 3))
      (signals error (trivial-simd/blas:make-matrix-view s32 2 2))
      (is (= 4 (trivial-simd/blas:matrix-view-rows (trivial-simd/blas:make-matrix-view f32 4 2))))
      (let ((view (trivial-simd/blas:make-matrix-view f32 4 2 :layout :column-major)))
        (setf (trivial-simd/blas:matrix-ref view 1 1) 42.0)
        (is (= 42.0 (trivial-simd/blas:matrix-ref view 1 1)))
        (is (= 42.0 (simd::view-ref f32 5)))))))
