(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defparameter *stride-types*
  '(single-float double-float (complex single-float) (complex double-float)
    (signed-byte 16) (unsigned-byte 8) (signed-byte 64)))

(defparameter *stride-triples* '((1 1 1) (2 2 2) (3 -1 2) (-2 3 1) (1 3 -3) (-1 -1 -1)))

(defparameter *stride-count* 7)

(defun scrambled-vector (type length seed)
  (let ((vector (make-array length :element-type type)))
    (dotimes (i length vector)
      (setf (aref vector i) (copy-element type (mod (* 7 (+ i seed)) 23))))))

(defun first-offset (stride count)
  "Offset of the first visited element, leaving room below for negative strides."
  (+ 2 (if (minusp stride) (* (1- count) (- stride)) 0)))

(defun plain-gather (vector start stride count)
  (let ((result (make-array count :element-type (array-element-type vector))))
    (dotimes (i count result)
      (setf (aref result i) (aref vector (+ start (* i stride)))))))

(defun plain-scatter (values vector start stride)
  (dotimes (i (length values) vector)
    (setf (aref vector (+ start (* i stride))) (aref values i))))

(defun prefixed-keyword (prefix suffix)
  (if prefix
      (intern (format nil "~A-~A" prefix suffix) :keyword)
      (if (eq suffix 'start) :start :stride)))

(defun vector-item-p (item)
  (and (consp item) (member (first item) '(:in :out :in-out))))

(defun check-strided (function items strides &key extra-keys (count *stride-count*))
  "Call FUNCTION on ITEMS with per-vector STRIDES and compare each vector with
a contiguous call on plainly gathered temporaries. An item is a scalar or
(ROLE VECTOR PREFIX) with ROLE :IN, :OUT or :IN-OUT."
  (let* ((remaining strides)
         (entries (loop for item in items
                        collect (if (vector-item-p item)
                                    (destructuring-bind (role vector prefix) item
                                      (let ((stride (pop remaining)))
                                        (list role vector prefix stride
                                              (first-offset stride count) (copy-seq vector))))
                                    item)))
         (temporaries
           (loop for entry in entries
                 collect (if (vector-item-p entry)
                             (destructuring-bind (role vector prefix stride start expected) entry
                               (declare (ignore prefix expected))
                               (if (eq role :out)
                                   (make-array count :element-type (array-element-type vector))
                                   (plain-gather vector start stride count)))
                             entry))))
    (apply function (append temporaries (list :end count) extra-keys))
    (loop for entry in entries for temporary in temporaries
          when (and (vector-item-p entry) (member (first entry) '(:out :in-out)))
            do (destructuring-bind (role vector prefix stride start expected) entry
                 (declare (ignore role vector prefix))
                 (plain-scatter temporary expected start stride)))
    (apply function
           (append (loop for entry in entries collect (if (vector-item-p entry) (second entry) entry))
                   (list :end (let ((shared (find nil (remove-if-not #'vector-item-p entries)
                                                  :key #'third)))
                                (if shared (+ (fifth shared) count) count)))
                   (loop for entry in entries
                         when (vector-item-p entry)
                           append (destructuring-bind (role vector prefix stride start expected) entry
                                    (declare (ignore role vector expected))
                                    (list (prefixed-keyword prefix 'start) start
                                          (prefixed-keyword prefix 'stride) stride)))
                   extra-keys))
    (loop for entry in entries
          when (vector-item-p entry)
            do (is (equalp (second entry) (sixth entry))))))

(defmacro with-stride-cases ((type-var types triple-var) &body body)
  `(dolist (,type-var ,types)
     (dolist (backend (available-backends))
       (with-backend (backend)
         (dolist (,triple-var *stride-triples*)
           ,@body)))))

(test strided-binary-operations
  (with-stride-cases (type *stride-types* strides)
    (dolist (operation '(simd:add! simd:subtract! simd:multiply! simd:divide!))
      (check-strided operation
                     (list (list :out (copy-vector type 40 300) :destination)
                           (list :in (copy-vector type 40 0) :left)
                           (list :in (copy-vector type 40 7) :right))
                     strides))))

(test strided-binary-scalar-operands
  (with-stride-cases (type '(single-float (signed-byte 16) (complex double-float)) strides)
    (let ((scalar (copy-element type 3)))
      (check-strided 'simd:add!
                     (list (list :out (copy-vector type 40 300) :destination)
                           (list :in (copy-vector type 40 0) :left) scalar)
                     (subseq strides 0 2))
      (check-strided 'simd:multiply!
                     (list (list :out (copy-vector type 40 300) :destination)
                           scalar (list :in (copy-vector type 40 0) :right))
                     (subseq strides 0 2)))))

(test strided-scale-axpy-and-fma
  (with-stride-cases (type *stride-types* strides)
    (let ((scalar (copy-element type 3)))
      (check-strided 'simd:scale! (list (list :in-out (copy-vector type 40 0) nil) scalar)
                     (subseq strides 0 1))
      (check-strided 'simd:axpy!
                     (list (list :in-out (copy-vector type 40 300) :y) scalar
                           (list :in (copy-vector type 40 0) :x))
                     (subseq strides 0 2))))
  (with-stride-cases (type '(single-float double-float) strides)
    (check-strided 'simd:fma!
                   (list (list :out (copy-vector type 40 300) :destination)
                         (list :in (copy-vector type 40 0) :x)
                         (list :in (copy-vector type 40 3) :y)
                         (list :in (copy-vector type 40 5) :z))
                   (append strides (list 2)))))

(test strided-unary-and-bounded-operations
  (with-stride-cases (type '(single-float double-float (complex single-float)
                             (signed-byte 16) (signed-byte 64))
                      strides)
    (dolist (operation '(simd:negate! simd:abs!))
      (unless (and (eq operation 'simd:abs!) (consp type) (eq (first type) 'complex))
        (check-strided operation
                       (list (list :out (copy-vector type 40 300) :destination)
                             (list :in (scrambled-vector type 40 1) :input))
                       (subseq strides 0 2)))))
  (with-stride-cases (type '(single-float double-float) strides)
    (dolist (operation '(simd:sqrt! simd:reciprocal!))
      (check-strided operation
                     (list (list :out (copy-vector type 40 300) :destination)
                           (list :in (scrambled-vector type 40 1) :input))
                     (subseq strides 0 2)))
    (dolist (operation '(simd:min! simd:max!))
      (check-strided operation
                     (list (list :out (copy-vector type 40 300) :destination)
                           (list :in (scrambled-vector type 40 1) :left)
                           (list :in (scrambled-vector type 40 4) :right))
                     strides))
    (check-strided 'simd:clamp!
                   (list (list :out (copy-vector type 40 300) :destination)
                         (list :in (scrambled-vector type 40 1) :input)
                         (list :in (scrambled-vector type 40 9) :lower)
                         (copy-element type 30))
                   strides)))

(test strided-complex-abs-writes-real-vector
  (with-stride-cases (type '((complex single-float) (complex double-float)) strides)
    (let ((real-type (if (equal type '(complex single-float)) 'single-float 'double-float)))
      (check-strided 'simd:abs!
                     (list (list :out (copy-vector real-type 40 300) :destination)
                           (list :in (scrambled-vector type 40 1) :input))
                     (subseq strides 0 2)))))

(test strided-convert
  (with-stride-cases (type '(single-float double-float) strides)
    (dolist (target '(double-float single-float (signed-byte 32)))
      (unless (equal target type)
        (check-strided 'simd:convert!
                       (list (list :out (copy-vector target 40 300) :destination)
                             (list :in (scrambled-vector type 40 1) :input))
                       (subseq strides 0 2))))))

(test strided-convert-encoded
  (with-stride-cases (type '(single-float) strides)
    (dolist (encoding '(:bf16 :f16))
      (check-strided 'simd:convert!
                     (list (list :out (copy-vector '(unsigned-byte 16) 40 300) :destination)
                           (list :in (scrambled-vector 'single-float 40 1) :input))
                     (subseq strides 0 2)
                     :extra-keys (list :destination-encoding encoding))
      (check-strided 'simd:convert!
                     (list (list :out (copy-vector 'single-float 40 300) :destination)
                           (list :in (scrambled-vector '(unsigned-byte 16) 40 1) :input))
                     (subseq strides 0 2)
                     :extra-keys (list :input-encoding encoding)))))

(test strided-masks
  (with-stride-cases (type '(single-float (signed-byte 32) (complex double-float)) strides)
    (let ((mask (make-array 40 :element-type '(unsigned-byte 8) :initial-element 9)))
      (dolist (operator '(:eq :ne))
        (check-strided 'simd:compare!
                       (list (list :out mask :mask) operator
                             (list :in (scrambled-vector type 40 1) :left)
                             (list :in (scrambled-vector type 40 4) :right))
                       strides))
      (unless (consp type)
        (check-strided 'simd:compare!
                       (list (list :out mask :mask) :lt
                             (list :in (scrambled-vector type 40 1) :left)
                             (list :in (scrambled-vector type 40 4) :right))
                       strides))
      (let ((bits (scrambled-vector '(unsigned-byte 8) 40 0)))
        (dotimes (i 40) (setf (aref bits i) (mod (aref bits i) 2)))
        (check-strided 'simd:select!
                       (list (list :out (copy-vector type 40 300) :destination)
                             (list :in bits :mask)
                             (list :in (scrambled-vector type 40 1) :true)
                             (list :in (scrambled-vector type 40 4) :false))
                       (append strides (list 2)))
        (dolist (operation '(simd:count simd:any simd:all))
          (loop for stride in '(2 3 -1 -3)
                for start = (first-offset stride 9)
                do (is (equal (funcall operation bits :end 9 :mask-start start :mask-stride stride)
                              (funcall operation (plain-gather bits start stride 9))))))))))

(test strided-reductions
  (dolist (type '(single-float double-float (signed-byte 32) (unsigned-byte 16)))
    (dolist (backend (available-backends))
      (with-backend (backend)
        (dolist (stride '(2 3 -1 -3))
          (let* ((count 9) (start (first-offset stride count))
                 (v (scrambled-vector type 40 1)) (w (scrambled-vector type 40 5))
                 (gv (plain-gather v start stride count))
                 (gw (plain-gather w start stride count)))
            (flet ((strided (function &rest keys)
                     (apply function v :end count :input-start start :input-stride stride keys)))
              (is (equalp (strided #'simd:sum) (simd:sum gv)))
              (is (equalp (strided #'simd:asum) (simd:asum gv)))
              (is (equalp (strided #'simd:minimum) (simd:minimum gv)))
              (is (equalp (strided #'simd:maximum) (simd:maximum gv)))
              (is (= (strided #'simd:argmin) (+ start (* stride (simd:argmin gv)))))
              (is (= (strided #'simd:argmax) (+ start (* stride (simd:argmax gv)))))
              (is (equalp (simd:dot v w :end count :left-start start :left-stride stride
                                        :right-start start :right-stride stride)
                          (simd:dot gv gw)))
              (is (equalp (simd:dot v w :start start :end (+ start count) :stride stride)
                          (simd:dot gv gw)))
              (when (member type '(single-float double-float))
                (is (equalp (strided #'simd:nrm2) (simd:nrm2 gv)))
                (is (equalp (strided #'simd:sum :accumulate :f64)
                            (simd:sum gv :accumulate :f64)))))))))))

(test strided-complex-reductions
  (dolist (type '((complex single-float) (complex double-float)))
    (dolist (backend (available-backends))
      (with-backend (backend)
        (dolist (stride '(2 -3))
          (let* ((count 9) (start (first-offset stride count))
                 (v (scrambled-vector type 40 1)) (w (scrambled-vector type 40 5))
                 (gv (plain-gather v start stride count))
                 (gw (plain-gather w start stride count)))
            (is (equalp (simd:sum v :end count :input-start start :stride stride) (simd:sum gv)))
            (is (equalp (simd:asum v :end count :input-start start :stride stride) (simd:asum gv)))
            (is (equalp (simd:nrm2 v :end count :input-start start :stride stride) (simd:nrm2 gv)))
            (is (equalp (simd:dotc v w :end count :left-start start :right-start start :stride stride)
                        (simd:dotc gv gw)))))))))

(test strided-literal-examples
  (dolist (backend (available-backends))
    (with-backend (backend)
      (let ((a (make-array 8 :element-type 'single-float
                             :initial-contents '(0f0 1f0 2f0 3f0 4f0 5f0 6f0 7f0)))
            (out (make-array 4 :element-type 'single-float :initial-element -1f0)))
        (is (= 12f0 (simd:sum a :end 4 :stride 2)))
        (is (= 16f0 (simd:sum a :end 4 :input-start 7 :stride -2)))
        (is (= 6 (simd:argmax a :end 4 :stride 2)))
        (is (= 7 (simd:argmax a :end 4 :input-start 7 :stride -2)))
        (is (= 1 (simd:argmin a :end 4 :input-start 7 :stride -2)))
        (simd:add! out a 10f0 :end 4 :left-start 7 :left-stride -2)
        (is (equalp out #(17f0 15f0 13f0 11f0)))
        (simd:copy! a out :end 4 :destination-stride 2)
        (is (equalp a #(17f0 1f0 15f0 3f0 13f0 5f0 11f0 7f0)))
        (simd:fill! a 0f0 :start 1 :end 4 :stride 2)
        (is (equalp a #(17f0 0f0 15f0 0f0 13f0 0f0 11f0 7f0)))))))

(test strided-aliased-destination-reads-inputs-first
  (dolist (type '(single-float (signed-byte 32) (complex single-float)))
    (dolist (backend (available-backends))
      (with-backend (backend)
        (let* ((v (copy-vector type 20 0)) (expected (copy-seq v)))
          (simd:add! v v v :end 6 :destination-stride 2 :left-stride 2 :right-stride 2)
          (dotimes (i 6) (setf (aref expected (* 2 i)) (+ (aref expected (* 2 i)) (aref expected (* 2 i)))))
          (is (equalp v expected)))
        (let* ((v (copy-vector type 20 0)) (original (copy-seq v)))
          (simd:copy! v v :end 5 :destination-start 2 :destination-stride 2
                          :source-start 0 :source-stride 2)
          (dotimes (i 5) (setf (aref original (+ 2 (* 2 i))) (copy-element type (* 2 i))))
          (is (equalp v original)))))))

(test strided-empty-slices
  (dolist (backend (available-backends))
    (with-backend (backend)
      (let ((v (copy-vector 'single-float 10 0)))
        (is (= 0f0 (simd:sum v :end 0 :stride 3)))
        (is (null (simd:argmax v :end 0 :stride -2)))
        (is (null (simd:minimum v :input-start 10 :end 0 :stride 5)))
        (is (eq v (simd:scale! v 2f0 :end 0 :stride 4)))))))

(test strided-errors
  (let ((v (copy-vector 'single-float 40 0)) (out (copy-vector 'single-float 40 0))
        (mask (make-array 40 :element-type '(unsigned-byte 8))))
    (signals error (simd:sum v :stride 0))
    (signals error (simd:sum v :stride 1.5))
    (signals error (simd:sum v :input-stride 0 :end 4))
    (signals error (simd:add! out v v :left-stride 0))
    (signals error (simd:sum v :end 15 :stride 3))
    (finishes (simd:sum v :end 14 :stride 3))
    (signals error (simd:sum v :end 3 :stride -1))
    (signals error (simd:sum v :end 3 :input-start 1 :stride -1))
    (finishes (simd:sum v :end 3 :input-start 2 :stride -1))
    (signals error (simd:sum v :end 2 :input-start 40 :stride -1))
    (signals error (simd:add! out v 1f0 :right-stride 2))
    (signals error (simd:add! out v v :end 7 :destination-stride 7))
    (signals error (simd:select! out mask v 1f0 :false-stride 2))
    (signals error (simd:copy! out v :stride 2))
    (signals error (simd:fill! out 1f0 :stride 2))
    (signals error (simd:swap! out v :end 25 :x-stride 2))))

(test lisp-bulk-loops-define-three-top-level-functions-per-type
  (let ((expansion (macroexpand-1 '(simd::define-lisp-bulk-loops))))
    (is (= 10 (length (rest expansion))))
    (dolist (per-type (rest expansion))
      (is (equal '(defun defun defun) (mapcar #'first (rest per-type)))))))

(test strided-convert-extended-encoded
  (with-stride-cases (type '(double-float) strides)
    (dolist (encoding '(:bf16 :f16))
      (dolist (rounding '(:nearest-even :truncate :floor :ceiling))
        (check-strided 'simd:convert!
                       (list (list :out (copy-vector '(unsigned-byte 16) 40 300) :destination)
                             (list :in (scrambled-vector type 40 1) :input))
                       (subseq strides 0 2)
                       :extra-keys (list :destination-encoding encoding :rounding rounding))
        (check-strided 'simd:convert!
                       (list (list :out (copy-vector type 40 300) :destination)
                             (list :in (scrambled-vector '(unsigned-byte 16) 40 1) :input))
                       (subseq strides 0 2)
                       :extra-keys (list :input-encoding encoding :rounding rounding))
        (check-strided 'simd:convert!
                       (list (list :out (copy-vector '(unsigned-byte 16) 40 300) :destination)
                             (list :in (scrambled-vector '(unsigned-byte 16) 40 1) :input))
                       (subseq strides 0 2)
                       :extra-keys (list :input-encoding encoding
                                         :destination-encoding (if (eq encoding :bf16) :f16 :bf16)
                                         :rounding rounding))))))
