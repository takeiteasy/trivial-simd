(require :asdf)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (find-package :ql)
    (let ((setup (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname))))
      (when (probe-file setup) (load setup))))
  (let ((root #.(merge-pathnames "../"
                                (uiop:pathname-directory-pathname
                                 (or *compile-file-truename* *load-truename*)))))
    (asdf:initialize-source-registry
     `(:source-registry (:directory ,root) :inherit-configuration)))
  (asdf:load-asd #.(merge-pathnames "../trivial-simd.asd"
                                   (uiop:pathname-directory-pathname
                                    (or *compile-file-truename* *load-truename*))))
  (asdf:load-system "trivial-simd")
  (load #.(merge-pathnames "benchmark-timing.lisp"
                          (uiop:pathname-directory-pathname
                           (or *compile-file-truename* *load-truename*)))))

(trivial-simd:define-kernel stage-softmax (x)
  (/ (exp (- x (trivial-simd:maximum x)))
     (trivial-simd:sum (exp (- x (trivial-simd:maximum x))))))
(trivial-simd:define-kernel stage-silu (x) (/ x (+ 1 (exp (- x)))))
(trivial-simd:define-kernel stage-sine (x) (sin x))
(trivial-simd:define-kernel stage-cosine (x) (cos x))

(macrolet ((references ()
             `(progn
                ,@(loop for type in '(single-float double-float) for suffix in '(f32 f64)
                        append
                        (loop for stage in '(softmax silu rope)
                              collect
                              `(defun ,(intern (format nil "REFERENCE-~A-~A" stage suffix))
                                   (out cosine x start count)
                                 (declare (type (simple-array ,type (*)) out cosine x)
                                          (type fixnum start count) (ignorable cosine)
                                          (optimize (speed 3) (safety 1)))
                                 ,(case stage
                                    (softmax
                                     `(when (plusp count)
                                        (let ((maximum (aref x start)) (sum ,(coerce 0 type)))
                                          (declare (type ,type maximum sum))
                                          (dotimes (i count) (setf maximum (max maximum (aref x (+ start i)))))
                                          (dotimes (i count)
                                            (let ((value (exp (- (aref x (+ start i)) maximum))))
                                              (setf (aref out (+ start i)) value)
                                              (incf sum value)))
                                          (dotimes (i count)
                                            (setf (aref out (+ start i)) (/ (aref out (+ start i)) sum))))))
                                    (silu
                                     `(dotimes (i count)
                                        (let ((x (aref x (+ start i))))
                                          (setf (aref out (+ start i)) (/ x (+ ,(coerce 1 type) (exp (- x))))))))
                                    (rope
                                     `(dotimes (i count)
                                        (let ((angle (aref x (+ start i))))
                                          (setf (aref out (+ start i)) (sin angle)
                                                (aref cosine (+ start i)) (cos angle))))))
                                 out))))))
  (references))

(defun allocated-bytes ()
  #+sbcl (sb-ext:get-bytes-consed)
  #+ccl (ccl:total-bytes-allocated)
  #-(or sbcl ccl) nil)

(defun stage-measure (function)
  (let* ((time (benchmark-time function))
        (iterations (max 100 (min 10000 (ceiling 20000d0 time))))
        (before (allocated-bytes)))
    (dotimes (i iterations) (funcall function))
    (values time (when before (/ (- (allocated-bytes) before) (float iterations 1d0))))))

(defun benchmark-stage (type n stage backend &optional (rows 1))
  (let* ((length (* rows n))
         (x (make-array length :element-type type
                              :initial-contents (loop for i below length collect (coerce (/ (- (mod i 17) 8) 4) type))))
         (out (make-array length :element-type type))
         (cosine (make-array length :element-type type))
         (reference (intern (format nil "REFERENCE-~A-~A" stage (if (eq type 'single-float) 'f32 'f64))))
         (trivial-simd::*backend* (if (eq backend :native-copy) :native backend))
         (trivial-simd::*native-array-access* (if (eq backend :native-copy) :copy :pointer)))
    (flet ((run ()
             (if (eq backend :typed)
                 (dotimes (row rows out) (funcall reference out cosine x (* row n) n))
                 (ecase stage
                   (softmax (if (= rows 1) (stage-softmax out x)
                                (stage-softmax out x :rows rows :row-length n)))
                   (silu (stage-silu out x))
                   (rope (stage-sine out x) (stage-cosine cosine x))))))
      (run)
      (let ((expected (make-array length :element-type type))
            (expected-cosine (make-array length :element-type type))
            (tolerance (if (eq type 'single-float) 1f-5 1d-12)))
        (dotimes (row rows) (funcall reference expected expected-cosine x (* row n) n))
        (dotimes (i length)
          (assert (<= (abs (- (aref expected i) (aref out i))) tolerance))
          (when (eq stage 'rope)
            (assert (<= (abs (- (aref expected-cosine i) (aref cosine i))) tolerance)))))
      (multiple-value-bind (time bytes) (stage-measure #'run)
        (format t "~&~10A ~11A ~7D ~4D ~8A ~12,3F ~12A~%" backend type n rows stage time
                (if bytes (format nil "~,0F" bytes) "unavailable"))))))

(format t "~&~A ~A, ~A ~A~%" (lisp-implementation-type) (lisp-implementation-version)
        (machine-type) (machine-version))
(format t "Scalar system math; softmax kernel: 3 passes, 2 exp evaluations per element.~%")
(format t "Typed softmax caches exp values in output; RoPE times both output tables.~%")
(format t "Backend    Type              N Rows Stage       us/stage   Lisp-bytes/stage~%")
(dolist (backend (append '(:typed :lisp)
                         (when trivial-simd::*native-available-p* '(:native :native-copy))
                         (when trivial-simd::*sbcl-simd-available-p* '(:sbcl))))
  (dolist (type '(single-float double-float))
    (dolist (n '(32 128 512 4096 32768))
      (dolist (stage '(softmax silu rope)) (benchmark-stage type n stage backend)))
    (dolist (shape '((32 32) (32 512)))
      (benchmark-stage type (second shape) 'softmax backend (first shape)))))
