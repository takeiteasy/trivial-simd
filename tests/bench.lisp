(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :asdf)
  (unless (find-package :ql)
    (dolist (candidate '("quicklisp/setup.lisp" ".roswell/lisp/quicklisp/setup.lisp"))
      (let ((path (merge-pathnames candidate (user-homedir-pathname))))
        (when (probe-file path) (load path) (return))))))

(defmacro load-benchmark-system ()
  `(progn
     (asdf:load-asd ,(merge-pathnames "../trivial-simd.asd"
                                     (uiop:pathname-directory-pathname
                                      (or *compile-file-truename* *load-truename*))))
     (asdf:load-system "trivial-simd")))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (load-benchmark-system))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (load #.(merge-pathnames "benchmark-timing.lisp"
                          (uiop:pathname-directory-pathname
                           (or *compile-file-truename* *load-truename*)))))

(macrolet ((define-scalar-loops (type add multiply-add dot zero)
             `(progn
                (defun ,add (out a b)
                  (declare (type (simple-array ,type (*)) out a b)
                           (optimize (speed 3)))
                  (dotimes (i (length out) out)
                    (setf (aref out i) (+ (aref a i) (aref b i)))))
                (defun ,multiply-add (out a b c)
                  (declare (type (simple-array ,type (*)) out a b c)
                           (optimize (speed 3)))
                  (dotimes (i (length out) out)
                    (setf (aref out i) (+ (* (aref a i) (aref b i)) (aref c i)))))
                (defun ,dot (a b)
                  (declare (type (simple-array ,type (*)) a b)
                           (optimize (speed 3)))
                  (let ((result ,zero))
                    (declare (type ,type result))
                    (dotimes (i (length a) result)
                      (incf result (* (aref a i) (aref b i)))))))))
  (define-scalar-loops single-float scalar-add-f32 scalar-multiply-add-f32
    scalar-dot-f32 0.0f0)
  (define-scalar-loops double-float scalar-add-f64 scalar-multiply-add-f64
    scalar-dot-f64 0.0d0))

(dolist (name '(scalar-add-f32 scalar-multiply-add-f32 scalar-dot-f32
               scalar-add-f64 scalar-multiply-add-f64 scalar-dot-f64))
  (unless (compiled-function-p (symbol-function name))
    (compile name)))

(defun benchmark-backends ()
  (append '(:lisp)
          (when trivial-simd::*native-available-p* '(:native :native-copy))
          (when trivial-simd::*sbcl-simd-available-p* '(:sbcl))))

(defmacro with-benchmark-backend ((backend) &body body)
  `(let ((trivial-simd::*backend* (if (eq ,backend :native-copy) :native ,backend))
         (trivial-simd::*native-array-access*
           (if (eq ,backend :native-copy) :copy trivial-simd::*native-array-access*)))
     ,@body))

(trivial-simd:define-kernel multiply-add (a b c) (+ (* a b) c))

(trivial-simd:define-kernel rounded-multiply-add (a b c) (trivial-simd:fma a b c))

(defun benchmark-input-value (input index)
  (ecase input
    (:a (/ (- (mod index 13) 6) 4))
    (:b (/ (1+ (mod index 7)) 2))
    (:c (/ (- (mod index 5) 2) 8))))

(defun benchmark-input (input length type)
  (make-array length :element-type type :initial-contents
              (loop for i below length collect
                (coerce (benchmark-input-value input i) type))))

(defun array-benchmark-call (operation implementation type out a b c)
  (let ((function
          (if (eq implementation :scalar)
              (ecase type
                (single-float
                 (ecase operation
                   (:add #'scalar-add-f32)
                   (:multiply-add #'scalar-multiply-add-f32)
                   (:dot #'scalar-dot-f32)))
                (double-float
                 (ecase operation
                   (:add #'scalar-add-f64)
                   (:multiply-add #'scalar-multiply-add-f64)
                   (:dot #'scalar-dot-f64))))
              (ecase operation
                (:add #'trivial-simd:add!)
                (:multiply-add #'multiply-add)
                (:dot #'trivial-simd:dot)))))
    (ecase operation
      (:add (lambda () (funcall function out a b)))
      (:multiply-add (lambda () (funcall function out a b c)))
      (:dot (lambda () (funcall function a b))))))

(defun check-array-benchmark (operation function type length out)
  (unless (eq operation :dot)
    (fill out (coerce -1000 type)))
  (let ((result (funcall function))
        (tolerance (if (eq type 'single-float) 1.0e-5 1.0d-12)))
    (labels ((check (actual expected)
               (unless (<= (abs (- actual expected))
                           (* tolerance (max 1 (abs expected))))
                 (error "~A ~A benchmark mismatch: got ~S, expected ~S"
                        operation type actual expected)))
             (product (i)
               (* (benchmark-input-value :a i) (benchmark-input-value :b i))))
      (if (eq operation :dot)
          (check result (coerce (loop for i below length sum (product i)) type))
          (progn
            (unless (= (length result) length)
              (error "~A benchmark returned the wrong array length" operation))
            (dotimes (i length)
              (check (aref result i)
                     (coerce
                      (ecase operation
                        (:add (+ (benchmark-input-value :a i)
                                 (benchmark-input-value :b i)))
                        (:multiply-add (+ (product i) (benchmark-input-value :c i))))
                      type))))))))

(defun check-array-benchmarks ()
  (dolist (type '(single-float double-float))
    (dolist (length '(0 1 3 5 17))
      (let ((a (benchmark-input :a length type))
            (b (benchmark-input :b length type))
            (c (benchmark-input :c length type))
            (out (make-array length :element-type type)))
        (dolist (operation '(:add :multiply-add :dot))
          (dolist (implementation (cons :scalar (benchmark-backends)))
            (with-benchmark-backend (implementation)
              (check-array-benchmark
               operation (array-benchmark-call operation implementation type out a b c)
               type length out))))))))

(format t "~A ~A | machine ~A | backend ~A | native array access ~A~%"
        (lisp-implementation-type) (lisp-implementation-version) (machine-type)
        (trivial-simd:backend)
        (if trivial-simd::*native-available-p* trivial-simd::*native-array-access*
            :unavailable))
(check-array-benchmarks)
(format t "~%Array comparison (speedup = scalar / implementation)~%")
(format t "~12A ~12A ~8A ~14A ~10A ~10A~%"
        "Operation" "Type" "Elements" "Implementation" "us/call" "Speedup")
(dolist (type '(single-float double-float))
  (dolist (length '(32 1024 65536))
    (let ((a (benchmark-input :a length type))
          (b (benchmark-input :b length type))
          (c (benchmark-input :c length type))
          (out (make-array length :element-type type)))
      (dolist (operation '(:add :multiply-add :dot))
        (let ((scalar-time nil))
          (dolist (implementation (cons :scalar (benchmark-backends)))
            (with-benchmark-backend (implementation)
              (let ((function (array-benchmark-call
                               operation implementation type out a b c)))
                (check-array-benchmark operation function type length out)
                (let ((time (benchmark-time function)))
                  (unless (plusp time)
                    (error "Timer resolution is insufficient for ~A" operation))
                  (when (eq implementation :scalar)
                    (setf scalar-time time))
                  (format t "~12A ~12A ~8D ~14A ~10,3F ~9,2Fx~%"
                          operation type length implementation time
                          (/ scalar-time time)))))))))))

;; Scalar baselines for the reductions beyond sum and dot.
(macrolet ((define-reduction-loops (type asum argmax zero)
             `(progn
                (defun ,asum (a)
                  (declare (type (simple-array ,type (*)) a) (optimize (speed 3)))
                  (let ((result ,zero))
                    (declare (type ,type result))
                    (dotimes (i (length a) result)
                      (incf result (abs (aref a i))))))
                (defun ,argmax (a)
                  (declare (type (simple-array ,type (*)) a) (optimize (speed 3)))
                  (let ((index 0))
                    (declare (type fixnum index))
                    (dotimes (i (length a) index)
                      (when (> (aref a i) (aref a index)) (setf index i))))))))
  (define-reduction-loops single-float scalar-asum-f32 scalar-argmax-f32 0.0f0)
  (define-reduction-loops double-float scalar-asum-f64 scalar-argmax-f64 0.0d0))

(dolist (name '(scalar-asum-f32 scalar-argmax-f32 scalar-asum-f64 scalar-argmax-f64))
  (unless (compiled-function-p (symbol-function name))
    (compile name)))

(defun reduction-benchmark-function (operation implementation type a)
  (if (eq implementation :scalar)
      (let ((function (ecase operation
                        (:asum (if (eq type 'single-float) #'scalar-asum-f32 #'scalar-asum-f64))
                        (:argmax (if (eq type 'single-float) #'scalar-argmax-f32 #'scalar-argmax-f64))
                        (:nrm2 (if (eq type 'single-float) #'scalar-dot-f32 #'scalar-dot-f64)))))
        (if (eq operation :nrm2)
            (lambda () (sqrt (funcall function a a)))
            (lambda () (funcall function a))))
      (let ((function (ecase operation
                        (:asum #'trivial-simd:asum)
                        (:argmax #'trivial-simd:argmax)
                        (:nrm2 #'trivial-simd:nrm2))))
        (lambda () (funcall function a)))))

(format t "~%Reduction comparison (speedup = scalar / implementation)~%")
(format t "~12A ~12A ~8A ~14A ~10A ~10A~%"
        "Operation" "Type" "Elements" "Implementation" "us/call" "Speedup")
(dolist (type '(single-float double-float))
  (dolist (length '(32 1024 65536))
    (let ((a (benchmark-input :a length type)))
      (dolist (operation '(:asum :argmax :nrm2))
        (let ((scalar-time nil)
              (expected (funcall (reduction-benchmark-function operation :scalar type a))))
          (dolist (implementation (cons :scalar (benchmark-backends)))
            (with-benchmark-backend (implementation)
              (let* ((function (reduction-benchmark-function operation implementation type a))
                     (result (funcall function)))
                (unless (<= (abs (- result expected)) (* 1d-4 (max 1 (abs expected))))
                  (error "~A ~A benchmark mismatch: got ~S, expected ~S"
                         operation type result expected))
                (let ((time (benchmark-time function)))
                  (when (eq implementation :scalar)
                    (setf scalar-time time))
                  (format t "~12A ~12A ~8D ~14A ~10,3F ~9,2Fx~%"
                          operation type length implementation time
                          (/ scalar-time time)))))))))))

(format t "~%Kernel a*b+c~%")
(dolist (length '(32 1024 65536))
  (let ((a (make-array length :element-type 'single-float :initial-element 1.0))
        (b (make-array length :element-type 'single-float :initial-element 2.0))
        (c (make-array length :element-type 'single-float :initial-element 3.0))
        (out (make-array length :element-type 'single-float)))
    (flet ((time-call (name function)
             (format t "~7D elements ~18A ~,3F us/call~%"
                     length name (benchmark-time function))))
      (dolist (backend (append '(:lisp)
                               (when trivial-simd::*native-available-p* '(:native))
                               (when trivial-simd::*sbcl-simd-available-p* '(:sbcl))))
        (let ((trivial-simd::*backend* backend))
          (time-call (format nil "~A kernel" backend)
                  (lambda () (multiply-add out a b c)))
          (time-call (format nil "~A true FMA" backend)
                     (lambda () (rounded-multiply-add out a b c)))
          (time-call (format nil "~A two ops" backend)
                  (lambda ()
                    (trivial-simd:multiply! out a b)
                    (trivial-simd:add! out out c))))))))

(trivial-simd:define-kernel kernel-dot (a b) (trivial-simd:sum (* a b)))

(format t "~%Kernel sum(a*b)~%")
(dolist (type '(single-float double-float))
  (dolist (length '(32 1024 65536))
    (let ((a (make-array length :element-type type :initial-element (coerce 1 type)))
          (b (make-array length :element-type type :initial-element (coerce 2 type)))
          (out (make-array length :element-type type)))
      (dolist (backend (append '(:lisp)
                               (when trivial-simd::*native-available-p* '(:native))
                               (when trivial-simd::*sbcl-simd-available-p* '(:sbcl))))
        (let ((trivial-simd::*backend* backend))
          (dolist (case (list (cons "sum kernel" (lambda () (kernel-dot a b)))
                             (cons "multiply+sum" (lambda () (trivial-simd:multiply! out a b)
                                                          (trivial-simd:sum out)))
                             (cons "bulk dot" (lambda () (trivial-simd:dot a b)))))
            (unless (= (funcall (cdr case)) (coerce (* 2 length) type))
              (error "Reduction benchmark result mismatch"))
            (format t "~12A ~7D elements ~7A ~14A ~,3F us/call~%"
                    type length backend (car case)
                    (benchmark-time (cdr case)))))))))

#+ecl
(defun make-ecl-benchmark-kernel (definition reduction-p other-p)
  (let* ((name (gensym "BENCHMARK-KERNEL"))
         (expression (if other-p '(- (* a b) c) '(+ (* a b) c)))
         (form `(trivial-simd:define-kernel ,name (a b c)
                  ,(if reduction-p `(trivial-simd:sum ,expression) expression))))
    (unwind-protect
         (progn
           (ecase definition
             (:compiled (funcall (compile nil `(lambda () ,form))))
             (:eval (eval form)))
           (symbol-function name))
      (when (fboundp name) (fmakunbound name)))))

#+ecl
(defun ecl-kernel-call (function reduction-p output a b c)
  (if reduction-p
      (lambda () (funcall function a b c))
      (lambda () (funcall function output a b c))))

#+ecl
(defun check-ecl-kernel-result (result reduction-p output expected)
  (unless (if reduction-p
              (= result (* expected (length output)))
              (and (eq result output) (every (lambda (value) (= value expected)) output)))
    (error "ECL shared runner benchmark result mismatch")))

#+ecl
(defun benchmark-ecl-runners (definition reduction-p type)
  (let ((cold nil) (shared nil) (function nil) (other nil)
        (trivial-simd::*backend* :native))
    (dotimes (trial 5)
      (let* ((trivial-simd::*native-kernel-runners* (make-hash-table :test 'eql))
             #+threads (trivial-simd::*native-kernel-runner-lock* (mp:make-lock :name "benchmark runners"))
             (a (make-array 32 :element-type type :initial-element (coerce 1 type)))
             (b (make-array 32 :element-type type :initial-element (coerce 2 type)))
             (c (make-array 32 :element-type type :initial-element (coerce 3 type)))
             (output (make-array 32 :element-type type)))
        (setf function (make-ecl-benchmark-kernel definition reduction-p nil)
              other (make-ecl-benchmark-kernel definition reduction-p t))
        (dolist (case (list (cons function 5) (cons other -1)))
          (let* ((call (ecl-kernel-call (car case) reduction-p output a b c))
                 (start (get-internal-real-time))
                 (result (funcall call))
                 (elapsed (/ (* 1000.0d0 (- (get-internal-real-time) start))
                             internal-time-units-per-second)))
            (check-ecl-kernel-result result reduction-p output (cdr case))
            (if (eq (car case) function) (push elapsed cold) (push elapsed shared))))
        (unless (and (= 1 (hash-table-count trivial-simd::*native-kernel-runners*))
                     (functionp (gethash (+ 6 (if reduction-p 1 0)) trivial-simd::*native-kernel-runners*)))
          (error "Expected one compiled runner shared by both definitions"))))
    (format t "~A ~A ~A: cold signature ~,3F ms; shared-definition first call ~,3F ms (five-trial medians; clock ~,3F ms)~%"
            definition (if reduction-p :sum :elementwise) type
            (third (sort cold #'<)) (third (sort shared #'<))
            (/ 1000.0d0 internal-time-units-per-second))
    (dolist (count '(32 1024 65536))
      (let* ((a (make-array count :element-type type :initial-element (coerce 1 type)))
             (b (make-array count :element-type type :initial-element (coerce 2 type)))
             (c (make-array count :element-type type :initial-element (coerce 3 type)))
             (output (make-array count :element-type type)))
        (dolist (case (list (cons function 5) (cons other -1)))
          (let ((call (ecl-kernel-call (car case) reduction-p output a b c)))
            (check-ecl-kernel-result (funcall call) reduction-p output (cdr case))
            (format t "~A ~A ~A ~7D ~A: ~,3F us/call (warm)~%"
                    definition (if reduction-p :sum :elementwise) type count
                    (if (eq (car case) function) :first :shared)
                    (benchmark-time call))))))))

#+ecl
(when trivial-simd::*native-available-p*
  (format t "~%ECL shared native runners~%")
  (dolist (definition '(:compiled :eval))
    (dolist (reduction-p '(nil t))
      (dolist (type '(single-float double-float))
        (benchmark-ecl-runners definition reduction-p type)))))
