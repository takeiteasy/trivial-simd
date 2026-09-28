(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :asdf)
  (load (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname))))

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
(when trivial-simd::*native-available-p*
  (format t "~%ECL eval-defined native kernel~%")
  (eval '(trivial-simd:define-kernel eval-multiply-add (a b c) (+ (* a b) c)))
  (let ((trivial-simd::*backend* :native))
    (dolist (length '(32 1024 65536))
      (let ((a (make-array length :element-type 'single-float :initial-element 1.0))
            (b (make-array length :element-type 'single-float :initial-element 2.0))
            (c (make-array length :element-type 'single-float :initial-element 3.0))
            (out (make-array length :element-type 'single-float)))
        (when (= length 32)
          (let ((start (get-internal-real-time)))
            (eval-multiply-add out a b c)
            (format t "First call including helper compilation: ~,3F ms~%"
                    (/ (* 1000.0 (- (get-internal-real-time) start))
                       internal-time-units-per-second))))
        (format t "~7D elements ~18A ~,3F us/call~%"
                length "eval kernel (warm)"
                (benchmark-time (lambda () (eval-multiply-add out a b c))))))))
