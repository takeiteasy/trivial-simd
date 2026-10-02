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

(defun scalar-add (out a b)
  (declare (type (simple-array single-float (*)) out a b) (optimize (speed 3)))
  (dotimes (i (length out) out)
    (setf (aref out i) (+ (aref a i) (aref b i)))))

(defun scalar-dot (a b)
  (declare (type (simple-array single-float (*)) a b) (optimize (speed 3)))
  (let ((result 0.0f0))
    (declare (type single-float result))
    (dotimes (i (length a) result)
      (incf result (* (aref a i) (aref b i))))))

(format t "~A ~A | machine ~A~%"
        (lisp-implementation-type) (lisp-implementation-version) (machine-type))
(format t "~%Call overhead, single-float, microseconds per call~%")
(format t "~12A ~8A ~8A ~10A~%" "Backend" "Op" "Elements" "us/call")

(defun benchmark-calls (backend inline operations)
  "Time OPERATIONS under BACKEND with the inline small-array path on or off."
  (let ((trivial-simd::*backend* backend)
        (trivial-simd::*inline-small-arrays* inline))
    (dolist (n (if inline '(4 32) '(4 32 1024)))
      (let ((a (make-array n :element-type 'single-float :initial-element 1.0f0))
            (b (make-array n :element-type 'single-float :initial-element 2.0f0))
            (out (make-array n :element-type 'single-float))
            (mask (make-array n :element-type '(unsigned-byte 8))))
        (dolist (operation operations)
          (format t "~12A ~8A ~8D ~10,4F~%"
                  (if inline :inline backend) operation n
                  (benchmark-time
                   (ecase operation
                     (:scalar-add (lambda () (scalar-add out a b)))
                     (:scalar-dot (lambda () (scalar-dot a b)))
                     (:add! (lambda () (trivial-simd:add! out a b)))
                     (:dot (lambda () (trivial-simd:dot a b)))
                     (:min! (lambda () (trivial-simd:min! out a b)))
                     (:compare! (lambda () (trivial-simd:compare! mask :lt a b)))))))))))

(benchmark-calls :lisp nil '(:scalar-add :scalar-dot))
(benchmark-calls :lisp t '(:add! :dot :min!))
(dolist (backend (append '(:lisp)
                         (when trivial-simd::*native-available-p* '(:native))
                         (when trivial-simd::*sbcl-simd-available-p* '(:sbcl))))
  (benchmark-calls backend nil '(:add! :dot :min! :compare!)))
