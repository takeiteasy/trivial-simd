(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :asdf)
  (unless (find-package :ql)
    (let ((setup (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname))))
      (when (probe-file setup) (load setup)))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (asdf:load-asd #.(merge-pathnames "../trivial-simd.asd"
                                   (uiop:pathname-directory-pathname
                                    (or *compile-file-truename* *load-truename*))))
  (asdf:load-system "trivial-simd")
  (load #.(merge-pathnames "benchmark-timing.lisp"
                          (uiop:pathname-directory-pathname
                           (or *compile-file-truename* *load-truename*)))))

(trivial-simd:define-kernel row-bench-dot (a b) (trivial-simd:sum (* a b)))

(defun benchmark-row-case (type rows width backend)
  (let* ((a (make-array (* rows width) :element-type type
                        :initial-contents (loop for i below (* rows width)
                                                collect (coerce (/ (- (mod i 13) 6) 4) type))))
         (b (make-array width :element-type type
                        :initial-contents (loop for i below width collect (coerce (/ (1+ (mod i 7)) 2) type))))
         (separate (make-array rows :element-type type))
         (batch (make-array rows :element-type type))
         (trivial-simd::*backend* (if (eq backend :native-copy) :native backend))
         (trivial-simd::*native-array-access* (if (eq backend :native-copy) :copy :pointer)))
    (flet ((single-calls ()
             (dotimes (row rows separate)
               (setf (aref separate row)
                     (row-bench-dot a b :end width :a-start (* row width)))))
           (batch-call ()
             (row-bench-dot a b :rows rows :row-length width :b-row-stride 0 :destination batch)))
      (single-calls)
      (batch-call)
      (assert (equalp separate batch))
      (let ((single-time (benchmark-time #'single-calls))
            (batch-time (benchmark-time #'batch-call)))
        (format t "~&~12A ~11A ~4D ~5D ~12,3F ~12,3F ~11,4F ~11,4F ~7,2Fx~%"
                backend type rows width single-time batch-time
                (/ single-time rows) (/ batch-time rows) (/ single-time batch-time))))))

(format t "~&~A ~A, ~A ~A~%" (lisp-implementation-type) (lisp-implementation-version)
        (machine-type) (machine-version))
(format t "Native batching: ~A~%" trivial-simd::*native-kernel-rows-available-p*)
(format t "Backend      Type        Rows Width Separate-us     Batch-us Separate/row   Batch/row Speedup~%")
(dolist (backend (append '(:lisp)
                         (when trivial-simd::*native-kernel-rows-available-p* '(:native :native-copy))
                         (when trivial-simd::*sbcl-simd-available-p* '(:sbcl))))
  (dolist (type '(single-float double-float))
    (dolist (rows '(1 32 1024))
      (dolist (width '(4 32 256 1024))
        (benchmark-row-case type rows width backend)))))
