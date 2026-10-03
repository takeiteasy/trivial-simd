(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :asdf))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (find-package :ql)
    (dolist (candidate '("quicklisp/setup.lisp" ".roswell/lisp/quicklisp/setup.lisp"))
      (let ((path (merge-pathnames candidate (user-homedir-pathname))))
        (when (probe-file path) (load path) (return)))))
  (let ((root (merge-pathnames "../" (uiop:pathname-directory-pathname
                                      (or *compile-file-truename* *load-truename*)))))
    (asdf:initialize-source-registry
     `(:source-registry (:directory ,root) :inherit-configuration))
    (asdf:load-system "trivial-simd")
    (load (merge-pathnames "tests/benchmark-timing.lisp" root))))

(trivial-simd:define-kernel view-bench-product (a b) (trivial-simd:sum (* a b)))

;; One operand is a view of foreign memory; the others are Lisp vectors. Each cell
;; is microseconds with a view / with a Lisp vector.
(let* ((count 1048576)
       (pointer (cffi:foreign-alloc :float :count count :initial-element 1.0))
       (view (trivial-simd:make-vector-view pointer :f32 count))
       (single (make-array count :element-type 'single-float :initial-element 1.0))
       (other (make-array count :element-type 'single-float :initial-element 2.0))
       (wide (make-array count :element-type 'double-float :initial-element 0d0)))
  (unwind-protect
       (progn
         (format t "~&~A ~A, ~:D single-floats (microseconds: view / Lisp vector)~%"
                 (lisp-implementation-type) (lisp-implementation-version) count)
         (dolist (backend (remove-if-not
                           (lambda (backend)
                             (or (eq backend :lisp)
                                 (and (eq backend :native) trivial-simd::*native-available-p*)
                                 (and (eq backend :sbcl)
                                      (boundp 'trivial-simd::*sbcl-simd-available-p*)
                                      (symbol-value 'trivial-simd::*sbcl-simd-available-p*))))
                           '(:native :sbcl :lisp)))
           (let ((trivial-simd::*backend* backend))
             (flet ((row (name function)
                      (format t "~&~12A ~22A ~10,0F / ~10,0F~%" backend name
                              (benchmark-time (lambda () (funcall function view)))
                              (benchmark-time (lambda () (funcall function single))))))
               (row "dot" (lambda (x) (trivial-simd:dot x other)))
               (row "axpy!" (lambda (x) (trivial-simd:axpy! other 0.5 x)))
               (row "convert! to f64" (lambda (x) (trivial-simd:convert! wide x)))
               (row "(sum (* a b))" (lambda (x) (view-bench-product x other)))))))
    (cffi:foreign-free pointer)))
