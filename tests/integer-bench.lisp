(require :asdf)
(unless (find-package :ql)
  (dolist (candidate '("quicklisp/setup.lisp" ".roswell/lisp/quicklisp/setup.lisp"))
    (let ((path (merge-pathnames candidate (user-homedir-pathname))))
      (when (probe-file path) (load path) (return)))))
(asdf:load-asd (merge-pathnames "../trivial-simd.asd"
                                (uiop:pathname-directory-pathname *load-truename*)))
(asdf:load-system "trivial-simd")
(load (merge-pathnames "benchmark-timing.lisp" *load-truename*))

(trivial-simd:define-kernel integer-benchmark-kernel (a b) (+ (* a b) 3))

(format t "Backend Type Elements Add-us Dot-us Kernel-us~%")
(dolist (backend (if trivial-simd::*native-available-p* '(:lisp :native) '(:lisp)))
  (dolist (type '((signed-byte 8) (unsigned-byte 64)))
    (dolist (count '(32 1024 65536))
      (let* ((trivial-simd::*backend* backend)
             (a (make-array count :element-type type :initial-element 7))
             (b (make-array count :element-type type :initial-element 3))
             (out (make-array count :element-type type)))
        (format t "~A ~A ~D ~,3F ~,3F ~,3F~%"
                backend type count
                (benchmark-time (lambda () (trivial-simd:add! out a b)))
                (benchmark-time (lambda () (trivial-simd:dot a b)))
                (benchmark-time (lambda () (integer-benchmark-kernel out a b))))))))
