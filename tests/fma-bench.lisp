(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :asdf)
  (unless (find-package :ql)
    (dolist (candidate '("quicklisp/setup.lisp" ".roswell/lisp/quicklisp/setup.lisp"))
      (let ((path (merge-pathnames candidate (user-homedir-pathname))))
        (when (probe-file path) (load path) (return))))))

(defmacro load-benchmark-system ()
  `(progn
     (asdf:initialize-source-registry
      '(:source-registry
        (:directory ,(merge-pathnames "../" (uiop:pathname-directory-pathname
                                             (or *compile-file-truename* *load-truename*))))
        :inherit-configuration))
     (asdf:clear-system "trivial-simd")
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



(trivial-simd:define-kernel profile-fma (a b c) (trivial-simd:fma a b c))
(trivial-simd:define-kernel profile-fma-sum (a b c) (trivial-simd:sum (trivial-simd:fma a b c)))

(format t "~A ~A / ~A~%Source: ~A~%" (lisp-implementation-type)
        (lisp-implementation-version) (machine-type)
        (asdf:system-source-directory "trivial-simd"))
(format t "Hardware FMA: ~S~%" (trivial-simd::hardware-fma-available-p))
(format t "Type Elements Backend FMA-mode Kernel-us Sum-us~%")
(dolist (type '(single-float double-float))
  (dolist (count '(32 1024 65536))
    (let ((a (make-array count :element-type type :initial-element (coerce 1.25 type)))
          (b (make-array count :element-type type :initial-element (coerce 2.5 type)))
          (c (make-array count :element-type type :initial-element (coerce -0.5 type)))
          (out (make-array count :element-type type)))
      (dolist (backend (append '(:lisp)
                               (when trivial-simd::*sbcl-simd-available-p* '(:sbcl))
                               (when trivial-simd::*native-available-p* '(:native))))
        (dolist (mode (if (eq backend :native) '(:auto) '(:portable :native :auto)))
          (let ((trivial-simd::*backend* backend) (trivial-simd::*fma-mode* mode))
            (profile-fma out a b c)
            (unless (and (every (lambda (v) (= v 2.625)) out)
                         (= (profile-fma-sum a b c) (* count 2.625)))
              (error "FMA benchmark result mismatch"))
            (format t "~A ~D ~A ~A ~,3F ~,3F~%" type count backend mode
                    (benchmark-time (lambda () (profile-fma out a b c)))
                    (benchmark-time (lambda () (profile-fma-sum a b c))))))))))
