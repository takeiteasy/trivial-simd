(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :asdf)
  (load (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname))))

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


(trivial-simd:define-kernel slice-add (a b) (+ a b))
(trivial-simd:define-kernel slice-dot (a b) (trivial-simd:sum (* a b)))

(format t "~A ~A / ~A~%" (lisp-implementation-type) (lisp-implementation-version) (machine-type))
(format t "Source: ~A~%" (asdf:system-source-directory "trivial-simd"))
(format t "Type Parent Count Operation Copy-us Pointer-us~%")
(dolist (type '(single-float double-float))
  (dolist (case '((1024 32) (65536 32) (65536 1024) (65536 65536)))
    (destructuring-bind (parent count) case
      (let* ((a (make-array parent :element-type type :initial-element (coerce 1 type)))
             (b (make-array parent :element-type type :initial-element (coerce 2 type)))
             (out (make-array parent :element-type type :initial-element (coerce -1 type)))
             (start (- parent count))
             (trivial-simd::*backend* :native))
        (dolist (operation '(:add :sum :dot :kernel :kernel-sum))
          (let ((function
                  (ecase operation
                    (:add (lambda () (trivial-simd:add! out a b :start start :end parent)))
                    (:sum (lambda () (trivial-simd:sum a :start start :end parent)))
                    (:dot (lambda () (trivial-simd:dot a b :start start :end parent)))
                    (:kernel (lambda () (slice-add out a b :start start :end parent)))
                    (:kernel-sum (lambda () (slice-dot a b :start start :end parent))))))
            (format t "~A ~D ~D ~A" type parent count operation)
            (dolist (mode '(:copy :pointer))
              (let* ((trivial-simd::*native-array-access* mode)
                     (result (funcall function)))
                (unless (if (member operation '(:add :kernel))
                            (and (= 3 (aref out start)) (= 3 (aref out (1- parent)))
                                 (or (zerop start) (= -1 (aref out (1- start)))))
                            (= result (* count (if (eq operation :sum) 1 2))))
                  (error "Slice benchmark result mismatch"))
                (format t " ~,3F" (benchmark-time function))))
            (terpri)))))))
