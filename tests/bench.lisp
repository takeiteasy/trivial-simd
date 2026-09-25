(require :asdf)
(load (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname)))
(asdf:load-asd (merge-pathnames "../trivial-simd.asd"
                                (uiop:pathname-directory-pathname *load-truename*)))
(asdf:load-system "trivial-simd")

(format t "Backend ~A~%" (trivial-simd:backend))
(dolist (length '(32 1024 65536))
  (let ((a (make-array length :element-type 'single-float :initial-element 1.0))
        (b (make-array length :element-type 'single-float :initial-element 2.0))
        (out (make-array length :element-type 'single-float)))
    (dolist (backend (append '(:lisp)
                             (when trivial-simd::*native-available-p* '(:native :native-copy))
                             (when trivial-simd::*sbcl-simd-available-p* '(:sbcl))))
      (let ((trivial-simd::*backend* (if (eq backend :native-copy) :native backend))
            (trivial-simd::*native-array-access*
              (if (eq backend :native-copy) :copy trivial-simd::*native-array-access*))
            (iterations (max 20 (floor 1000000 length))))
        (let ((start (get-internal-real-time)))
          (dotimes (i iterations)
            (trivial-simd:add! out a b))
          (format t "~7D elements ~7A ~,3F us/call~%"
                  length backend
                  (/ (* 1000000.0 (- (get-internal-real-time) start))
                     internal-time-units-per-second iterations)))))))
