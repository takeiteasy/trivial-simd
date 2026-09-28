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

(defun benchmark-time (function iterations)
  (funcall function)
  (second
   (sort (loop repeat 3 collect
           (let ((start (get-internal-real-time)))
             (dotimes (i iterations) (funcall function))
             (/ (* 1000000.0 (- (get-internal-real-time) start))
                internal-time-units-per-second iterations)))
         #'<)))

(format t "Backend ~A~%" (trivial-simd:backend))
(dolist (type '(single-float double-float))
  (dolist (length '(32 1024 65536))
    (let ((a (make-array length :element-type type :initial-element (coerce 1 type)))
          (b (make-array length :element-type type :initial-element (coerce 2 type)))
          (out (make-array length :element-type type)))
      (dolist (backend (append '(:lisp)
                               (when trivial-simd::*native-available-p* '(:native :native-copy))
                               (when trivial-simd::*sbcl-simd-available-p* '(:sbcl))))
        (let ((trivial-simd::*backend* (if (eq backend :native-copy) :native backend))
              (trivial-simd::*native-array-access*
                (if (eq backend :native-copy) :copy trivial-simd::*native-array-access*)))
          (format t "~A ~7D elements ~7A ~,3F us/call~%"
                  type length backend
                  (benchmark-time (lambda () (trivial-simd:add! out a b))
                                  (max 20 (floor 1000000 length)))))))))

(trivial-simd:define-kernel multiply-add (a b c) (+ (* a b) c))

(trivial-simd:define-kernel rounded-multiply-add (a b c) (trivial-simd:fma a b c))

(format t "~%Kernel a*b+c~%")
(dolist (length '(32 1024 65536))
  (let ((a (make-array length :element-type 'single-float :initial-element 1.0))
        (b (make-array length :element-type 'single-float :initial-element 2.0))
        (c (make-array length :element-type 'single-float :initial-element 3.0))
        (out (make-array length :element-type 'single-float))
        (iterations (max 20 (floor 1000000 length))))
    (flet ((time-call (name function)
             (format t "~7D elements ~18A ~,3F us/call~%"
                     length name (benchmark-time function iterations))))
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
(dolist (length '(32 1024 65536))
  (let ((a (make-array length :element-type 'single-float :initial-element 1.0))
        (b (make-array length :element-type 'single-float :initial-element 2.0))
        (out (make-array length :element-type 'single-float))
        (iterations (max 20 (floor 1000000 length))))
    (dolist (backend (append '(:lisp)
                             (when trivial-simd::*native-available-p* '(:native))
                             (when trivial-simd::*sbcl-simd-available-p* '(:sbcl))))
      (let ((trivial-simd::*backend* backend))
        (dolist (case (list (cons "sum kernel" (lambda () (kernel-dot a b)))
                           (cons "multiply+sum" (lambda () (trivial-simd:multiply! out a b)
                                                        (trivial-simd:sum out)))
                           (cons "bulk dot" (lambda () (trivial-simd:dot a b)))))
          (format t "~7D elements ~7A ~14A ~,3F us/call~%"
                  length backend (car case)
                  (benchmark-time (cdr case) iterations)))))))

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
                (benchmark-time (lambda () (eval-multiply-add out a b c))
                                (max 20 (floor 1000000 length))))))))
