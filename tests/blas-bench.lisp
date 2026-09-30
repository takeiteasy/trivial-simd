(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :asdf)
  (unless (find-package :ql)
    (dolist (candidate '("quicklisp/setup.lisp" ".roswell/lisp/quicklisp/setup.lisp"))
      (let ((path (merge-pathnames candidate (user-homedir-pathname))))
        (when (probe-file path) (load path) (return))))))

(defmacro load-blas-benchmark-system ()
  `(progn
     (asdf:load-asd ,(merge-pathnames "../trivial-simd.asd"
                                     (uiop:pathname-directory-pathname
                                      (or *compile-file-truename* *load-truename*))))
     (asdf:load-system "trivial-simd/blas")))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (load-blas-benchmark-system)
  (when (find-package :ql) (uiop:symbol-call :ql :quickload "cffi" :silent t)))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (load #.(merge-pathnames "benchmark-timing.lisp"
                          (uiop:pathname-directory-pathname
                           (or *compile-file-truename* *load-truename*)))))

(defpackage #:trivial-simd/blas-bench
  (:use #:cl))
(in-package #:trivial-simd/blas-bench)

(defmacro without-float-traps (&body body)
  #+sbcl `(sb-int:with-float-traps-masked (:divide-by-zero :invalid :overflow :inexact)
            ,@body)
  #+ecl `(progn (ext:trap-fpe t nil)
                (unwind-protect (progn ,@body) (ext:trap-fpe t t)))
  #-(or sbcl ecl) `(progn ,@body))

(defun load-reference-blas ()
  (loop for library in '("/System/Library/Frameworks/Accelerate.framework/Accelerate"
                         "libopenblas.so" "libblas.so.3" "libblas.so")
        thereis (handler-case
                    (progn (cffi:load-foreign-library library)
                           (cffi:foreign-symbol-pointer "cblas_daxpy"))
                  (error () nil))))

(defun benchmark-sizes ()
  (let ((value (uiop:getenv "BLAS_BENCH_SIZES")))
    (if (and value (plusp (length value)))
        (with-input-from-string (stream value)
          (loop for n = (read stream nil) while n collect n))
        '(16 64))))

(defun fill-random (vector seed)
  (let ((state seed))
    (dotimes (i (length vector) vector)
      (setf state (mod (+ (* state 1103515245) 12345) 2147483648))
      (setf (aref vector i)
            (coerce (/ (- (mod (ash state -8) 17) 8) 8) (array-element-type vector))))))

(defun dominant-square (n layout)
  (let* ((data (fill-random (make-array (* n n) :element-type 'double-float) 7))
         (view (trivial-simd/blas:make-matrix-view data n n :layout layout)))
    (dotimes (i n view)
      (incf (aref data (+ (* i n) i)) (coerce (* 2 n) 'double-float)))))

(defun foreign-copy (vector)
  (let* ((type (array-element-type vector))
         (ftype (if (eq type 'single-float) :float :double))
         (pointer (cffi:foreign-alloc ftype :count (length vector))))
    (dotimes (i (length vector) pointer)
      (setf (cffi:mem-aref pointer ftype i) (aref vector i)))))

(defun foreign-reset (destination source bytes)
  (cffi:foreign-funcall "memcpy" :pointer destination :pointer source
                        :unsigned-long bytes :pointer))

(defvar *rows* nil)

(defun report (routine layout n ours reference)
  (format t "~8A ~13A ~5D ~12,3F ~12,3F ~9,2F~%" routine layout n ours reference
          (/ ours reference))
  (finish-output)
  (push (list routine layout n ours reference) *rows*))

(defmacro timed (&body body)
  `(cl-user::benchmark-time (lambda () ,@body)))

(defmacro timed-reference (&body body)
  `(cl-user::benchmark-time (lambda () (without-float-traps ,@body))))

(defun order (layout) (if (eq layout :row-major) 101 102))

(defun bench-gemm (layout n)
  (let* ((a (fill-random (make-array (* n n) :element-type 'double-float) 1))
         (b (fill-random (make-array (* n n) :element-type 'double-float) 2))
         (c (make-array (* n n) :element-type 'double-float :initial-element 0d0))
         (av (trivial-simd/blas:make-matrix-view a n n :layout layout))
         (bv (trivial-simd/blas:make-matrix-view b n n :layout layout))
         (cv (trivial-simd/blas:make-matrix-view c n n :layout layout))
         (fa (foreign-copy a)) (fb (foreign-copy b)) (fc (foreign-copy c)))
    (report "dgemm" layout n
            (timed (trivial-simd/blas:dgemm :no-transpose :no-transpose
                                           1d0 av bv 0d0 cv))
            (timed-reference
              (cffi:foreign-funcall "cblas_dgemm" :int (order layout) :int 111 :int 111
                                    :int n :int n :int n :double 1d0 :pointer fa :int n
                                    :pointer fb :int n :double 0d0 :pointer fc :int n
                                    :void)))))

(defun bench-syrk (layout n)
  (let* ((a (fill-random (make-array (* n n) :element-type 'double-float) 1))
         (c (make-array (* n n) :element-type 'double-float :initial-element 0d0))
         (av (trivial-simd/blas:make-matrix-view a n n :layout layout))
         (cv (trivial-simd/blas:make-matrix-view c n n :layout layout))
         (fa (foreign-copy a)) (fc (foreign-copy c)))
    (report "dsyrk" layout n
            (timed (trivial-simd/blas:dsyrk :upper :no-transpose 1d0 av 0d0 cv))
            (timed-reference
              (cffi:foreign-funcall "cblas_dsyrk" :int (order layout) :int 121 :int 111
                                    :int n :int n :double 1d0 :pointer fa :int n
                                    :double 0d0 :pointer fc :int n :void)))))

(defun bench-trsm (layout n)
  (let* ((a (dominant-square n layout))
         (b (fill-random (make-array (* n n) :element-type 'double-float) 2))
         (saved (copy-seq b))
         (bv (trivial-simd/blas:make-matrix-view b n n :layout layout))
         (fa (foreign-copy (trivial-simd/blas:matrix-view-data a)))
         (fb (foreign-copy b)) (fsaved (foreign-copy b)))
    (report "dtrsm" layout n
            (timed (replace b saved)
                   (trivial-simd/blas:dtrsm :left :upper :no-transpose :non-unit
                                           1d0 a bv))
            (timed-reference
              (foreign-reset fb fsaved (* 8 n n))
              (cffi:foreign-funcall "cblas_dtrsm" :int (order layout) :int 141 :int 121
                                    :int 111 :int 131 :int n :int n :double 1d0
                                    :pointer fa :int n :pointer fb :int n :void)))))

(defun bench-gemv (layout n)
  (let* ((a (fill-random (make-array (* n n) :element-type 'double-float) 1))
         (x (fill-random (make-array n :element-type 'double-float) 2))
         (y (make-array n :element-type 'double-float :initial-element 0d0))
         (av (trivial-simd/blas:make-matrix-view a n n :layout layout))
         (fa (foreign-copy a)) (fx (foreign-copy x)) (fy (foreign-copy y)))
    (report "dgemv" layout n
            (timed (trivial-simd/blas:dgemv :no-transpose 1d0 av x 1 0d0 y 1))
            (timed-reference
              (cffi:foreign-funcall "cblas_dgemv" :int (order layout) :int 111 :int n
                                    :int n :double 1d0 :pointer fa :int n :pointer fx
                                    :int 1 :double 0d0 :pointer fy :int 1 :void)))))

(defun bench-gbmv (layout n)
  (let* ((kl 3) (ku 2) (width (+ kl ku 1))
         (data (fill-random (make-array (* n width) :element-type 'double-float) 1))
         (x (fill-random (make-array n :element-type 'double-float) 2))
         (y (make-array n :element-type 'double-float :initial-element 0d0))
         (view (trivial-simd/blas:make-band-matrix-view data n n :kl kl :ku ku
                                                                 :layout layout))
         (fa (foreign-copy data)) (fx (foreign-copy x)) (fy (foreign-copy y)))
    (report "dgbmv" layout n
            (timed (trivial-simd/blas:dgbmv :no-transpose 1d0 view x 1 0d0 y 1))
            (timed-reference
              (cffi:foreign-funcall "cblas_dgbmv" :int (order layout) :int 111 :int n
                                    :int n :int kl :int ku :double 1d0 :pointer fa
                                    :int width :pointer fx :int 1 :double 0d0
                                    :pointer fy :int 1 :void)))))

(defun bench-trsv (layout n)
  (let* ((a (dominant-square n layout))
         (x (fill-random (make-array n :element-type 'double-float) 2))
         (saved (copy-seq x))
         (fa (foreign-copy (trivial-simd/blas:matrix-view-data a)))
         (fx (foreign-copy x)) (fsaved (foreign-copy x)))
    (report "dtrsv" layout n
            (timed (replace x saved)
                   (trivial-simd/blas:dtrsv :upper :no-transpose :non-unit a x 1))
            (timed-reference
              (foreign-reset fx fsaved (* 8 n))
              (cffi:foreign-funcall "cblas_dtrsv" :int (order layout) :int 121 :int 111
                                    :int 131 :int n :pointer fa :int n :pointer fx
                                    :int 1 :void)))))

(defun bench-ger (layout n)
  (let* ((a (make-array (* n n) :element-type 'double-float :initial-element 0d0))
         (x (fill-random (make-array n :element-type 'double-float) 1))
         (y (fill-random (make-array n :element-type 'double-float) 2))
         (view (trivial-simd/blas:make-matrix-view a n n :layout layout))
         (fa (foreign-copy a)) (fx (foreign-copy x)) (fy (foreign-copy y)))
    (report "dger" layout n
            (timed (trivial-simd/blas:dger 1d-6 x 1 y 1 view))
            (timed-reference
              (cffi:foreign-funcall "cblas_dger" :int (order layout) :int n :int n
                                    :double 1d-6 :pointer fx :int 1 :pointer fy :int 1
                                    :pointer fa :int n :void)))))

(defun run-blas-benchmarks ()
  (unless (load-reference-blas)
    (error "No system CBLAS library found"))
  (format t "# ~A ~A; times are microseconds per call; ratio = ours / CBLAS~%"
          (lisp-implementation-type) (lisp-implementation-version))
  (format t "~8A ~13A ~5A ~12A ~12A ~9A~%" "routine" "layout" "n" "ours" "cblas" "ratio")
  (dolist (n (benchmark-sizes))
    (dolist (layout '(:row-major :column-major))
      (dolist (bench '(bench-gemm bench-syrk bench-trsm bench-gemv bench-gbmv
                       bench-trsv bench-ger))
        (funcall bench layout n)))))

(run-blas-benchmarks)
