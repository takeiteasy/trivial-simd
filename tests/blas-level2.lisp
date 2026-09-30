(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(test blas-level2-matrix-views
  (let* ((data (blas-vector 'single-float '(0.0 1.0 2.0 99.0
                                            3.0 4.0 5.0 99.0)))
         (view (trivial-simd/blas:make-matrix-view
                data 2 3 :leading-dimension 4))
         (sub (trivial-simd/blas:matrix-subview view 0 1 2 2)))
    (is (= (trivial-simd/blas:matrix-ref sub 1 1) 5.0))
    (setf (trivial-simd/blas:matrix-ref sub 0 0) 7.0)
    (is (= (aref data 1) 7.0))
    (signals error (trivial-simd/blas:make-matrix-view data 3 3
                                                       :leading-dimension 4))
    (signals error (trivial-simd/blas:make-matrix-view data 2 3
                                                       :leading-dimension 2))
    (signals error (trivial-simd/blas:make-packed-matrix-view data 4))
    (signals error (trivial-simd/blas:matrix-subview view 1 2 2 1))
    (is (trivial-simd/blas:matrix-view-p
         (trivial-simd/blas:matrix-subview view 2 3 0 0)))))

(defun level2-call (name &rest arguments)
  (apply (symbol-function (find-symbol (string-upcase name) :trivial-simd/blas))
         arguments))

(test blas-level2-gemv-and-ger
  (dolist (case '((single-float "s" 1.0f0 2.0f0)
                  (double-float "d" 1.0d0 2.0d0)
                  ((complex single-float) "c" #C(1.0f0 1.0f0) #C(2.0f0 -1.0f0))
                  ((complex double-float) "z" #C(1.0d0 1.0d0) #C(2.0d0 -1.0d0))))
    (destructuring-bind (type prefix a b) case
      (dolist (layout '(:row-major :column-major))
        (let* ((data (make-array 10 :element-type type :initial-element a))
               (view (trivial-simd/blas:make-matrix-view
                      data 2 3 :layout layout :leading-dimension 4))
               (x (make-array 3 :element-type type :initial-element b))
               (y (make-array 2 :element-type type :initial-element a)))
          (level2-call (concatenate 'string prefix "gemv")
                       :no-transpose a view x 1 a y 1)
          (is (= (aref y 0) (+ (* a a) (* a 3 a b))))
          (is (= (aref y 1) (aref y 0)))
          (let ((before (trivial-simd/blas:matrix-ref view 1 2)))
            (level2-call (if (member prefix '("c" "z") :test #'string=)
                             (concatenate 'string prefix "geru")
                             (concatenate 'string prefix "ger"))
                         a y 1 x 1 view)
            (is (= (trivial-simd/blas:matrix-ref view 1 2)
                   (+ before (* a (aref y 1) b))))))))))

(test blas-level2-transpose-and-strides
  (let* ((data (blas-vector 'double-float '(1d0 2d0 3d0 0d0 4d0 5d0 6d0)))
         (view (trivial-simd/blas:make-matrix-view
                data 2 3 :leading-dimension 4))
         (x (blas-vector 'double-float '(1d0 0d0 2d0)))
         (y (blas-vector 'double-float '(10d0 0d0 20d0 0d0 30d0))))
    (trivial-simd/blas:dgemv :transpose 1d0 view x -2 1d0 y 2)
    (is (equalp y (blas-vector 'double-float '(16d0 0d0 29d0 0d0 42d0))))
    (signals error (trivial-simd/blas:dgemv :bad 1d0 view x 1 0d0 y 1))
    (signals error (trivial-simd/blas:dgemv :no-transpose 1d0 view x 0 0d0 y 1))))

(test blas-level2-triangular
  (dolist (layout '(:row-major :column-major))
    (let* ((data (blas-vector 'double-float
                              (if (eq layout :row-major)
                                  '(2d0 3d0 4d0 0d0 5d0 6d0 0d0 0d0 7d0)
                                  '(2d0 0d0 0d0 3d0 5d0 0d0 4d0 6d0 7d0))))
           (view (trivial-simd/blas:make-matrix-view data 3 3 :layout layout))
           (x (blas-vector 'double-float '(1d0 2d0 3d0))))
      (trivial-simd/blas:dtrmv :upper :no-transpose :non-unit view x 1)
      (is (equalp x (blas-vector 'double-float '(20d0 28d0 21d0))))
      (trivial-simd/blas:dtrsv :upper :no-transpose :non-unit view x 1)
      (is (equalp x (blas-vector 'double-float '(1d0 2d0 3d0)))))))

(test blas-level2-packed-and-band
  (dolist (layout '(:row-major :column-major))
    (let* ((packed (blas-vector 'single-float
                                (if (eq layout :row-major)
                                    '(2.0 3.0 4.0 5.0 6.0 7.0)
                                    '(2.0 3.0 5.0 4.0 6.0 7.0))))
           (view (trivial-simd/blas:make-packed-matrix-view packed 3
                                                             :layout layout))
           (x (blas-vector 'single-float '(1.0 2.0 3.0))))
      (trivial-simd/blas:stpmv :upper :no-transpose :non-unit view x 1)
      (is (equalp x (blas-vector 'single-float '(20.0 28.0 21.0)))))
    (let* ((band (blas-vector 'single-float
                              (if (eq layout :row-major)
                                  '(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 0.0)
                                  '(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 0.0))))
           (view (trivial-simd/blas:make-band-matrix-view
                  band 3 3 :kl 1 :ku 1 :layout layout
                  :leading-dimension (if (eq layout :row-major) 3 3)))
           (x (blas-vector 'single-float '(1.0 1.0 1.0)))
           (y (blas-vector 'single-float '(0.0 0.0 0.0))))
      (trivial-simd/blas:sgbmv :no-transpose 1.0 view x 1 0.0 y 1)
      (is (= (length y) 3)))))

(test blas-level2-hermitian
  (let* ((data (blas-vector '(complex single-float)
                            '(#C(2.0 9.0) #C(1.0 2.0)
                              #C(0.0 0.0) #C(3.0 8.0))))
         (view (trivial-simd/blas:make-matrix-view data 2 2))
         (x (blas-vector '(complex single-float)
                         '(#C(1.0 0.0) #C(0.0 1.0))))
         (y (blas-vector '(complex single-float)
                         '(#C(0.0 0.0) #C(0.0 0.0)))))
    (trivial-simd/blas:chemv :upper #C(1.0 0.0) view x 1 #C(0.0 0.0) y 1)
    (is (= (aref y 0) #C(0.0 1.0)))
    (is (= (aref y 1) #C(1.0 1.0)))
    (trivial-simd/blas:cher :upper 1.0 x 1 view)
    (is (zerop (imagpart (trivial-simd/blas:matrix-ref view 0 0))))))

(test blas-level2-reference-gemv
  (when (reference-blas-available-p)
    (dolist (layout '((:row-major 101) (:column-major 102)))
      (destructuring-bind (layout-symbol enum) layout
        (let* ((data (blas-vector 'double-float '(1d0 2d0 3d0 4d0
                                                  5d0 6d0 7d0 8d0 9d0)))
               (view (trivial-simd/blas:make-matrix-view
                      data 3 3 :layout layout-symbol))
               (x (blas-vector 'double-float '(2d0 3d0 4d0)))
               (y (blas-vector 'double-float '(1d0 2d0 3d0))))
          (cffi:with-foreign-objects ((fa :double 9) (fx :double 3) (fy :double 3))
            (dotimes (i 9) (setf (cffi:mem-aref fa :double i) (aref data i)))
            (dotimes (i 3)
              (setf (cffi:mem-aref fx :double i) (aref x i)
                    (cffi:mem-aref fy :double i) (aref y i)))
            (cffi:foreign-funcall "cblas_dgemv" :int enum :int 111 :int 3 :int 3
                                  :double 1.25d0 :pointer fa :int 3 :pointer fx :int 1
                                  :double 0.5d0 :pointer fy :int 1 :void)
            (trivial-simd/blas:dgemv :no-transpose 1.25d0 view x 1 0.5d0 y 1)
            (dotimes (i 3)
              (is (close-enough-p (aref y i) (cffi:mem-aref fy :double i)
                                  'double-float)))))))))

(test blas-level2-reference-band-and-packed
  (when (reference-blas-available-p)
    (dolist (layout '((:row-major 101) (:column-major 102)))
      (destructuring-bind (layout-symbol enum) layout
        (let* ((band (blas-vector 'double-float
                                  '(0d0 1d0 2d0 3d0 4d0 5d0 6d0 7d0 0d0)))
               (view (trivial-simd/blas:make-band-matrix-view
                      band 3 3 :kl 1 :ku 1 :layout layout-symbol))
               (x (blas-vector 'double-float '(1d0 2d0 3d0)))
               (y (blas-vector 'double-float '(4d0 5d0 6d0))))
          (cffi:with-foreign-objects ((fa :double 9) (fx :double 3) (fy :double 3))
            (dotimes (i 9) (setf (cffi:mem-aref fa :double i) (aref band i)))
            (dotimes (i 3)
              (setf (cffi:mem-aref fx :double i) (aref x i)
                    (cffi:mem-aref fy :double i) (aref y i)))
            (cffi:foreign-funcall "cblas_dgbmv" :int enum :int 111 :int 3 :int 3
                                  :int 1 :int 1 :double 1d0 :pointer fa :int 3
                                  :pointer fx :int 1 :double 0d0 :pointer fy :int 1
                                  :void)
            (trivial-simd/blas:dgbmv :no-transpose 1d0 view x 1 0d0 y 1)
            (dotimes (i 3)
              (is (close-enough-p (aref y i) (cffi:mem-aref fy :double i)
                                  'double-float)))))
        (let* ((packed (blas-vector 'double-float
                                    (if (eq layout-symbol :row-major)
                                        '(2d0 3d0 4d0 5d0 6d0 7d0)
                                        '(2d0 3d0 5d0 4d0 6d0 7d0))))
               (view (trivial-simd/blas:make-packed-matrix-view
                      packed 3 :layout layout-symbol))
               (x (blas-vector 'double-float '(1d0 2d0 3d0))))
          (cffi:with-foreign-objects ((fa :double 6) (fx :double 3))
            (dotimes (i 6) (setf (cffi:mem-aref fa :double i) (aref packed i)))
            (dotimes (i 3) (setf (cffi:mem-aref fx :double i) (aref x i)))
            (cffi:foreign-funcall "cblas_dtpmv" :int enum :int 121 :int 111
                                  :int 131 :int 3 :pointer fa :pointer fx :int 1
                                  :void)
            (trivial-simd/blas:dtpmv :upper :no-transpose :non-unit view x 1)
            (dotimes (i 3)
              (is (close-enough-p (aref x i) (cffi:mem-aref fx :double i)
                                  'double-float)))))))))

(test blas-level2-zero-scalars
  (let* ((data (blas-vector 'single-float '(1.0 2.0 3.0 4.0)))
         (view (trivial-simd/blas:make-matrix-view data 2 2))
         (x (blas-vector 'single-float '(1.0 1.0)))
         (y (blas-vector 'single-float '(5.0 6.0))))
    (trivial-simd/blas:sgemv :no-transpose 0.0 view x 1 1.0 y 1)
    (is (equalp y (blas-vector 'single-float '(5.0 6.0))))
    (trivial-simd/blas:sgemv :no-transpose 1.0 view x 1 0.0 y 1)
    (is (equalp y (blas-vector 'single-float '(3.0 7.0))))
    (trivial-simd/blas:sger 0.0 x 1 y 1 view)
    (is (equalp data (blas-vector 'single-float '(1.0 2.0 3.0 4.0))))))

(test blas-level2-family-smoke
  (dolist (case '((single-float "s" 1.0f0 0.0f0)
                  (double-float "d" 1.0d0 0.0d0)
                  ((complex single-float) "c" #C(1.0f0 0.0f0) #C(0.0f0 0.0f0))
                  ((complex double-float) "z" #C(1.0d0 0.0d0) #C(0.0d0 0.0d0))))
    (destructuring-bind (type prefix one zero) case
      (flet ((call (suffix &rest arguments)
               (apply #'level2-call
                      (concatenate 'string prefix suffix) arguments))
             (fresh (kind)
               (case kind
                 (:dense (trivial-simd/blas:make-matrix-view
                          (blas-vector type (list one zero zero one)) 2 2))
                 (:packed (trivial-simd/blas:make-packed-matrix-view
                           (blas-vector type (list one zero one)) 2))
                 (otherwise
                  (trivial-simd/blas:make-band-matrix-view
                   (blas-vector type (list one one)) 2 2
                   :kind kind
                   :kl 0 :ku 0)))))
        (let ((x (blas-vector type (list one one)))
              (y (blas-vector type (list zero zero))))
          (dolist (family '(("gemv" :dense) ("gbmv" :general)))
            (destructuring-bind (suffix kind) family
              (call suffix :no-transpose one (fresh kind) x 1 zero y 1)
              (is (equalp y x))))
          (when (member prefix '("s" "d") :test #'string=)
            (dolist (family '(("symv" :dense) ("spmv" :packed)))
              (destructuring-bind (suffix kind) family
                (call suffix :upper one (fresh kind) x 1 zero y 1)
                (is (equalp y x))))
            (call "sbmv" :upper one (fresh :symmetric) x 1 zero y 1)
            (is (equalp y x)))
          (when (member prefix '("c" "z") :test #'string=)
            (dolist (family '(("hemv" :dense) ("hbmv" :hermitian)
                              ("hpmv" :packed)))
              (destructuring-bind (suffix kind) family
                (call suffix :upper one (fresh kind) x 1 zero y 1)
                (is (equalp y x)))))
          (dolist (family '(("trmv" :dense) ("tbmv" :triangular)
                            ("tpmv" :packed) ("trsv" :dense)
                            ("tbsv" :triangular) ("tpsv" :packed)))
            (destructuring-bind (suffix kind) family
              (let ((work (copy-seq x)))
                (call suffix :upper :no-transpose :non-unit
                      (fresh kind) work 1)
                (is (equalp work x)))))
          (let ((ger (if (member prefix '("c" "z") :test #'string=)
                         "geru" "ger")))
            (is (trivial-simd/blas:matrix-view-p
                 (call ger one x 1 x 1 (fresh :dense)))))
          (when (member prefix '("c" "z") :test #'string=)
            (is (trivial-simd/blas:matrix-view-p
                 (call "gerc" one x 1 x 1 (fresh :dense)))))
          (when (member prefix '("s" "d") :test #'string=)
            (dolist (family '(("syr" :dense) ("spr" :packed)
                              ("syr2" :dense) ("spr2" :packed)))
              (destructuring-bind (suffix kind) family
                (is (trivial-simd/blas:matrix-view-p
                     (if (search "2" suffix)
                         (call suffix :upper one x 1 x 1 (fresh kind))
                         (call suffix :upper one x 1 (fresh kind))))))))
          (when (member prefix '("c" "z") :test #'string=)
            (let ((real-one (realpart one)))
              (dolist (family '(("her" :dense) ("hpr" :packed)
                                ("her2" :dense) ("hpr2" :packed)))
                (destructuring-bind (suffix kind) family
                  (is (trivial-simd/blas:matrix-view-p
                       (if (search "2" suffix)
                           (call suffix :upper one x 1 x 1 (fresh kind))
                           (call suffix :upper real-one x 1 (fresh kind))))))))))))))

(test blas-level2-lower-conjugate-triangle
  (let* ((data (blas-vector '(complex single-float)
                            '(#C(1.0 0.0) #C(0.0 0.0)
                              #C(2.0 1.0) #C(3.0 0.0))))
         (view (trivial-simd/blas:make-matrix-view data 2 2))
         (x (blas-vector '(complex single-float)
                         '(#C(1.0 1.0) #C(2.0 -1.0)))))
    (trivial-simd/blas:ctrmv :lower :conjugate-transpose :non-unit view x 1)
    (is (= (aref x 0) #C(4.0 -3.0)))
    (is (= (aref x 1) #C(6.0 -3.0)))
    (trivial-simd/blas:ctrsv :lower :conjugate-transpose :non-unit view x 1)
    (is (= (aref x 0) #C(1.0 1.0)))
    (is (= (aref x 1) #C(2.0 -1.0)))))

(test blas-level2-rank-update-values
  (let* ((data (blas-vector 'double-float '(1d0 0d0 0d0 1d0)))
         (view (trivial-simd/blas:make-matrix-view data 2 2))
         (x (blas-vector 'double-float '(2d0 3d0)))
         (y (blas-vector 'double-float '(4d0 5d0))))
    (trivial-simd/blas:dsyr2 :upper 2d0 x 1 y 1 view)
    (is (= (trivial-simd/blas:matrix-ref view 0 0) 33d0))
    (is (= (trivial-simd/blas:matrix-ref view 0 1) 44d0))
    (is (= (trivial-simd/blas:matrix-ref view 1 0) 0d0)))
  (let* ((data (blas-vector '(complex double-float)
                            '(#C(1d0 9d0) #C(0d0 0d0)
                              #C(0d0 0d0) #C(1d0 9d0))))
         (view (trivial-simd/blas:make-matrix-view data 2 2))
         (x (blas-vector '(complex double-float)
                         '(#C(1d0 1d0) #C(2d0 0d0)))))
    (trivial-simd/blas:zher :upper 2d0 x 1 view)
    (is (= (trivial-simd/blas:matrix-ref view 0 0) #C(5d0 0d0)))
    (is (= (trivial-simd/blas:matrix-ref view 0 1) #C(4d0 4d0)))
    (is (= (trivial-simd/blas:matrix-ref view 1 0) #C(0d0 0d0)))))

(test blas-level2-convenience
  (let* ((data (blas-vector 'double-float '(2d0 1d0 0d0 3d0)))
         (view (trivial-simd/blas:make-matrix-view data 2 2))
         (x (blas-vector 'double-float '(1d0 2d0)))
         (y (blas-vector 'double-float '(0d0 0d0))))
    (is (eq y (trivial-simd/blas/convenience:gemv! y 1d0 view x)))
    (is (equalp y (blas-vector 'double-float '(4d0 6d0))))
    (trivial-simd/blas/convenience:trsv! y view)
    (is (equalp y x))
    (trivial-simd/blas/convenience:ger! view 1d0 x x)
    (is (= (trivial-simd/blas:matrix-ref view 1 1) 7d0))))

(test blas-level2-reference-symmetric-and-rank
  (when (reference-blas-available-p)
    (dolist (layout '((:row-major 101) (:column-major 102)))
      (destructuring-bind (layout-symbol enum) layout
        (let* ((original (blas-vector 'double-float
                                      '(2d0 3d0 4d0 5d0 6d0 7d0 8d0 9d0 10d0)))
               (x (blas-vector 'double-float '(1d0 2d0 3d0)))
               (y (blas-vector 'double-float '(4d0 5d0 6d0))))
          (cffi:with-foreign-objects ((fa :double 9) (fx :double 3) (fy :double 3))
            (dotimes (i 3)
              (setf (cffi:mem-aref fx :double i) (aref x i)
                    (cffi:mem-aref fy :double i) (aref y i)))
            (flet ((reset-matrix ()
                     (dotimes (i 9)
                       (setf (cffi:mem-aref fa :double i) (aref original i)))
                     (dotimes (i 3)
                       (setf (cffi:mem-aref fy :double i) (aref y i))))
                   (compare-matrix (data)
                     (dotimes (i 9)
                       (is (close-enough-p (aref data i)
                                           (cffi:mem-aref fa :double i)
                                           'double-float)))))
              (reset-matrix)
              (let* ((data (copy-seq original))
                     (view (trivial-simd/blas:make-matrix-view
                            data 3 3 :layout layout-symbol))
                     (actual (copy-seq y)))
                (cffi:foreign-funcall "cblas_dsymv" :int enum :int 121 :int 3
                                      :double 1d0 :pointer fa :int 3 :pointer fx
                                      :int 1 :double 1d0 :pointer fy :int 1 :void)
                (trivial-simd/blas:dsymv :upper 1d0 view x 1 1d0 actual 1)
                (dotimes (i 3)
                  (is (close-enough-p (aref actual i)
                                      (cffi:mem-aref fy :double i)
                                      'double-float))))
              (reset-matrix)
              (let* ((data (copy-seq original))
                     (view (trivial-simd/blas:make-matrix-view
                            data 3 3 :layout layout-symbol)))
                (cffi:foreign-funcall "cblas_dger" :int enum :int 3 :int 3
                                      :double 1d0 :pointer fx :int 1 :pointer fy
                                      :int 1 :pointer fa :int 3 :void)
                (trivial-simd/blas:dger 1d0 x 1 y 1 view)
                (compare-matrix data))
              (reset-matrix)
              (let* ((data (copy-seq original))
                     (view (trivial-simd/blas:make-matrix-view
                            data 3 3 :layout layout-symbol)))
                (cffi:foreign-funcall "cblas_dsyr" :int enum :int 121 :int 3
                                      :double 1d0 :pointer fx :int 1 :pointer fa
                                      :int 3 :void)
                (trivial-simd/blas:dsyr :upper 1d0 x 1 view)
                (compare-matrix data)))))))))

(test blas-level2-reference-hermitian
  (when (and (reference-blas-available-p)
             (ignore-errors (cffi:foreign-symbol-pointer "cblas_zhemv")))
    (let* ((data (blas-vector '(complex double-float)
                              '(#C(2d0 9d0) #C(1d0 2d0)
                                #C(0d0 0d0) #C(3d0 8d0))))
           (view (trivial-simd/blas:make-matrix-view data 2 2))
           (x (blas-vector '(complex double-float)
                           '(#C(1d0 1d0) #C(2d0 -1d0))))
           (y (blas-vector '(complex double-float)
                           '(#C(4d0 1d0) #C(5d0 -1d0)))))
      (cffi:with-foreign-objects ((fa :double 8) (fx :double 4) (fy :double 4)
                                  (alpha :double 2) (beta :double 2))
        (dotimes (i 4)
          (setf (cffi:mem-aref fa :double (* 2 i)) (realpart (aref data i))
                (cffi:mem-aref fa :double (1+ (* 2 i))) (imagpart (aref data i))))
        (dotimes (i 2)
          (setf (cffi:mem-aref fx :double (* 2 i)) (realpart (aref x i))
                (cffi:mem-aref fx :double (1+ (* 2 i))) (imagpart (aref x i))
                (cffi:mem-aref fy :double (* 2 i)) (realpart (aref y i))
                (cffi:mem-aref fy :double (1+ (* 2 i))) (imagpart (aref y i))))
        (setf (cffi:mem-aref alpha :double 0) 1d0
              (cffi:mem-aref alpha :double 1) 0d0
              (cffi:mem-aref beta :double 0) 0d0
              (cffi:mem-aref beta :double 1) 0d0)
        (cffi:foreign-funcall "cblas_zhemv" :int 101 :int 121 :int 2
                              :pointer alpha :pointer fa :int 2 :pointer fx :int 1
                              :pointer beta :pointer fy :int 1 :void)
        (trivial-simd/blas:zhemv :upper #C(1d0 0d0) view x 1
                                 #C(0d0 0d0) y 1)
        (dotimes (i 2)
          (is (close-enough-p (realpart (aref y i))
                              (cffi:mem-aref fy :double (* 2 i)) 'double-float))
          (is (close-enough-p (imagpart (aref y i))
                              (cffi:mem-aref fy :double (1+ (* 2 i)))
                              'double-float)))))))
