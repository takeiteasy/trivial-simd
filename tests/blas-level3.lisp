(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defun level3-number (type real &optional (imaginary 0))
  (if (consp type)
      (complex (coerce real (second type)) (coerce imaginary (second type)))
      (coerce real type)))

(defun level3-view (type values &key (layout :row-major) (rows 2) (cols 2))
  (trivial-simd/blas:make-matrix-view
   (blas-vector type (mapcar (lambda (value)
                              (if (complexp value)
                                  (level3-number type (realpart value) (imagpart value))
                                  (level3-number type value)))
                            values))
   rows cols :layout layout))

(defun level3-call (name &rest arguments)
  (apply (symbol-function (find-symbol (string-upcase name) :trivial-simd/blas))
         arguments))

(test blas-level3-all-variants
  (dolist (case '((single-float "s") (double-float "d")
                  ((complex single-float) "c")
                  ((complex double-float) "z")))
    (destructuring-bind (type prefix) case
      (let* ((one (level3-number type 1))
             (zero (level3-number type 0))
             (identity (list one zero zero one))
             (a (level3-view type identity))
             (b (level3-view type identity))
             (c (level3-view type (list zero zero zero zero))))
        (is (eq c (level3-call (concatenate 'string prefix "gemm")
                               :no-transpose :no-transpose one a b zero c)))
        (is (equalp (trivial-simd/blas:matrix-view-data c)
                    (trivial-simd/blas:matrix-view-data a)))
        (dolist (family '("symm" "syrk" "syr2k" "trmm" "trsm"))
          (let ((out (level3-view type identity)))
            (is (trivial-simd/blas:matrix-view-p
                 (cond ((string= family "symm")
                        (level3-call (concatenate 'string prefix family)
                                     :left :upper one a b zero out))
                       ((string= family "syrk")
                        (level3-call (concatenate 'string prefix family)
                                     :upper :no-transpose one a zero out))
                       ((string= family "syr2k")
                        (level3-call (concatenate 'string prefix family)
                                     :upper :no-transpose one a b zero out))
                       (t (level3-call (concatenate 'string prefix family)
                                       :left :upper :no-transpose :non-unit
                                       one a out)))))))
        (if (consp type)
            (dolist (family '("hemm" "herk" "her2k"))
              (let ((out (level3-view type identity))
                    (real-one (level3-number (second type) 1)))
                (is (trivial-simd/blas:matrix-view-p
                     (cond ((string= family "hemm")
                            (level3-call (concatenate 'string prefix family)
                                         :left :upper one a b zero out))
                           ((string= family "herk")
                            (level3-call (concatenate 'string prefix family)
                                         :upper :no-transpose real-one a
                                         (level3-number (second type) 0) out))
                           (t (level3-call (concatenate 'string prefix family)
                                           :upper :no-transpose one a b
                                           (level3-number (second type) 0)
                                           out)))))))
            (is (= 1 (trivial-simd/blas:matrix-ref c 0 0))))))))

(test blas-level3-gemm-views-and-validation
  (dolist (layout '(:row-major :column-major))
    (let* ((a (level3-view 'double-float '(1 2 3 4 5 6) :rows 2 :cols 3
                           :layout layout))
           (b (level3-view 'double-float '(7 8 9 10 11 12) :rows 3 :cols 2
                           :layout layout))
           (data (blas-vector 'double-float (make-list 12 :initial-element -9d0)))
           (parent (trivial-simd/blas:make-matrix-view
                    data 3 3 :layout layout :leading-dimension 4))
           (c (trivial-simd/blas:matrix-subview parent 1 1 2 2)))
      (is (eq c (trivial-simd/blas:dgemm :no-transpose :no-transpose
                                        1d0 a b 0d0 c)))
      (is (= (trivial-simd/blas:matrix-ref c 0 0)
             (if (eq layout :row-major) 58d0 76d0)))
      (is (= (trivial-simd/blas:matrix-ref c 1 1)
             (if (eq layout :row-major) 154d0 136d0)))
      (is (= (aref data 0) -9d0))
      (signals error (trivial-simd/blas:dgemm :bad :no-transpose 1d0 a b 0d0 c))
      (signals error (trivial-simd/blas:dgemm :no-transpose :no-transpose
                                              1d0 a a 0d0 c))))
  (let* ((storage (blas-vector 'double-float '(1d0 2d0 3d0 4d0 5d0 6d0)))
         (a (trivial-simd/blas:make-matrix-view storage 2 2))
         (c (trivial-simd/blas:make-matrix-view storage 2 2 :offset 1))
         (b (level3-view 'double-float '(1 0 0 1))))
    (signals error (trivial-simd/blas:dgemm :no-transpose :no-transpose
                                            1d0 a b 0d0 c))
    (is (= 2d0 (trivial-simd/blas:matrix-ref c 0 0))))
  (let* ((storage (blas-vector 'double-float '(1d0 0d0 0d0 1d0
                                                2d0 0d0 0d0 2d0)))
         (a (trivial-simd/blas:make-matrix-view storage 2 2))
         (c (trivial-simd/blas:make-matrix-view storage 2 2 :offset 4)))
    (trivial-simd/blas:dgemm :no-transpose :no-transpose 1d0 a a 0d0 c)
    (is (= 1d0 (trivial-simd/blas:matrix-ref c 0 0))))
  (let* ((a (level3-view 'double-float '(1 2 3 4)))
         (b (level3-view 'double-float '(5 6 7 8)))
         (c (level3-view 'double-float '(9 10 11 12))))
    (trivial-simd/blas:dgemm :no-transpose :no-transpose 0d0 a b 0d0 c)
    (is (equalp (trivial-simd/blas:matrix-view-data c)
                (blas-vector 'double-float '(0d0 0d0 0d0 0d0))))))

(test blas-level3-triangular-flags
  (dolist (side '(:left :right))
    (dolist (uplo '(:upper :lower))
      (dolist (transpose '(:no-transpose :transpose :conjugate-transpose))
        (let* ((a (level3-view '(complex double-float)
                               '(#C(2 1) #C(3 -1) #C(4 2) #C(5 -2))))
               (b (level3-view '(complex double-float)
                               '(#C(1 1) #C(2 -1) #C(3 2) #C(4 -2))))
               (original (copy-seq (trivial-simd/blas:matrix-view-data b))))
          (trivial-simd/blas:ztrmm side uplo transpose :non-unit
                                  #C(1d0 0d0) a b)
          (trivial-simd/blas:ztrsm side uplo transpose :non-unit
                                  #C(1d0 0d0) a b)
          (dotimes (i 4)
            (is (close-enough-p (aref original i)
                                (aref (trivial-simd/blas:matrix-view-data b) i)
                                '(complex double-float))))))))
  (let* ((a (level3-view 'double-float '(0 1 0 0)))
         (b (level3-view 'double-float '(1 2 3 4))))
    (trivial-simd/blas:dtrsm :left :upper :no-transpose :unit 1d0 a b)
    (is (equalp (trivial-simd/blas:matrix-view-data b)
                (blas-vector 'double-float '(-2d0 -2d0 3d0 4d0))))))

(test blas-level3-rank-and-shortcuts
  (let* ((a (level3-view '(complex double-float)
                         '(#C(1 1) #C(2 -1) #C(3 2) #C(4 1))))
         (b (level3-view '(complex double-float)
                         '(#C(2 1) #C(1 -1) #C(0 1) #C(3 2))))
         (c (level3-view '(complex double-float)
                         '(#C(1 9) #C(0 0) #C(7 8) #C(2 9)))))
    (trivial-simd/blas:zher2k :upper :no-transpose #C(1d0 1d0)
                              a b 0d0 c)
    (is (zerop (imagpart (trivial-simd/blas:matrix-ref c 0 0))))
    (is (= #C(7d0 8d0) (trivial-simd/blas:matrix-ref c 1 0)))
    (signals error (trivial-simd/blas:zherk :upper :transpose 1d0 a 0d0 c))
    (signals type-error (trivial-simd/blas:zherk :upper :no-transpose
                                                  #C(1d0 0d0) a 0d0 c))
    (trivial-simd/blas:zherk :lower :conjugate-transpose 0d0 a 0d0 c)
    (is (zerop (trivial-simd/blas:matrix-ref c 1 0)))))

(test blas-level3-convenience
  (let* ((a (level3-view 'double-float '(2 0 0 3)))
         (b (level3-view 'double-float '(1 2 3 4)))
         (c (level3-view 'double-float '(0 0 0 0))))
    (is (eq c (trivial-simd/blas/convenience:gemm! c 1d0 a b)))
    (is (= 12d0 (trivial-simd/blas:matrix-ref c 1 1)))
    (is (eq c (trivial-simd/blas/convenience:trsm! c 1d0 a)))
    (is (equalp (trivial-simd/blas:matrix-view-data c)
                (trivial-simd/blas:matrix-view-data b)))))

(test blas-level3-reference-real
  (when (reference-blas-available-p)
    (dolist (layout '((:row-major 101) (:column-major 102)))
      (destructuring-bind (layout-symbol enum) layout
        (dolist (family '("gemm" "symm" "syrk" "syr2k" "trmm" "trsm"))
          (let* ((a (level3-view 'double-float '(2 1 0 3)
                                 :layout layout-symbol))
                 (b (level3-view 'double-float '(4 5 6 7)
                                 :layout layout-symbol))
                 (c (level3-view 'double-float '(8 9 10 11)
                                 :layout layout-symbol))
                 (output (if (member family '("trmm" "trsm") :test #'string=)
                             b c))
                 (original (copy-seq (trivial-simd/blas:matrix-view-data output))))
            (cffi:with-foreign-objects ((fa :double 4) (fb :double 4)
                                        (fc :double 4))
              (dotimes (i 4)
                (setf (cffi:mem-aref fa :double i)
                      (aref (trivial-simd/blas:matrix-view-data a) i)
                      (cffi:mem-aref fb :double i)
                      (aref (trivial-simd/blas:matrix-view-data b) i)
                      (cffi:mem-aref fc :double i)
                      (aref (trivial-simd/blas:matrix-view-data c) i)))
              (cond
                ((string= family "gemm")
                 (cffi:foreign-funcall "cblas_dgemm" :int enum :int 111 :int 111
                                       :int 2 :int 2 :int 2 :double 1d0
                                       :pointer fa :int 2 :pointer fb :int 2
                                       :double 0.5d0 :pointer fc :int 2 :void)
                 (trivial-simd/blas:dgemm :no-transpose :no-transpose
                                         1d0 a b 0.5d0 c))
                ((string= family "symm")
                 (cffi:foreign-funcall "cblas_dsymm" :int enum :int 141 :int 121
                                       :int 2 :int 2 :double 1d0 :pointer fa
                                       :int 2 :pointer fb :int 2 :double 0.5d0
                                       :pointer fc :int 2 :void)
                 (trivial-simd/blas:dsymm :left :upper 1d0 a b 0.5d0 c))
                ((string= family "syrk")
                 (cffi:foreign-funcall "cblas_dsyrk" :int enum :int 121 :int 111
                                       :int 2 :int 2 :double 1d0 :pointer fa
                                       :int 2 :double 0.5d0 :pointer fc :int 2 :void)
                 (trivial-simd/blas:dsyrk :upper :no-transpose 1d0 a 0.5d0 c))
                ((string= family "syr2k")
                 (cffi:foreign-funcall "cblas_dsyr2k" :int enum :int 121 :int 111
                                       :int 2 :int 2 :double 1d0 :pointer fa :int 2
                                       :pointer fb :int 2 :double 0.5d0 :pointer fc
                                       :int 2 :void)
                 (trivial-simd/blas:dsyr2k :upper :no-transpose
                                          1d0 a b 0.5d0 c))
                ((string= family "trmm")
                 (cffi:foreign-funcall "cblas_dtrmm" :int enum :int 141 :int 121
                                       :int 111 :int 131 :int 2 :int 2 :double 1d0
                                       :pointer fa :int 2 :pointer fb :int 2 :void)
                 (trivial-simd/blas:dtrmm :left :upper :no-transpose
                                         :non-unit 1d0 a b))
                (t
                 (cffi:foreign-funcall "cblas_dtrsm" :int enum :int 141 :int 121
                                       :int 111 :int 131 :int 2 :int 2 :double 1d0
                                       :pointer fa :int 2 :pointer fb :int 2 :void)
                 (trivial-simd/blas:dtrsm :left :upper :no-transpose
                                         :non-unit 1d0 a b)))
              (dotimes (i 4)
                (is (close-enough-p
                     (aref (trivial-simd/blas:matrix-view-data output) i)
                     (cffi:mem-aref (if (member family '("trmm" "trsm")
                                        :test #'string=) fb fc)
                                    :double i)
                     'double-float))
                (when (and (member family '("syrk" "syr2k") :test #'string=)
                           (= i (if (eq layout-symbol :row-major) 2 1)))
                  (is (= (aref original i)
                         (aref (trivial-simd/blas:matrix-view-data output) i))))))))))))

(defun level3-store-complex (pointer values)
  (dotimes (i (length values))
    (setf (cffi:mem-aref pointer :double (* 2 i))
          (realpart (aref values i))
          (cffi:mem-aref pointer :double (1+ (* 2 i)))
          (imagpart (aref values i)))))

(test blas-level3-reference-single
  (when (and (reference-blas-available-p)
             (ignore-errors (cffi:foreign-symbol-pointer "cblas_cgemm")))
    (let* ((a (level3-view 'single-float '(1 2 3 4)))
           (b (level3-view 'single-float '(5 6 7 8)))
           (c (level3-view 'single-float '(0 0 0 0)))
           (ca (level3-view '(complex single-float)
                            '(#C(1 1) #C(2 -1) #C(3 2) #C(4 0))))
           (cb (level3-view '(complex single-float)
                            '(#C(2 1) #C(1 -2) #C(0 1) #C(3 2))))
           (cc (level3-view '(complex single-float)
                            '(#C(0 0) #C(0 0) #C(0 0) #C(0 0)))))
      (cffi:with-foreign-objects ((fa :float 4) (fb :float 4) (fc :float 4)
                                  (fca :float 8) (fcb :float 8) (fcc :float 8)
                                  (alpha :float 2) (beta :float 2))
        (dotimes (i 4)
          (setf (cffi:mem-aref fa :float i)
                (aref (trivial-simd/blas:matrix-view-data a) i)
                (cffi:mem-aref fb :float i)
                (aref (trivial-simd/blas:matrix-view-data b) i)
                (cffi:mem-aref fc :float i) 0.0
                (cffi:mem-aref fca :float (* 2 i))
                (realpart (aref (trivial-simd/blas:matrix-view-data ca) i))
                (cffi:mem-aref fca :float (1+ (* 2 i)))
                (imagpart (aref (trivial-simd/blas:matrix-view-data ca) i))
                (cffi:mem-aref fcb :float (* 2 i))
                (realpart (aref (trivial-simd/blas:matrix-view-data cb) i))
                (cffi:mem-aref fcb :float (1+ (* 2 i)))
                (imagpart (aref (trivial-simd/blas:matrix-view-data cb) i))
                (cffi:mem-aref fcc :float (* 2 i)) 0.0
                (cffi:mem-aref fcc :float (1+ (* 2 i))) 0.0))
        (setf (cffi:mem-aref alpha :float 0) 1.0
              (cffi:mem-aref alpha :float 1) 0.0
              (cffi:mem-aref beta :float 0) 0.0
              (cffi:mem-aref beta :float 1) 0.0)
        (cffi:foreign-funcall "cblas_sgemm" :int 101 :int 111 :int 111
                              :int 2 :int 2 :int 2 :float 1.0 :pointer fa
                              :int 2 :pointer fb :int 2 :float 0.0
                              :pointer fc :int 2 :void)
        (cffi:foreign-funcall "cblas_cgemm" :int 101 :int 111 :int 113
                              :int 2 :int 2 :int 2 :pointer alpha :pointer fca
                              :int 2 :pointer fcb :int 2 :pointer beta
                              :pointer fcc :int 2 :void)
        (trivial-simd/blas:sgemm :no-transpose :no-transpose 1.0 a b 0.0 c)
        (trivial-simd/blas:cgemm :no-transpose :conjugate-transpose
                                #C(1.0 0.0) ca cb #C(0.0 0.0) cc)
        (dotimes (i 4)
          (is (close-enough-p (aref (trivial-simd/blas:matrix-view-data c) i)
                              (cffi:mem-aref fc :float i) 'single-float))
          (let ((actual (aref (trivial-simd/blas:matrix-view-data cc) i)))
            (is (close-enough-p (realpart actual)
                                (cffi:mem-aref fcc :float (* 2 i)) 'single-float))
            (is (close-enough-p (imagpart actual)
                                (cffi:mem-aref fcc :float (1+ (* 2 i)))
                                'single-float))))))))

(test blas-level3-reference-complex
  (when (and (reference-blas-available-p)
             (ignore-errors (cffi:foreign-symbol-pointer "cblas_zher2k")))
    (dolist (layout '((:row-major 101) (:column-major 102)))
      (destructuring-bind (layout-symbol enum) layout
        (dolist (family '("gemm" "hemm" "herk" "her2k"))
          (let* ((a (level3-view '(complex double-float)
                                 '(#C(2 9) #C(1 2) #C(0 0) #C(3 8))
                                 :layout layout-symbol))
                 (b (level3-view '(complex double-float)
                                 '(#C(1 1) #C(2 -1) #C(3 2) #C(4 -2))
                                 :layout layout-symbol))
                 (c (level3-view '(complex double-float)
                                 '(#C(5 9) #C(6 1) #C(7 2) #C(8 9))
                                 :layout layout-symbol)))
            (cffi:with-foreign-objects ((fa :double 8) (fb :double 8)
                                        (fc :double 8) (alpha :double 2)
                                        (beta :double 2))
              (level3-store-complex fa (trivial-simd/blas:matrix-view-data a))
              (level3-store-complex fb (trivial-simd/blas:matrix-view-data b))
              (level3-store-complex fc (trivial-simd/blas:matrix-view-data c))
              (setf (cffi:mem-aref alpha :double 0) 1d0
                    (cffi:mem-aref alpha :double 1) 1d0
                    (cffi:mem-aref beta :double 0) 0.5d0
                    (cffi:mem-aref beta :double 1) 0d0)
              (cond
                ((string= family "gemm")
                 (cffi:foreign-funcall "cblas_zgemm" :int enum :int 113 :int 111
                                       :int 2 :int 2 :int 2 :pointer alpha
                                       :pointer fa :int 2 :pointer fb :int 2
                                       :pointer beta :pointer fc :int 2 :void)
                 (trivial-simd/blas:zgemm :conjugate-transpose :no-transpose
                                         #C(1d0 1d0) a b #C(0.5d0 0d0) c))
                ((string= family "hemm")
                 (cffi:foreign-funcall "cblas_zhemm" :int enum :int 141 :int 121
                                       :int 2 :int 2 :pointer alpha :pointer fa
                                       :int 2 :pointer fb :int 2 :pointer beta
                                       :pointer fc :int 2 :void)
                 (trivial-simd/blas:zhemm :left :upper #C(1d0 1d0)
                                         a b #C(0.5d0 0d0) c))
                ((string= family "herk")
                 (cffi:foreign-funcall "cblas_zherk" :int enum :int 121 :int 111
                                       :int 2 :int 2 :double 1d0 :pointer fa
                                       :int 2 :double 0.5d0 :pointer fc :int 2 :void)
                 (trivial-simd/blas:zherk :upper :no-transpose 1d0 a 0.5d0 c))
                (t
                 (cffi:foreign-funcall "cblas_zher2k" :int enum :int 121 :int 111
                                       :int 2 :int 2 :pointer alpha :pointer fa
                                       :int 2 :pointer fb :int 2 :double 0.5d0
                                       :pointer fc :int 2 :void)
                 (trivial-simd/blas:zher2k :upper :no-transpose
                                          #C(1d0 1d0) a b 0.5d0 c)))
              (dotimes (i 4)
                (let ((actual (aref (trivial-simd/blas:matrix-view-data c) i)))
                  (is (close-enough-p (realpart actual)
                                      (cffi:mem-aref fc :double (* 2 i))
                                      'double-float))
                  (is (close-enough-p (imagpart actual)
                                      (cffi:mem-aref fc :double (1+ (* 2 i)))
                                      'double-float)))))))))))
