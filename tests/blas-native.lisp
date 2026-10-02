(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defparameter *native-blas-types* '(single-float double-float))

(defun native-blas-usable-p ()
  (and trivial-simd::*native-blas-available-p*
       (not (eq (simd:backend) :lisp))
       (eq trivial-simd::*native-array-access* :pointer)))

(defmacro with-blas-path ((path) &body body)
  "Run BODY with PATH set to :native (always) or :lisp (never)."
  `(let* ((trivial-simd/blas::*native-blas-threshold*
            (if (eq ,path :native) 1 most-positive-fixnum))
          (trivial-simd/blas::*native-blas-level2-threshold*
            trivial-simd/blas::*native-blas-threshold*))
     ,@body))

(defun padded-view (type rows cols layout &optional dominant)
  (let* ((minor (if (eq layout :row-major) cols rows))
         (major (if (eq layout :row-major) rows cols))
         (ld (+ minor 3)) (offset 5)
         (data (blas-filled-vector type (+ offset (* major ld) 2))))
    (let ((view (trivial-simd/blas:make-matrix-view data rows cols :layout layout
                                                                  :leading-dimension ld
                                                                  :offset offset)))
      (when dominant
        (dotimes (i (length data))
          (setf (aref data i) (/ (aref data i) (coerce (max rows cols) type))))
        (dotimes (i (min rows cols))
          (incf (aref data (+ offset (trivial-simd/blas::matrix-index view i i)))
                (coerce 4 type))))
      view)))

(defun copy-view (view)
  (trivial-simd/blas:make-matrix-view
   (copy-seq (trivial-simd/blas:matrix-view-data view))
   (trivial-simd/blas:matrix-view-rows view)
   (trivial-simd/blas:matrix-view-cols view)
   :layout (trivial-simd/blas:matrix-view-layout view)
   :leading-dimension (trivial-simd/blas:matrix-view-leading-dimension view)
   :offset (trivial-simd/blas:matrix-view-offset view)))

(defun views-close-p (a b type)
  (let ((tolerance (blas-tolerance type)))
    (every (lambda (x y) (<= (abs (- x y)) (* tolerance (max 1 (abs y)))))
           (trivial-simd/blas:matrix-view-data a)
           (trivial-simd/blas:matrix-view-data b))))

(defun view-elements-close-p (a b type)
  (let ((tolerance (blas-tolerance type)))
    (dotimes (i (trivial-simd/blas:matrix-view-rows a) t)
      (dotimes (j (trivial-simd/blas:matrix-view-cols a))
        (let ((x (trivial-simd/blas:matrix-ref a i j))
              (y (trivial-simd/blas:matrix-ref b i j)))
          (unless (<= (abs (- x y)) (* tolerance (max 1 (abs y))))
            (return-from view-elements-close-p nil)))))))

(defun operated-dimensions (transpose rows cols)
  (if (eq transpose :no-transpose) (values rows cols) (values cols rows)))

(test blas-native-gemm-matches-lisp
  (when (native-blas-usable-p)
    (dolist (type *native-blas-types*)
      (dolist (layout '(:row-major :column-major))
        (dolist (ta '(:no-transpose :transpose))
          (dolist (tb '(:no-transpose :transpose))
            (dolist (shape '((1 1 1) (7 5 6) (13 9 17) (130 70 300) (3 257 5)))
              (destructuring-bind (m n k) shape
                (multiple-value-bind (ar ac) (operated-dimensions ta m k)
                  (multiple-value-bind (br bc) (operated-dimensions tb k n)
                    (let* ((a (padded-view type ar ac layout))
                           (b (padded-view type br bc layout))
                           (c (padded-view type m n layout))
                           (expected (copy-view c))
                           (alpha (blas-scalar type 3/4))
                           (beta (blas-scalar type 1/2)))
                      (with-blas-path (:lisp)
                        (funcall (blas-routine type "gemm") ta tb alpha a b beta
                                 expected))
                      (with-blas-path (:native)
                        (funcall (blas-routine type "gemm") ta tb alpha a b beta c))
                      (is (views-close-p c expected type)
                          "gemm ~S ~S ~S ~S ~S" type layout ta tb shape))))))))))))

(test blas-native-symm-matches-lisp
  (when (native-blas-usable-p)
    (dolist (type *native-blas-types*)
      (dolist (side '(:left :right))
        (dolist (uplo '(:upper :lower))
          (let* ((m 40) (n 33) (order (if (eq side :left) m n))
                 (a (padded-view type order order :row-major))
                 (b (padded-view type m n :column-major))
                 (c (padded-view type m n :row-major))
                 (expected (copy-view c)))
            (with-blas-path (:lisp)
              (funcall (blas-routine type "symm") side uplo
                       (blas-scalar type 3/4) a b (blas-scalar type 1/2) expected))
            (with-blas-path (:native)
              (funcall (blas-routine type "symm") side uplo
                       (blas-scalar type 3/4) a b (blas-scalar type 1/2) c))
            (is (views-close-p c expected type) "symm ~S ~S ~S" type side uplo)))))))

(test blas-native-gemm-zero-scalars
  (when (native-blas-usable-p)
    (dolist (type *native-blas-types*)
      (let ((poison (blas-poison type)))
        (when poison
          (let ((a (padded-view type 20 20 :row-major))
                (b (padded-view type 20 20 :row-major))
                (c (padded-view type 20 20 :row-major))
                (zero (blas-scalar type 0)) (one (blas-scalar type 1)))
            (let ((before (copy-seq (trivial-simd/blas:matrix-view-data c))))
              (fill (trivial-simd/blas:matrix-view-data a) poison)
              (fill (trivial-simd/blas:matrix-view-data b) poison)
              (with-blas-path (:native)
                (funcall (blas-routine type "gemm") :no-transpose :no-transpose
                         zero a b one c))
              (is (equalp before (trivial-simd/blas:matrix-view-data c))
                  "native gemm with zero alpha must not read A or B"))
            (let ((a (padded-view type 20 20 :row-major))
                  (b (padded-view type 20 20 :row-major))
                  (expected (padded-view type 20 20 :row-major)))
              (fill (trivial-simd/blas:matrix-view-data c) poison)
              (with-blas-path (:native)
                (funcall (blas-routine type "gemm") :no-transpose :no-transpose
                         (blas-scalar type 3/4) a b zero c))
              (with-blas-path (:lisp)
                (funcall (blas-routine type "gemm") :no-transpose :no-transpose
                         (blas-scalar type 3/4) a b zero expected))
              (is (view-elements-close-p c expected type)
                  "native gemm with zero beta must not read C"))))))))

(test blas-native-falls-back-without-library-symbols
  (let ((trivial-simd::*native-blas-available-p* nil)
        (trivial-simd/blas::*native-blas-threshold* 1)
        (trivial-simd/blas::*native-blas-level2-threshold* 1))
    (dolist (type *native-blas-types*)
      (let* ((a (padded-view type 9 8 :row-major))
             (b (padded-view type 8 7 :row-major))
             (c (padded-view type 9 7 :row-major))
             (expected (copy-view c)))
        (funcall (blas-routine type "gemm") :no-transpose :no-transpose
                 (blas-scalar type 3/4) a b (blas-scalar type 1/2) c)
        (with-blas-path (:lisp)
          (funcall (blas-routine type "gemm") :no-transpose :no-transpose
                   (blas-scalar type 3/4) a b (blas-scalar type 1/2) expected))
        (is (equalp (trivial-simd/blas:matrix-view-data c)
                    (trivial-simd/blas:matrix-view-data expected)))))))

(test blas-native-rank-matches-lisp
  (when (native-blas-usable-p)
    (dolist (type *native-blas-types*)
      (dolist (family '("syrk" "syr2k"))
        (dolist (uplo '(:upper :lower))
          (dolist (transpose '(:no-transpose :transpose))
            (dolist (layout '(:row-major :column-major))
              (dolist (shape '((5 3) (40 33) (17 70) (130 260) (300 7)))
                (destructuring-bind (n k) shape
                  (multiple-value-bind (rows cols) (operated-dimensions transpose n k)
                    (let* ((a (padded-view type rows cols layout))
                           (b (padded-view type rows cols layout))
                           (c (padded-view type n n layout))
                           (expected (copy-view c))
                           (alpha (blas-scalar type 3/4)) (beta (blas-scalar type 1/2))
                           (arguments (if (string= family "syrk") (list a) (list a b))))
                      (with-blas-path (:lisp)
                        (apply (blas-routine type family) uplo transpose alpha
                               (append arguments (list beta expected))))
                      (with-blas-path (:native)
                        (apply (blas-routine type family) uplo transpose alpha
                               (append arguments (list beta c))))
                      (is (views-close-p c expected type)
                          "~A ~S ~S ~S ~S ~S" family type uplo transpose layout shape))))))))))))

(test blas-native-triangular-matches-lisp
  (when (native-blas-usable-p)
    (dolist (type *native-blas-types*)
      (dolist (family '("trmm" "trsm"))
        (dolist (side '(:left :right))
          (dolist (uplo '(:upper :lower))
            (dolist (transpose '(:no-transpose :transpose))
              (dolist (diag '(:unit :non-unit))
                (dolist (layout '(:row-major :column-major))
                  (dolist (shape '((5 3) (40 33) (19 50) (150 70) (70 131) (300 20) (20 300)))
                    (destructuring-bind (m n) shape
                      (let* ((order (if (eq side :left) m n))
                             (a (padded-view type order order layout t))
                             (b (padded-view type m n layout))
                             (expected (copy-view b))
                             (alpha (blas-scalar type 3/4)))
                        (with-blas-path (:lisp)
                          (funcall (blas-routine type family) side uplo transpose diag
                                   alpha a expected))
                        (with-blas-path (:native)
                          (funcall (blas-routine type family) side uplo transpose diag
                                   alpha a b))
                        (is (views-close-p b expected type)
                            "~A ~S ~S ~S ~S ~S ~S ~S" family type side uplo transpose
                            diag layout shape)))))))))))))

(defun padded-vector (type count increment)
  (blas-filled-vector type (+ 3 (* (max 0 (1- count)) (abs increment)))))

(defun vectors-close-p (a b type)
  (let ((tolerance (blas-tolerance type)))
    (every (lambda (x y) (<= (abs (- x y)) (* tolerance (max 1 (abs y))))) a b)))

(test blas-native-gemv-matches-lisp
  (when (native-blas-usable-p)
    (dolist (type *native-blas-types*)
      (dolist (layout '(:row-major :column-major))
        (dolist (transpose '(:no-transpose :transpose))
          (dolist (increments '((1 1) (2 -3) (-1 1)))
            (dolist (shape '((1 1) (7 5) (40 33) (130 70)))
              (destructuring-bind (rows cols) shape
                (destructuring-bind (incx incy) increments
                  (let* ((a (padded-view type rows cols layout))
                         (m (if (eq transpose :no-transpose) rows cols))
                         (n (if (eq transpose :no-transpose) cols rows))
                         (x (padded-vector type n incx))
                         (y (padded-vector type m incy))
                         (expected (copy-seq y)))
                    (with-blas-path (:lisp)
                      (funcall (blas-routine type "gemv") transpose
                               (blas-scalar type 3/4) a x incx
                               (blas-scalar type 1/2) expected incy))
                    (with-blas-path (:native)
                      (funcall (blas-routine type "gemv") transpose
                               (blas-scalar type 3/4) a x incx
                               (blas-scalar type 1/2) y incy))
                    (is (vectors-close-p y expected type)
                        "gemv ~S ~S ~S ~S ~S" type layout transpose increments shape)))))))))))

(test blas-native-ger-matches-lisp
  (when (native-blas-usable-p)
    (dolist (type *native-blas-types*)
      (dolist (layout '(:row-major :column-major))
        (dolist (increments '((1 1) (2 -3) (-1 2)))
          (dolist (shape '((1 1) (7 5) (40 33) (130 70)))
            (destructuring-bind (rows cols) shape
              (destructuring-bind (incx incy) increments
                (let* ((a (padded-view type rows cols layout))
                       (expected (copy-view a))
                       (x (padded-vector type rows incx))
                       (y (padded-vector type cols incy)))
                  (with-blas-path (:lisp)
                    (funcall (blas-routine type "ger") (blas-scalar type 3/4)
                             x incx y incy expected))
                  (with-blas-path (:native)
                    (funcall (blas-routine type "ger") (blas-scalar type 3/4)
                             x incx y incy a))
                  (is (views-close-p a expected type)
                      "ger ~S ~S ~S ~S" type layout increments shape))))))))))

(test blas-native-trsv-matches-lisp
  (when (native-blas-usable-p)
    (dolist (type *native-blas-types*)
      (dolist (layout '(:row-major :column-major))
        (dolist (uplo '(:upper :lower))
          (dolist (transpose '(:no-transpose :transpose))
            (dolist (diag '(:unit :non-unit))
              (dolist (incx '(1 -2))
                (dolist (n '(1 7 40 130))
                  (let* ((a (padded-view type n n layout t))
                         (x (padded-vector type n incx))
                         (expected (copy-seq x)))
                    (with-blas-path (:lisp)
                      (funcall (blas-routine type "trsv") uplo transpose diag a
                               expected incx))
                    (with-blas-path (:native)
                      (funcall (blas-routine type "trsv") uplo transpose diag a x incx))
                    (is (vectors-close-p x expected type)
                        "trsv ~S ~S ~S ~S ~S ~S ~S"
                        type layout uplo transpose diag incx n)))))))))))
