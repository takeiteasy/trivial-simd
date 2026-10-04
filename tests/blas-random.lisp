(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defparameter *blas-random-types*
  '(single-float double-float (complex single-float) (complex double-float)))

(defparameter *blas-transposes* '(:no-transpose :transpose :conjugate-transpose))

(defvar *blas-random-seed* 12345)
(defvar *blas-bases* nil)
(defvar *blas-allocations* nil)

(defun blas-random (limit)
  (setf *blas-random-seed*
        (mod (+ (* *blas-random-seed* 1103515245) 12345) 2147483648))
  (mod (ash *blas-random-seed* -8) limit))

(defun blas-component (type) (if (consp type) (second type) type))
(defun blas-width (type) (if (consp type) 2 1))
(defun blas-ftype (type)
  (if (eq (blas-component type) 'single-float) :float :double))

(defun blas-prefix (type)
  (cond ((eq type 'single-float) "s")
        ((eq type 'double-float) "d")
        ((equal type '(complex single-float)) "c")
        (t "z")))

(defun blas-enum (keyword)
  (ecase keyword
    (:row-major 101) (:column-major 102)
    (:no-transpose 111) (:transpose 112) (:conjugate-transpose 113)
    (:upper 121) (:lower 122) (:non-unit 131) (:unit 132)
    (:left 141) (:right 142)))

(defparameter *blas-random-components*
  (loop for type in '(single-float double-float)
        collect (cons type (make-array 17 :element-type type
                                       :initial-contents
                                       (loop for i from -8 to 8 collect (coerce (/ i 8) type))))))

(defun blas-random-element (type)
  (let* ((components (cdr (assoc (blas-component type) *blas-random-components*)))
         (real (aref components (blas-random 17))))
    (if (consp type)
        (complex real (aref components (blas-random 17)))
        real)))

(defun blas-tolerance (type)
  (if (eq (blas-component type) 'single-float) 1.0e-4 1.0d-10))

(defmacro with-blas-cleanup (&body body)
  `(let ((*blas-bases* nil) (*blas-allocations* nil))
     (unwind-protect (progn ,@body)
       (mapc #'cffi:foreign-free *blas-allocations*))))

(defun foreign-from-lisp (data type)
  (let* ((width (blas-width type)) (ftype (blas-ftype type))
         (pointer (cffi:foreign-alloc ftype :count (* width (length data)))))
    (push pointer *blas-allocations*)
    (dotimes (i (length data) pointer)
      (let ((value (aref data i)))
        (if (= width 1)
            (setf (cffi:mem-aref pointer ftype i) value)
            (setf (cffi:mem-aref pointer ftype (* 2 i)) (realpart value)
                  (cffi:mem-aref pointer ftype (1+ (* 2 i))) (imagpart value)))))))

(defun element-close-p (actual expected type)
  (<= (abs (- actual expected))
      (* (blas-tolerance type) (max 1 (abs expected)))))

(defun foreign-matches-p (pointer data type)
  (let ((width (blas-width type)) (ftype (blas-ftype type)))
    (dotimes (i (length data) t)
      (let ((expected (if (= width 1)
                          (cffi:mem-aref pointer ftype i)
                          (complex (cffi:mem-aref pointer ftype (* 2 i))
                                   (cffi:mem-aref pointer ftype (1+ (* 2 i)))))))
        (unless (element-close-p (aref data i) expected type)
          (return nil))))))

(defun blas-filled-vector (type count)
  (let ((data (make-array count :element-type type)))
    (dotimes (i count data) (setf (aref data i) (blas-random-element type)))))

(defun new-matrix (type rows cols layout &optional dominant)
  "Return a padded, offset view and the CBLAS pointer to its first element."
  (let* ((minor (if (eq layout :row-major) cols rows))
         (major (if (eq layout :row-major) rows cols))
         (ld (+ minor 2)) (offset 3)
         (data (blas-filled-vector type (+ offset (* major ld) 2)))
         (view (trivial-simd/blas:make-matrix-view
                data rows cols :layout layout :leading-dimension ld
                :offset offset)))
    (when dominant
      (dotimes (i (min rows cols))
        (incf (aref data (+ offset (trivial-simd/blas::matrix-index view i i)))
              (coerce 4 type))))
    (let ((base (foreign-from-lisp data type)))
      (push (cons view base) *blas-bases*)
      (values view (cffi:inc-pointer base (* offset (blas-width type)
                                             (cffi:foreign-type-size
                                              (blas-ftype type))))))))

(defun new-vector (type count increment)
  (let* ((data (blas-filled-vector type (+ 2 (* (max 0 (1- count)) (abs increment)) 1)))
         (base (foreign-from-lisp data type)))
    (push (cons data base) *blas-bases*)
    (values data base)))

(defun result-matches-p (object type)
  (let ((base (cdr (assoc object *blas-bases*))))
    (foreign-matches-p base (if (typep object 'trivial-simd/blas:matrix-view)
                                (trivial-simd/blas:matrix-view-data object)
                                object)
                       type)))

(defvar *cblas-callers* (make-hash-table :test #'equal))

(defun cblas-caller (types)
  (or (gethash types *cblas-callers*)
      (setf (gethash types *cblas-callers*)
            (let ((names (loop for nil in types collect (gensym))))
              (compile nil `(lambda (function ,@names)
                              (cffi:foreign-funcall-pointer
                               function ()
                               ,@(loop for type in types for name in names
                                       append (list type name))
                               :void)))))))

(defun cblas-call (name &rest arguments)
  (without-float-traps
    (apply (cblas-caller (mapcar #'first arguments))
           (cffi:foreign-symbol-pointer name) (mapcar #'second arguments))))

(defun cblas-available-p (type family)
  (and (reference-blas-available-p)
       (cffi:foreign-symbol-pointer
        (format nil "cblas_~A~A" (blas-prefix type) family))))

(defun arg-int (value) (list :int value))
(defun arg-enum (keyword) (list :int (blas-enum keyword)))
(defun arg-pointer (pointer) (list :pointer pointer))

(defun arg-scalar (type value)
  "Pass VALUE of TYPE by value (real) or by pointer (complex)."
  (if (consp type)
      (arg-pointer (foreign-from-lisp (make-array 1 :element-type type
                                                    :initial-element value)
                                      type))
      (list (blas-ftype type) value)))

(defun arg-real (type value)
  (list (blas-ftype type) (coerce value (blas-component type))))

(defun blas-scalar (type real &optional (imaginary 0))
  (if (consp type)
      (complex (coerce real (second type)) (coerce imaginary (second type)))
      (coerce real type)))

(defun blas-routine (type family)
  (symbol-function
   (find-symbol (string-upcase (format nil "~A~A" (blas-prefix type) family))
                :trivial-simd/blas)))

(defmacro check-case (description matches)
  `(is ,matches "Mismatch against CBLAS: ~S" ,description))

;;; Level 3

(defun gemm-operated-shape (transpose rows cols)
  (if (eq transpose :no-transpose) (values rows cols) (values cols rows)))

(test blas-random-gemm
  (dolist (type *blas-random-types*)
    (when (cblas-available-p type "gemm")
      (dolist (layout '(:row-major :column-major))
        (dolist (ta *blas-transposes*)
          (dolist (tb *blas-transposes*)
            (with-blas-cleanup
              (let ((m 7) (n 5) (k 6)
                    (alpha (blas-scalar type 3/4 -1/4))
                    (beta (blas-scalar type 1/2 1/4)))
                (multiple-value-bind (a-rows a-cols) (gemm-operated-shape ta m k)
                  (multiple-value-bind (b-rows b-cols) (gemm-operated-shape tb k n)
                    (multiple-value-bind (a ap) (new-matrix type a-rows a-cols layout)
                      (multiple-value-bind (b bp) (new-matrix type b-rows b-cols layout)
                        (multiple-value-bind (c cp) (new-matrix type m n layout)
                          (cblas-call (format nil "cblas_~Agemm" (blas-prefix type))
                                      (arg-enum layout) (arg-enum ta) (arg-enum tb)
                                      (arg-int m) (arg-int n) (arg-int k)
                                      (arg-scalar type alpha)
                                      (arg-pointer ap)
                                      (arg-int (trivial-simd/blas:matrix-view-leading-dimension a))
                                      (arg-pointer bp)
                                      (arg-int (trivial-simd/blas:matrix-view-leading-dimension b))
                                      (arg-scalar type beta)
                                      (arg-pointer cp)
                                      (arg-int (trivial-simd/blas:matrix-view-leading-dimension c)))
                          (funcall (blas-routine type "gemm") ta tb alpha a b beta c)
                          (check-case (list type layout ta tb)
                                      (result-matches-p c type)))))))))))))))

(test blas-random-symm-hemm
  (dolist (type *blas-random-types*)
    (dolist (family (if (consp type) '("symm" "hemm") '("symm")))
      (when (cblas-available-p type family)
        (dolist (layout '(:row-major :column-major))
          (dolist (side '(:left :right))
            (dolist (uplo '(:upper :lower))
              (with-blas-cleanup
                (let* ((m 7) (n 5) (order (if (eq side :left) m n))
                       (alpha (blas-scalar type 3/4 -1/4))
                       (beta (blas-scalar type 1/2 1/4)))
                  (multiple-value-bind (a ap) (new-matrix type order order layout t)
                    (multiple-value-bind (b bp) (new-matrix type m n layout)
                      (multiple-value-bind (c cp) (new-matrix type m n layout)
                        (cblas-call (format nil "cblas_~A~A" (blas-prefix type) family)
                                    (arg-enum layout) (arg-enum side) (arg-enum uplo)
                                    (arg-int m) (arg-int n) (arg-scalar type alpha)
                                    (arg-pointer ap)
                                    (arg-int (trivial-simd/blas:matrix-view-leading-dimension a))
                                    (arg-pointer bp)
                                    (arg-int (trivial-simd/blas:matrix-view-leading-dimension b))
                                    (arg-scalar type beta) (arg-pointer cp)
                                    (arg-int (trivial-simd/blas:matrix-view-leading-dimension c)))
                        (funcall (blas-routine type family) side uplo alpha a b beta c)
                        (check-case (list type family layout side uplo)
                                    (result-matches-p c type))))))))))))))

(defun rank-transposes (family type)
  (cond ((member family '("herk" "her2k") :test #'string=)
         '(:no-transpose :conjugate-transpose))
        ((consp type) '(:no-transpose :transpose))
        (t *blas-transposes*)))

(test blas-random-rank-updates
  (dolist (type *blas-random-types*)
    (dolist (family (if (consp type)
                        '("syrk" "syr2k" "herk" "her2k")
                        '("syrk" "syr2k")))
      (when (cblas-available-p type family)
        (let ((hermitian (char= (char family 0) #\h))
              (second-input (search "2k" family)))
          (dolist (layout '(:row-major :column-major))
            (dolist (uplo '(:upper :lower))
              (dolist (transpose (rank-transposes family type))
                (with-blas-cleanup
                  (let* ((n 7) (k 5)
                         (alpha (if (and hermitian (not second-input))
                                    (coerce 3/4 (second type))
                                    (blas-scalar type 3/4 -1/4)))
                         (beta (if hermitian
                                   (coerce 1/2 (second type))
                                   (blas-scalar type 1/2 1/4)))
                         (rows (if (eq transpose :no-transpose) n k))
                         (cols (if (eq transpose :no-transpose) k n)))
                    (multiple-value-bind (a ap) (new-matrix type rows cols layout)
                      (multiple-value-bind (b bp) (new-matrix type rows cols layout)
                        (multiple-value-bind (c cp) (new-matrix type n n layout)
                          (apply #'cblas-call
                                 (format nil "cblas_~A~A" (blas-prefix type) family)
                                 (append
                                  (list (arg-enum layout) (arg-enum uplo)
                                        (arg-enum transpose) (arg-int n) (arg-int k)
                                        (if (and hermitian (not second-input))
                                            (arg-real type alpha)
                                            (arg-scalar type alpha))
                                        (arg-pointer ap)
                                        (arg-int (trivial-simd/blas:matrix-view-leading-dimension a)))
                                  (when second-input
                                    (list (arg-pointer bp)
                                          (arg-int (trivial-simd/blas:matrix-view-leading-dimension b))))
                                  (list (if hermitian (arg-real type beta)
                                            (arg-scalar type beta))
                                        (arg-pointer cp)
                                        (arg-int (trivial-simd/blas:matrix-view-leading-dimension c)))))
                          (if second-input
                              (funcall (blas-routine type family) uplo transpose
                                       alpha a b beta c)
                              (funcall (blas-routine type family) uplo transpose
                                       alpha a beta c))
                          (check-case (list type family layout uplo transpose)
                                      (result-matches-p c type)))))))))))))))

(test blas-random-triangular
  (dolist (type *blas-random-types*)
    (dolist (family '("trmm" "trsm"))
      (when (cblas-available-p type family)
        (dolist (layout '(:row-major :column-major))
          (dolist (side '(:left :right))
            (dolist (uplo '(:upper :lower))
              (dolist (transpose *blas-transposes*)
                (dolist (diag '(:non-unit :unit))
                  (with-blas-cleanup
                    (let* ((m 7) (n 5) (order (if (eq side :left) m n))
                           (alpha (blas-scalar type 3/4 -1/4)))
                      (multiple-value-bind (a ap) (new-matrix type order order layout t)
                        (multiple-value-bind (b bp) (new-matrix type m n layout)
                          (cblas-call (format nil "cblas_~A~A" (blas-prefix type) family)
                                      (arg-enum layout) (arg-enum side) (arg-enum uplo)
                                      (arg-enum transpose) (arg-enum diag)
                                      (arg-int m) (arg-int n) (arg-scalar type alpha)
                                      (arg-pointer ap)
                                      (arg-int (trivial-simd/blas:matrix-view-leading-dimension a))
                                      (arg-pointer bp)
                                      (arg-int (trivial-simd/blas:matrix-view-leading-dimension b)))
                          (funcall (blas-routine type family) side uplo transpose
                                   diag alpha a b)
                          (check-case (list type family layout side uplo transpose diag)
                                      (result-matches-p b type)))))))))))))))

;;; Level 2

(test blas-random-gemv
  (dolist (type *blas-random-types*)
    (when (cblas-available-p type "gemv")
      (dolist (layout '(:row-major :column-major))
        (dolist (transpose *blas-transposes*)
          (dolist (increments '((1 1) (2 -1) (-1 3) (-3 -2)))
            (destructuring-bind (incx incy) increments
              (with-blas-cleanup
                (let* ((m 9) (n 7)
                       (x-count (if (eq transpose :no-transpose) n m))
                       (y-count (if (eq transpose :no-transpose) m n))
                       (alpha (blas-scalar type 3/4 -1/4))
                       (beta (blas-scalar type 1/2 1/4)))
                  (multiple-value-bind (a ap) (new-matrix type m n layout)
                    (multiple-value-bind (x xb) (new-vector type x-count incx)
                      (multiple-value-bind (y yb) (new-vector type y-count incy)
                        (cblas-call (format nil "cblas_~Agemv" (blas-prefix type))
                                    (arg-enum layout) (arg-enum transpose)
                                    (arg-int m) (arg-int n) (arg-scalar type alpha)
                                    (arg-pointer ap)
                                    (arg-int (trivial-simd/blas:matrix-view-leading-dimension a))
                                    (arg-pointer xb) (arg-int incx)
                                    (arg-scalar type beta)
                                    (arg-pointer yb) (arg-int incy))
                        (funcall (blas-routine type "gemv") transpose alpha a x incx
                                 beta y incy)
                        (check-case (list type layout transpose incx incy)
                                    (result-matches-p y type))))))))))))))

(test blas-random-triangular-vector
  (dolist (type *blas-random-types*)
    (dolist (family '("trmv" "trsv"))
      (when (cblas-available-p type family)
        (dolist (layout '(:row-major :column-major))
          (dolist (uplo '(:upper :lower))
            (dolist (transpose *blas-transposes*)
              (dolist (diag '(:non-unit :unit))
                (dolist (incx '(1 -2))
                  (with-blas-cleanup
                    (let ((n 9))
                      (multiple-value-bind (a ap) (new-matrix type n n layout t)
                        (multiple-value-bind (x xb) (new-vector type n incx)
                          (cblas-call (format nil "cblas_~A~A" (blas-prefix type) family)
                                      (arg-enum layout) (arg-enum uplo)
                                      (arg-enum transpose) (arg-enum diag) (arg-int n)
                                      (arg-pointer ap)
                                      (arg-int (trivial-simd/blas:matrix-view-leading-dimension a))
                                      (arg-pointer xb) (arg-int incx))
                          (funcall (blas-routine type family) uplo transpose diag
                                   a x incx)
                          (check-case (list type family layout uplo transpose diag incx)
                                      (result-matches-p x type)))))))))))))))

(test blas-random-ger
  (dolist (type *blas-random-types*)
    (dolist (family (if (consp type) '("geru" "gerc") '("ger")))
      (when (cblas-available-p type family)
        (dolist (layout '(:row-major :column-major))
          (dolist (increments '((1 1) (-2 3) (3 -1)))
            (destructuring-bind (incx incy) increments
              (with-blas-cleanup
                (let ((m 9) (n 7) (alpha (blas-scalar type 3/4 -1/4)))
                  (multiple-value-bind (a ap) (new-matrix type m n layout)
                    (multiple-value-bind (x xb) (new-vector type m incx)
                      (multiple-value-bind (y yb) (new-vector type n incy)
                        (cblas-call (format nil "cblas_~A~A" (blas-prefix type) family)
                                    (arg-enum layout) (arg-int m) (arg-int n)
                                    (arg-scalar type alpha)
                                    (arg-pointer xb) (arg-int incx)
                                    (arg-pointer yb) (arg-int incy)
                                    (arg-pointer ap)
                                    (arg-int (trivial-simd/blas:matrix-view-leading-dimension a)))
                        (funcall (blas-routine type family) alpha x incx y incy a)
                        (check-case (list type family layout incx incy)
                                    (result-matches-p a type))))))))))))))

(test blas-random-symv-hemv
  (dolist (type *blas-random-types*)
    (let ((family (if (consp type) "hemv" "symv")))
      (when (cblas-available-p type family)
        (dolist (layout '(:row-major :column-major))
          (dolist (uplo '(:upper :lower))
            (dolist (increments '((1 1) (-2 3)))
              (destructuring-bind (incx incy) increments
                (with-blas-cleanup
                  (let ((n 9) (alpha (blas-scalar type 3/4 -1/4))
                        (beta (blas-scalar type 1/2 1/4)))
                    (multiple-value-bind (a ap) (new-matrix type n n layout)
                      (multiple-value-bind (x xb) (new-vector type n incx)
                        (multiple-value-bind (y yb) (new-vector type n incy)
                          (cblas-call (format nil "cblas_~A~A" (blas-prefix type) family)
                                      (arg-enum layout) (arg-enum uplo) (arg-int n)
                                      (arg-scalar type alpha) (arg-pointer ap)
                                      (arg-int (trivial-simd/blas:matrix-view-leading-dimension a))
                                      (arg-pointer xb) (arg-int incx)
                                      (arg-scalar type beta)
                                      (arg-pointer yb) (arg-int incy))
                          (funcall (blas-routine type family) uplo alpha a x incx
                                   beta y incy)
                          (check-case (list type family layout uplo incx incy)
                                      (result-matches-p y type)))))))))))))))

(test blas-random-syr-her
  (dolist (type *blas-random-types*)
    (let ((family (if (consp type) "her" "syr")))
      (when (cblas-available-p type family)
        (dolist (layout '(:row-major :column-major))
          (dolist (uplo '(:upper :lower))
            (dolist (incx '(1 -2))
              (with-blas-cleanup
                (let ((n 9) (alpha (coerce 3/4 (blas-component type))))
                  (multiple-value-bind (a ap) (new-matrix type n n layout)
                    (multiple-value-bind (x xb) (new-vector type n incx)
                      (cblas-call (format nil "cblas_~A~A" (blas-prefix type) family)
                                  (arg-enum layout) (arg-enum uplo) (arg-int n)
                                  (arg-real type alpha) (arg-pointer xb) (arg-int incx)
                                  (arg-pointer ap)
                                  (arg-int (trivial-simd/blas:matrix-view-leading-dimension a)))
                      (funcall (blas-routine type family) uplo alpha x incx a)
                      (check-case (list type family layout uplo incx)
                                  (result-matches-p a type)))))))))))))

;;; Band and packed Level 2

(defun view-ld (view) (trivial-simd/blas:matrix-view-leading-dimension view))

(defun band-width (kind kl ku)
  (if (eq kind :general) (+ kl ku 1) (1+ (max kl ku))))

(defun make-random-band (type rows cols layout kind kl ku uplo dominant)
  (let* ((ld (1+ (band-width kind kl ku))) (offset 2)
         (major (if (eq layout :row-major) rows cols))
         (data (blas-filled-vector type (+ offset (* major ld) 2)))
         (view (trivial-simd/blas:make-band-matrix-view
                data rows cols :kind kind :kl kl :ku ku :layout layout
                :leading-dimension ld :offset offset)))
    (when dominant
      (dotimes (i (min rows cols))
        (incf (aref data (+ offset (trivial-simd/blas::stored-index view i i uplo)))
              (coerce 4 type))))
    view))

(defun new-band (type rows cols layout kind kl ku &key uplo dominant)
  (let* ((view (make-random-band type rows cols layout kind kl ku uplo dominant))
         (base (foreign-from-lisp (trivial-simd/blas:matrix-view-data view) type)))
    (push (cons view base) *blas-bases*)
    (values view (cffi:inc-pointer base (* 2 (blas-width type)
                                           (cffi:foreign-type-size (blas-ftype type)))))))

(defun new-packed (type n layout &key uplo dominant)
  (let* ((offset 3)
         (data (blas-filled-vector type (+ offset (floor (* n (1+ n)) 2) 2)))
         (view (trivial-simd/blas:make-packed-matrix-view
                data n :layout layout :offset offset)))
    (when dominant
      (dotimes (i n)
        (incf (aref data (+ offset (trivial-simd/blas::stored-index view i i uplo)))
              (coerce 4 type))))
    (let ((base (foreign-from-lisp data type)))
      (push (cons view base) *blas-bases*)
      (values view (cffi:inc-pointer base (* offset (blas-width type)
                                             (cffi:foreign-type-size
                                              (blas-ftype type))))))))

(defun cblas-name (type family)
  (format nil "cblas_~A~A" (blas-prefix type) family))

(test blas-random-gbmv
  (dolist (type *blas-random-types*)
    (when (cblas-available-p type "gbmv")
      (dolist (layout '(:row-major :column-major))
        (dolist (transpose *blas-transposes*)
          (dolist (shape '((9 7 2 1) (6 6 0 3) (5 8 3 0)))
            (dolist (increments '((1 1) (-2 3)))
              (destructuring-bind (rows cols kl ku) shape
                (destructuring-bind (incx incy) increments
                  (with-blas-cleanup
                    (let* ((trans (not (eq transpose :no-transpose)))
                           (x-count (if trans rows cols)) (y-count (if trans cols rows))
                           (alpha (blas-scalar type 3/4 -1/4))
                           (beta (blas-scalar type 1/2 1/4)))
                      (multiple-value-bind (a ap)
                          (new-band type rows cols layout :general kl ku)
                        (multiple-value-bind (x xb) (new-vector type x-count incx)
                          (multiple-value-bind (y yb) (new-vector type y-count incy)
                            (cblas-call (cblas-name type "gbmv")
                                        (arg-enum layout) (arg-enum transpose)
                                        (arg-int rows) (arg-int cols) (arg-int kl)
                                        (arg-int ku) (arg-scalar type alpha)
                                        (arg-pointer ap) (arg-int (view-ld a))
                                        (arg-pointer xb) (arg-int incx)
                                        (arg-scalar type beta)
                                        (arg-pointer yb) (arg-int incy))
                            (funcall (blas-routine type "gbmv") transpose alpha a x
                                     incx beta y incy)
                            (check-case (list type layout transpose shape increments)
                                        (and (result-matches-p y type)
                                             (result-matches-p a type)))))))))))))))))

(test blas-random-sbmv-spmv
  (dolist (type *blas-random-types*)
    (let ((band-family (if (consp type) "hbmv" "sbmv"))
          (packed-family (if (consp type) "hpmv" "spmv"))
          (kind (if (consp type) :hermitian :symmetric)))
      (dolist (layout '(:row-major :column-major))
        (dolist (uplo '(:upper :lower))
          (dolist (increments '((1 1) (-2 3)))
            (destructuring-bind (incx incy) increments
              (when (cblas-available-p type band-family)
                (dolist (k '(0 2 5))
                  (with-blas-cleanup
                    (let ((n 9) (alpha (blas-scalar type 3/4 -1/4))
                          (beta (blas-scalar type 1/2 1/4)))
                      (multiple-value-bind (a ap) (new-band type n n layout kind k k)
                        (multiple-value-bind (x xb) (new-vector type n incx)
                          (multiple-value-bind (y yb) (new-vector type n incy)
                            (cblas-call (cblas-name type band-family)
                                        (arg-enum layout) (arg-enum uplo) (arg-int n)
                                        (arg-int k) (arg-scalar type alpha)
                                        (arg-pointer ap) (arg-int (view-ld a))
                                        (arg-pointer xb) (arg-int incx)
                                        (arg-scalar type beta)
                                        (arg-pointer yb) (arg-int incy))
                            (funcall (blas-routine type band-family) uplo alpha a x
                                     incx beta y incy)
                            (check-case (list type band-family layout uplo k incx)
                                        (and (result-matches-p y type)
                                             (result-matches-p a type))))))))))
              (when (cblas-available-p type packed-family)
                (with-blas-cleanup
                  (let ((n 9) (alpha (blas-scalar type 3/4 -1/4))
                        (beta (blas-scalar type 1/2 1/4)))
                    (multiple-value-bind (a ap) (new-packed type n layout)
                      (multiple-value-bind (x xb) (new-vector type n incx)
                        (multiple-value-bind (y yb) (new-vector type n incy)
                          (cblas-call (cblas-name type packed-family)
                                      (arg-enum layout) (arg-enum uplo) (arg-int n)
                                      (arg-scalar type alpha) (arg-pointer ap)
                                      (arg-pointer xb) (arg-int incx)
                                      (arg-scalar type beta)
                                      (arg-pointer yb) (arg-int incy))
                          (funcall (blas-routine type packed-family) uplo alpha a x
                                   incx beta y incy)
                          (check-case (list type packed-family layout uplo incx)
                                      (result-matches-p y type)))))))))))))))

(test blas-random-triangular-band-and-packed
  (dolist (type *blas-random-types*)
    (dolist (operation '("mv" "sv"))
      (let ((band-family (concatenate 'string "tb" operation))
            (packed-family (concatenate 'string "tp" operation)))
        (dolist (layout '(:row-major :column-major))
          (dolist (uplo '(:upper :lower))
            (dolist (transpose *blas-transposes*)
              (dolist (diag '(:non-unit :unit))
                (dolist (incx '(1 -2))
                  (when (cblas-available-p type band-family)
                    (dolist (k '(0 3))
                      (with-blas-cleanup
                        (let ((n 9))
                          (multiple-value-bind (a ap)
                              (new-band type n n layout :triangular k k
                                        :uplo uplo :dominant t)
                            (multiple-value-bind (x xb) (new-vector type n incx)
                              (cblas-call (cblas-name type band-family)
                                          (arg-enum layout) (arg-enum uplo)
                                          (arg-enum transpose) (arg-enum diag)
                                          (arg-int n) (arg-int k) (arg-pointer ap)
                                          (arg-int (view-ld a)) (arg-pointer xb)
                                          (arg-int incx))
                              (funcall (blas-routine type band-family) uplo transpose
                                       diag a x incx)
                              (check-case (list type band-family layout uplo transpose
                                                diag k incx)
                                          (result-matches-p x type))))))))
                  (when (cblas-available-p type packed-family)
                    (with-blas-cleanup
                      (let ((n 9))
                        (multiple-value-bind (a ap)
                            (new-packed type n layout :uplo uplo :dominant t)
                          (multiple-value-bind (x xb) (new-vector type n incx)
                            (cblas-call (cblas-name type packed-family)
                                        (arg-enum layout) (arg-enum uplo)
                                        (arg-enum transpose) (arg-enum diag)
                                        (arg-int n) (arg-pointer ap) (arg-pointer xb)
                                        (arg-int incx))
                            (funcall (blas-routine type packed-family) uplo transpose
                                     diag a x incx)
                            (check-case (list type packed-family layout uplo transpose
                                              diag incx)
                                        (result-matches-p x type))))))))))))))))

(test blas-random-rank-updates-two-vectors-and-packed
  (dolist (type *blas-random-types*)
    (let ((hermitian (consp type)))
      (dolist (layout '(:row-major :column-major))
        (dolist (uplo '(:upper :lower))
          (dolist (increments '((1 1) (-2 3)))
            (destructuring-bind (incx incy) increments
              ;; dense syr2/her2 and packed syr/spr2/hpr/hpr2
              (let ((family (if hermitian "her2" "syr2")))
                (when (cblas-available-p type family)
                  (with-blas-cleanup
                    (let ((n 9) (alpha (blas-scalar type 3/4 -1/4)))
                      (multiple-value-bind (a ap) (new-matrix type n n layout)
                        (multiple-value-bind (x xb) (new-vector type n incx)
                          (multiple-value-bind (y yb) (new-vector type n incy)
                            (cblas-call (cblas-name type family)
                                        (arg-enum layout) (arg-enum uplo) (arg-int n)
                                        (arg-scalar type alpha) (arg-pointer xb)
                                        (arg-int incx) (arg-pointer yb) (arg-int incy)
                                        (arg-pointer ap) (arg-int (view-ld a)))
                            (funcall (blas-routine type family) uplo alpha x incx y
                                     incy a)
                            (check-case (list type family layout uplo incx incy)
                                        (result-matches-p a type)))))))))
              (let ((family (if hermitian "hpr" "spr")))
                (when (cblas-available-p type family)
                  (with-blas-cleanup
                    (let ((n 9) (alpha (coerce 3/4 (blas-component type))))
                      (multiple-value-bind (a ap) (new-packed type n layout)
                        (multiple-value-bind (x xb) (new-vector type n incx)
                          (cblas-call (cblas-name type family)
                                      (arg-enum layout) (arg-enum uplo) (arg-int n)
                                      (arg-real type alpha) (arg-pointer xb)
                                      (arg-int incx) (arg-pointer ap))
                          (funcall (blas-routine type family) uplo alpha x incx a)
                          (check-case (list type family layout uplo incx)
                                      (result-matches-p a type))))))))
              (let ((family (if hermitian "hpr2" "spr2")))
                (when (cblas-available-p type family)
                  (with-blas-cleanup
                    (let ((n 9) (alpha (blas-scalar type 3/4 -1/4)))
                      (multiple-value-bind (a ap) (new-packed type n layout)
                        (multiple-value-bind (x xb) (new-vector type n incx)
                          (multiple-value-bind (y yb) (new-vector type n incy)
                            (cblas-call (cblas-name type family)
                                        (arg-enum layout) (arg-enum uplo) (arg-int n)
                                        (arg-scalar type alpha) (arg-pointer xb)
                                        (arg-int incx) (arg-pointer yb) (arg-int incy)
                                        (arg-pointer ap))
                            (funcall (blas-routine type family) uplo alpha x incx y
                                     incy a)
                            (check-case (list type family layout uplo incx incy)
                                        (result-matches-p a type))))))))))))))))

;;; Zero scalars never read the operands they scale

(defun blas-poison (type)
  "Return an infinity of TYPE's component, or NIL when the Lisp has no constant."
  (flet ((find-infinity (name)
           (let ((symbol (some (lambda (package)
                                 (let ((package (find-package package)))
                                   (and package (find-symbol name package))))
                               '("SB-EXT" "CCL" "EXT"))))
             (and symbol (boundp symbol) (symbol-value symbol)))))
    (let* ((component (blas-component type))
           (infinity (if (eq component 'single-float)
                         (or (find-infinity "SINGLE-FLOAT-POSITIVE-INFINITY")
                             (let ((double (find-infinity "DOUBLE-FLOAT-POSITIVE-INFINITY")))
                               (and double (ignore-errors (coerce double 'single-float)))))
                         (find-infinity "DOUBLE-FLOAT-POSITIVE-INFINITY"))))
      (when infinity
        (if (consp type) (complex infinity infinity) infinity)))))

(defun new-test-view (type rows cols)
  (trivial-simd/blas:make-matrix-view (blas-filled-vector type (* rows cols))
                                      rows cols))

(defun view-data (view) (trivial-simd/blas:matrix-view-data view))

(test blas-level3-zero-scalars-skip-operands
  (dolist (type *blas-random-types*)
    (let ((poison (blas-poison type)))
      (when poison
        (let ((zero (blas-scalar type 0)) (one (blas-scalar type 1))
              (alpha (blas-scalar type 3/4)))
          (let ((a (new-test-view type 6 5)) (b (new-test-view type 5 4))
                (poisoned (new-test-view type 6 4))
                (expected (new-test-view type 6 4)))
            (fill (view-data poisoned) poison)
            (funcall (blas-routine type "gemm") :no-transpose :no-transpose
                     alpha a b zero poisoned)
            (funcall (blas-routine type "gemm") :no-transpose :no-transpose
                     alpha a b zero expected)
            (is (equalp (view-data poisoned) (view-data expected))
                "gemm with zero beta must not read C"))
          (let ((a (new-test-view type 6 5)) (b (new-test-view type 5 4))
                (c (new-test-view type 6 4)))
            (let ((before (copy-seq (view-data c))))
              (fill (view-data a) poison)
              (fill (view-data b) poison)
              (funcall (blas-routine type "gemm") :no-transpose :no-transpose
                       zero a b one c)
              (is (equalp before (view-data c))
                  "gemm with zero alpha must not read A or B")))
          (let ((a (new-test-view type 6 5)) (c (new-test-view type 6 6)))
            (let ((before (copy-seq (view-data c))))
              (fill (view-data a) poison)
              (funcall (blas-routine type "syrk") :upper :no-transpose zero a one c)
              (is (equalp before (view-data c))
                  "syrk with zero alpha must not read A")))
          (let ((a (new-test-view type 5 5)) (b (new-test-view type 5 4)))
            (fill (view-data a) poison)
            (funcall (blas-routine type "trsm") :left :upper :no-transpose
                     :non-unit zero a b)
            (is (every #'zerop (view-data b))
                "trsm with zero alpha zeroes B without reading A")))))))

(test blas-level2-zero-scalars-skip-operands
  (dolist (type *blas-random-types*)
    (let ((poison (blas-poison type)))
      (when poison
        (let ((zero (blas-scalar type 0)) (one (blas-scalar type 1))
              (alpha (blas-scalar type 3/4)))
          (let ((a (new-test-view type 6 5)) (x (blas-filled-vector type 5))
                (y (blas-filled-vector type 6)))
            (let ((before (copy-seq y)))
              (fill (view-data a) poison)
              (fill x poison)
              (funcall (blas-routine type "gemv") :no-transpose zero a x 1 one y 1)
              (is (equalp before y)
                  "gemv with zero alpha and unit beta leaves y alone")))
          (let ((a (new-test-view type 6 5)) (x (blas-filled-vector type 5))
                (y (blas-filled-vector type 6))
                (expected (blas-filled-vector type 6)))
            (fill y poison)
            (funcall (blas-routine type "gemv") :no-transpose alpha a x 1 zero y 1)
            (funcall (blas-routine type "gemv") :no-transpose alpha a x 1 zero
                     expected 1)
            (is (equalp y expected) "gemv with zero beta must not read y")))))))

(test blas-hermitian-updates-keep-diagonal-real
  (dolist (type '((complex single-float) (complex double-float)))
    (let* ((n 6) (k 4)
           (a (new-test-view type n k)) (b (new-test-view type n k))
           (c (new-test-view type n n))
           (real-alpha (coerce 3/4 (second type)))
           (real-beta (coerce 1/2 (second type))))
      (funcall (blas-routine type "herk") :upper :no-transpose real-alpha a
               real-beta c)
      (dotimes (i n)
        (is (zerop (imagpart (trivial-simd/blas:matrix-ref c i i)))))
      (funcall (blas-routine type "her2k") :lower :no-transpose
               (blas-scalar type 3/4 1/4) a b real-beta c)
      (dotimes (i n)
        (is (zerop (imagpart (trivial-simd/blas:matrix-ref c i i))))))))

(test blas-random-elements-match-rational
  (dolist (type *blas-random-types*)
    (let* ((*blas-random-seed* 12345)
           (expected (loop repeat 64 collect
                           (let ((real (coerce (/ (- (blas-random 17) 8) 8) (blas-component type))))
                             (if (consp type)
                                 (complex real (coerce (/ (- (blas-random 17) 8) 8) (blas-component type)))
                                 real))))
           (final-seed *blas-random-seed*))
      (setf *blas-random-seed* 12345)
      (is (equalp expected (loop repeat 64 collect (blas-random-element type))))
      (is (= final-seed *blas-random-seed*)))))
