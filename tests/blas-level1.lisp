(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(test blas-level1-vector-families
  (dolist (case '((single-float sswap scopy sscal sdot sasum snrm2 isamax)
                  (double-float dswap dcopy dscal ddot dasum dnrm2 idamax)
                  ((complex single-float) cswap ccopy cscal cdotu scasum scnrm2 icamax)
                  ((complex double-float) zswap zcopy zscal zdotu dzasum dznrm2 izamax)))
    (destructuring-bind (type swap copy scale dot asum nrm2 iamax) case
      (let* ((a (if (consp type)
                    (complex (coerce 1 (second type)) (coerce 2 (second type)))
                    (coerce 1 type)))
             (b (if (consp type)
                    (complex (coerce 3 (second type)) (coerce 4 (second type)))
                    (coerce 3 type)))
             (zero (if (consp type)
                       (complex (coerce 0 (second type)) (coerce 0 (second type)))
                       (coerce 0 type)))
             (x (complex-vector type (list a zero b zero a)))
             (y (complex-vector type (list b zero a zero b)))
             (pkg (find-package :trivial-simd/blas)))
        (flet ((call (name &rest args)
                 (apply (symbol-function (find-symbol (symbol-name name) pkg)) args)))
          (is (= (call dot 3 x 2 y -2) (+ (* a b) (* b a) (* a b))))
          (is (= (call asum 3 x 2)
                 (+ (* 2 (trivial-simd/blas::absolute-one a))
                    (trivial-simd/blas::absolute-one b))))
          (is (plusp (call nrm2 3 x 2)))
          (is (= (call iamax 3 x 2) 1))
          (is (eq y (call copy 3 x 2 y -2)))
          (is (= (aref y 4) a))
          (call swap 3 x 2 y 2)
          (is (= (aref x 0) a))
          (is (eq x (call scale 3 a x 2)))
          (is (= (aref x 0) (* a a))))))))

(test blas-level1-special-cases
  (let* ((x (blas-vector 'single-float '(1.0f0 2.0f0 3.0f0 4.0f0 5.0f0)))
         (y (blas-vector 'single-float '(5.0f0 4.0f0 3.0f0 2.0f0 1.0f0))))
    (is (= (trivial-simd/blas:dsdot 3 x 2 y 2) 19.0d0))
    (is (= (trivial-simd/blas:sdsdot 3 1.0f0 x 2 y 2) 20.0f0))
    (is (= (trivial-simd/blas:isamax 0 x 1) 0))
    (trivial-simd/blas:srot 2 x 2 y 2 0.0f0 1.0f0)
    (is (= (aref x 0) 5.0f0))
    (is (= (aref y 0) -1.0f0))
    (let ((parameters (blas-vector 'single-float '(-2.0f0 0.0f0 0.0f0 0.0f0 0.0f0)))
          (before (copy-seq x)))
      (trivial-simd/blas:srotm 2 x 1 y 1 parameters)
      (is (equalp x before)))
    (multiple-value-bind (r z c s) (trivial-simd/blas:srotg 3.0f0 4.0f0)
      (is (close-enough-p r 5.0f0 'single-float))
      (is (close-enough-p z (/ 1.0f0 c) 'single-float))
      (is (close-enough-p s 0.8f0 'single-float)))
    (signals error (trivial-simd/blas:sscal 2 1.0d0 x 1))
    (signals error (trivial-simd/blas:scopy 3 x 0 y 1))
    (signals error (trivial-simd/blas:snrm2 3 x 2 :x-offset 1))))

(test blas-level1-convenience
  (let ((x (blas-vector 'double-float '(1.0d0 2.0d0 3.0d0)))
        (y (blas-vector 'double-float '(4.0d0 5.0d0 6.0d0))))
    (is (= (trivial-simd/blas/convenience:dot x y) 32.0d0))
    (is (eq x (trivial-simd/blas/convenience:scal! x 2.0d0 :start 1)))
    (is (equalp x (blas-vector 'double-float '(1.0d0 4.0d0 6.0d0))))
    (is (eq y (trivial-simd/blas/convenience:copy! y x)))
    (is (equalp y x))
    (is (= (trivial-simd/blas/convenience:norm2 y)
           (trivial-simd/blas:dnrm2 3 y 1)))
    (multiple-value-bind (first second)
        (trivial-simd/blas/convenience:swap! x y)
      (is (eq first x))
      (is (eq second y)))))

(test blas-complex-special-variants
  (dolist (case '((single-float caxpy csscal csrot crot)
                  (double-float zaxpy zdscal zdrot zrot)))
    (destructuring-bind (type axpy real-scal real-rot complex-rot) case
      (let* ((one (coerce 1 type)) (zero (coerce 0 type))
             (a (complex one (* 2 one)))
             (b (complex (* 3 one) (* 4 one)))
             (x (complex-vector `(complex ,type) (list a b a)))
             (y (complex-vector `(complex ,type) (list b a b)))
             (pkg (find-package :trivial-simd/blas)))
        (flet ((call (name &rest args)
                 (apply (symbol-function (find-symbol (symbol-name name) pkg)) args)))
          (call axpy 1 a x 1 y 1)
          (is (= (aref y 0) (+ (* a a) b)))
          (call real-scal 1 (* 2 one) x 1)
          (is (= (aref x 0) (* 2 a)))
          (call real-rot 1 x 1 y 1 zero one)
          (is (= (aref x 0) (+ (* a a) b)))
          (is (= (aref y 0) (* -2 a)))
          (let ((old-x (aref x 1)) (old-y (aref y 1))
                (imaginary (complex zero one)))
            (call complex-rot 1 x 1 y 1 zero imaginary :x-offset 1 :y-offset 1)
            (is (= (aref x 1) (* imaginary old-y)))
            (is (= (aref y 1) (* imaginary old-x)))))))))

(test blas-level1-rotmg-reference
  (when (reference-blas-available-p)
    (dolist (case '((single-float :float srotmg)
                    (double-float :double drotmg)))
      (destructuring-bind (type foreign routine) case
        (dolist (inputs '((1 2 3 4) (4 2 3 1) (1 0 2 3) (-1 2 3 4)
                          (1/100000000 1 2 3)))
          (destructuring-bind (d1 d2 b1 b2) inputs
            (setf d1 (coerce d1 type) d2 (coerce d2 type)
                  b1 (coerce b1 type) b2 (coerce b2 type))
            (cffi:with-foreign-objects ((fd1 foreign) (fd2 foreign)
                                        (fb1 foreign) (fp foreign 5))
              (setf (cffi:mem-ref fd1 foreign) d1
                    (cffi:mem-ref fd2 foreign) d2
                    (cffi:mem-ref fb1 foreign) b1)
              (if (eq type 'single-float)
                  (cffi:foreign-funcall "cblas_srotmg"
                                        :pointer fd1 :pointer fd2 :pointer fb1
                                        :float b2 :pointer fp :void)
                  (cffi:foreign-funcall "cblas_drotmg"
                                        :pointer fd1 :pointer fd2 :pointer fb1
                                        :double b2 :pointer fp :void))
              (multiple-value-bind (a b c parameters)
                  (funcall (symbol-function
                            (find-symbol (symbol-name routine) :trivial-simd/blas))
                           d1 d2 b1 b2)
                (is (close-enough-p a (cffi:mem-ref fd1 foreign) type))
                (is (close-enough-p b (cffi:mem-ref fd2 foreign) type))
                (is (close-enough-p c (cffi:mem-ref fb1 foreign) type))
                (is (= (aref parameters 0) (cffi:mem-aref fp foreign 0)))
                (dolist (index (case (round (aref parameters 0))
                                 (-1 '(1 2 3 4)) (0 '(2 3)) (1 '(1 4))))
                  (is (close-enough-p (aref parameters index)
                                      (cffi:mem-aref fp foreign index) type)))))))))))

(defmacro define-real-level1-reference (name type foreign dot-name norm-name
                                         asum-name iamax-name dot norm asum iamax)
  `(test ,name
     (when (reference-blas-available-p)
       (let ((x (blas-vector ',type
                             (mapcar (lambda (v) (coerce v ',type)) '(1 -2 3 -4 5))))
             (y (blas-vector ',type
                             (mapcar (lambda (v) (coerce v ',type)) '(2 3 4 5 6)))))
         (cffi:with-foreign-objects ((fx ,foreign 5) (fy ,foreign 5))
           (dotimes (i 5)
             (setf (cffi:mem-aref fx ,foreign i) (aref x i)
                   (cffi:mem-aref fy ,foreign i) (aref y i)))
           (is (close-enough-p (,dot 3 x 2 y 2)
                               (cffi:foreign-funcall ,dot-name :int 3 :pointer fx :int 2
                                                     :pointer fy :int 2 ,foreign)
                               ',type))
           (is (close-enough-p (,norm 3 x 2)
                               (cffi:foreign-funcall ,norm-name :int 3 :pointer fx :int 2
                                                     ,foreign)
                               ',type))
           (is (close-enough-p (,asum 3 x 2)
                               (cffi:foreign-funcall ,asum-name :int 3 :pointer fx :int 2
                                                     ,foreign)
                               ',type))
           (is (= (,iamax 3 x 2)
                  (cffi:foreign-funcall ,iamax-name :int 3 :pointer fx :int 2 :int))))))))

(define-real-level1-reference blas-single-reference single-float :float
  "cblas_sdot" "cblas_snrm2" "cblas_sasum" "cblas_isamax"
  trivial-simd/blas:sdot trivial-simd/blas:snrm2
  trivial-simd/blas:sasum trivial-simd/blas:isamax)

(define-real-level1-reference blas-double-reference double-float :double
  "cblas_ddot" "cblas_dnrm2" "cblas_dasum" "cblas_idamax"
  trivial-simd/blas:ddot trivial-simd/blas:dnrm2
  trivial-simd/blas:dasum trivial-simd/blas:idamax)

(defmacro define-complex-level1-reference (name element foreign dot-name norm-name
                                            asum-name iamax-name dot norm asum iamax)
  `(test ,name
     (when (reference-blas-available-p)
       (flet ((typed (real imaginary)
                (complex (coerce real ',(second element))
                         (coerce imaginary ',(second element)))))
       (let ((x (complex-vector ',element
                                (list (typed 1 2) (typed 0 0) (typed -3 4)
                                      (typed 0 0) (typed 5 -6))))
             (y (complex-vector ',element
                                (list (typed 2 -1) (typed 0 0) (typed 3 2)
                                      (typed 0 0) (typed -4 1)))))
         (cffi:with-foreign-objects ((fx ,foreign 10) (fy ,foreign 10)
                                     (result ,foreign 2))
           (dotimes (i 5)
             (setf (cffi:mem-aref fx ,foreign (* 2 i)) (realpart (aref x i))
                   (cffi:mem-aref fx ,foreign (1+ (* 2 i))) (imagpart (aref x i))
                   (cffi:mem-aref fy ,foreign (* 2 i)) (realpart (aref y i))
                   (cffi:mem-aref fy ,foreign (1+ (* 2 i))) (imagpart (aref y i))))
           (cffi:foreign-funcall ,dot-name :int 3 :pointer fx :int 2
                                 :pointer fy :int 2 :pointer result :void)
           (let ((reference (complex (cffi:mem-aref result ,foreign 0)
                                     (cffi:mem-aref result ,foreign 1)))
                 (actual (,dot 3 x 2 y 2)))
             (is (close-enough-p (realpart actual) (realpart reference)
                                 ',(second element)))
             (is (close-enough-p (imagpart actual) (imagpart reference)
                                 ',(second element))))
           (is (close-enough-p (,norm 3 x 2)
                               (cffi:foreign-funcall ,norm-name :int 3 :pointer fx :int 2
                                                     ,foreign)
                               ',(second element)))
           (is (close-enough-p (,asum 3 x 2)
                               (cffi:foreign-funcall ,asum-name :int 3 :pointer fx :int 2
                                                     ,foreign)
                               ',(second element)))
           (is (= (,iamax 3 x 2)
                  (cffi:foreign-funcall ,iamax-name :int 3 :pointer fx :int 2 :int)))))))))

(define-complex-level1-reference blas-complex-single-reference
  (complex single-float) :float "cblas_cdotc_sub" "cblas_scnrm2"
  "cblas_scasum" "cblas_icamax" trivial-simd/blas:cdotc
  trivial-simd/blas:scnrm2 trivial-simd/blas:scasum trivial-simd/blas:icamax)

(define-complex-level1-reference blas-complex-double-reference
  (complex double-float) :double "cblas_zdotc_sub" "cblas_dznrm2"
  "cblas_dzasum" "cblas_izamax" trivial-simd/blas:zdotc
  trivial-simd/blas:dznrm2 trivial-simd/blas:dzasum trivial-simd/blas:izamax)

(defmacro define-complex-rotg-reference (name foreign real-type c-name routine)
  `(test ,name
     (when (and (reference-blas-available-p)
                (ignore-errors (cffi:foreign-symbol-pointer ,c-name)))
       (let ((a (complex (coerce 3 ',real-type) (coerce 4 ',real-type)))
             (b (complex (coerce 5 ',real-type) (coerce -2 ',real-type))))
         (cffi:with-foreign-objects ((fa ,foreign 2) (fb ,foreign 2)
                                     (fc ,foreign) (fs ,foreign 2))
           (setf (cffi:mem-aref fa ,foreign 0) (realpart a)
                 (cffi:mem-aref fa ,foreign 1) (imagpart a)
                 (cffi:mem-aref fb ,foreign 0) (realpart b)
                 (cffi:mem-aref fb ,foreign 1) (imagpart b))
           ;; Some libraries compute an infinity or NaN here, so mask traps to
           ;; get a value to report instead of an error from inside the call.
           (without-float-traps
             (cffi:foreign-funcall ,c-name :pointer fa :pointer fb :pointer fc
                                   :pointer fs :void))
           (multiple-value-bind (r unchanged c s) (,routine a b)
             (let ((reference-r (complex (cffi:mem-aref fa ,foreign 0)
                                         (cffi:mem-aref fa ,foreign 1)))
                   (reference-c (cffi:mem-ref fc ,foreign))
                   (reference-s (complex (cffi:mem-aref fs ,foreign 0)
                                         (cffi:mem-aref fs ,foreign 1))))
               (flet ((check (label actual expected)
                        ;; An infinite reference would widen the tolerance to
                        ;; infinity and accept any value, so require a finite one.
                        (is (without-float-traps
                              (and (= expected expected)
                                   (< (abs expected) most-positive-double-float)
                                   (close-enough-p actual expected ',real-type)))
                            "~A(~A, ~A) ~A: computed ~A, reference ~A~%  ~
                             computed  r=~A c=~A s=~A~%  ~
                             reference r=~A c=~A s=~A"
                            ,c-name a b label actual expected
                            r c s reference-r reference-c reference-s)))
                 (is (= unchanged b) "~A: B changed from ~A to ~A"
                     ',routine b unchanged)
                 (check "realpart r" (realpart r) (realpart reference-r))
                 (check "imagpart r" (imagpart r) (imagpart reference-r))
                 (check "c" c reference-c)
                 (check "realpart s" (realpart s) (realpart reference-s))
                 (check "imagpart s" (imagpart s) (imagpart reference-s))))))))))

(define-complex-rotg-reference blas-crotg-reference :float single-float
  "cblas_crotg" trivial-simd/blas:crotg)
(define-complex-rotg-reference blas-zrotg-reference :double double-float
  "cblas_zrotg" trivial-simd/blas:zrotg)

(defmacro define-real-rotation-reference (name type foreign rot-name rotm-name
                                           rotg-name rot rotm rotg)
  `(test ,name
     (when (reference-blas-available-p)
       (let ((x (blas-vector ',type
                             (mapcar (lambda (v) (coerce v ',type)) '(1 2 3 4 5))))
             (y (blas-vector ',type
                             (mapcar (lambda (v) (coerce v ',type)) '(5 4 3 2 1))))
             (c (coerce 3/5 ',type))
             (s (coerce 4/5 ',type)))
         (cffi:with-foreign-objects ((fx ,foreign 5) (fy ,foreign 5)
                                     (fa ,foreign) (fb ,foreign)
                                     (fc ,foreign) (fs ,foreign)
                                     (fp ,foreign 5))
           (dotimes (i 5)
             (setf (cffi:mem-aref fx ,foreign i) (aref x i)
                   (cffi:mem-aref fy ,foreign i) (aref y i)))
           (cffi:foreign-funcall ,rot-name :int 3 :pointer fx :int 2
                                 :pointer fy :int 2 ,foreign c ,foreign s :void)
           (,rot 3 x 2 y 2 c s)
           (dotimes (i 5)
             (is (close-enough-p (aref x i) (cffi:mem-aref fx ,foreign i) ',type))
             (is (close-enough-p (aref y i) (cffi:mem-aref fy ,foreign i) ',type)))
           (setf (cffi:mem-ref fa ,foreign) (coerce 3 ',type)
                 (cffi:mem-ref fb ,foreign) (coerce 4 ',type))
           (cffi:foreign-funcall ,rotg-name :pointer fa :pointer fb
                                 :pointer fc :pointer fs :void)
           (multiple-value-bind (r z cosine sine)
               (,rotg (coerce 3 ',type) (coerce 4 ',type))
             (is (close-enough-p r (cffi:mem-ref fa ,foreign) ',type))
             (is (close-enough-p z (cffi:mem-ref fb ,foreign) ',type))
             (is (close-enough-p cosine (cffi:mem-ref fc ,foreign) ',type))
             (is (close-enough-p sine (cffi:mem-ref fs ,foreign) ',type)))
           (dolist (flag '(-1 0 1 -2))
             (let ((parameters (blas-vector ',type
                                            (mapcar (lambda (v) (coerce v ',type))
                                                    (list flag 2 -1 3 4))))
                   (actual-x (copy-seq x)) (actual-y (copy-seq y)))
               (dotimes (i 5)
                 (setf (cffi:mem-aref fx ,foreign i) (aref x i)
                       (cffi:mem-aref fy ,foreign i) (aref y i)))
               (dotimes (i 5)
                 (setf (cffi:mem-aref fp ,foreign i) (aref parameters i)))
               (cffi:foreign-funcall ,rotm-name :int 3 :pointer fx :int 1
                                     :pointer fy :int 1 :pointer fp :void)
               (,rotm 3 actual-x 1 actual-y 1 parameters)
               (dotimes (i 5)
                 (is (close-enough-p (aref actual-x i)
                                     (cffi:mem-aref fx ,foreign i) ',type))
                 (is (close-enough-p (aref actual-y i)
                                     (cffi:mem-aref fy ,foreign i) ',type))))))))))

(define-real-rotation-reference blas-single-rotation-reference
  single-float :float "cblas_srot" "cblas_srotm" "cblas_srotg"
  trivial-simd/blas:srot trivial-simd/blas:srotm trivial-simd/blas:srotg)
(define-real-rotation-reference blas-double-rotation-reference
  double-float :double "cblas_drot" "cblas_drotm" "cblas_drotg"
  trivial-simd/blas:drot trivial-simd/blas:drotm trivial-simd/blas:drotg)
