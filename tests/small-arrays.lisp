(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

;; Plain functions give the compiler macros a call site to expand: ECL does not
;; apply them to calls written directly inside FiveAM test bodies.
(defmacro define-callers (&rest operations)
  `(progn
     ,@(loop for (name lambda-list operation) in operations
             collect `(defun ,name ,lambda-list (,operation ,@lambda-list)))))

(define-callers
  (call-add! (out left right) simd:add!)
  (call-subtract! (out left right) simd:subtract!)
  (call-multiply! (out left right) simd:multiply!)
  (call-divide! (out left right) simd:divide!)
  (call-scale! (x a) simd:scale!)
  (call-axpy! (y a x) simd:axpy!)
  (call-sum (input) simd:sum)
  (call-dot (left right) simd:dot)
  (call-negate! (out input) simd:negate!)
  (call-abs! (out input) simd:abs!)
  (call-reciprocal! (out input) simd:reciprocal!)
  (call-min! (out left right) simd:min!)
  (call-max! (out left right) simd:max!)
  (call-clamp! (out input lower upper) simd:clamp!))

(defmacro with-inline-small-arrays ((enabled) &body body)
  `(let ((simd::*inline-small-arrays* ,enabled)) ,@body))

(defun small-lengths ()
  (append (loop for length from 0 to 9 collect length)
          (list 31 32 33 simd::+small-array-limit+ (1+ simd::+small-array-limit+) 200)))

(defun fresh (length type)
  (make-array length :element-type type))

(defmacro define-inline-elementwise-test (name caller)
  `(test ,name
     (dolist (backend (available-backends))
       (dolist (type '(single-float double-float))
         (dolist (length (small-lengths))
           (let ((left (values-for length type 1))
                 (right (values-for length type 2))
                 (inline (fresh length type))
                 (full (fresh length type)))
             (with-backend (backend)
               (with-inline-small-arrays (t)
                 (is (eq inline (,caller inline left right))))
               (with-inline-small-arrays (nil)
                 (,caller full left right)))
             (is (equalp inline full))))))))

(define-inline-elementwise-test inline-add-matches-backend call-add!)
(define-inline-elementwise-test inline-subtract-matches-backend call-subtract!)
(define-inline-elementwise-test inline-multiply-matches-backend call-multiply!)
(define-inline-elementwise-test inline-divide-matches-backend call-divide!)
(define-inline-elementwise-test inline-min-matches-backend call-min!)
(define-inline-elementwise-test inline-max-matches-backend call-max!)

(defmacro define-inline-unary-test (name caller)
  `(test ,name
     (dolist (backend (available-backends))
       (dolist (type '(single-float double-float))
         (dolist (length (small-lengths))
           (let ((input (make-array length :element-type type
                                           :initial-contents
                                           (loop for i below length
                                                 collect (coerce (* (if (oddp i) -1 1) (1+ i)) type))))
                 (inline (fresh length type))
                 (full (fresh length type)))
             (with-backend (backend)
               (with-inline-small-arrays (t)
                 (is (eq inline (,caller inline input))))
               (with-inline-small-arrays (nil)
                 (,caller full input)))
             (is (equalp inline full))))))))

(define-inline-unary-test inline-negate-matches-backend call-negate!)
(define-inline-unary-test inline-abs-matches-backend call-abs!)
(define-inline-unary-test inline-reciprocal-matches-backend call-reciprocal!)

(test inline-clamp-matches-backend-and-keeps-validating
  (dolist (backend (available-backends))
    (dolist (type '(single-float double-float))
      (dolist (length (small-lengths))
        (let ((input (values-for length type 1))
              (inline (fresh length type))
              (full (fresh length type))
              (low (coerce 3 type))
              (high (coerce 7 type)))
          (with-backend (backend)
            (with-inline-small-arrays (t)
              (is (eq inline (call-clamp! inline input low high))))
            (with-inline-small-arrays (nil)
              (call-clamp! full input low high))
            (with-inline-small-arrays (t)
              (signals error (call-clamp! inline input high low))))
          (is (equalp inline full)))))))

(test inline-scale-and-axpy-match-backend
  (dolist (backend (available-backends))
    (dolist (type '(single-float double-float))
      (dolist (length (small-lengths))
        (let ((scalar (coerce 3 type))
              (x (values-for length type 2))
              (inline (values-for length type 1))
              (full (values-for length type 1)))
          (with-backend (backend)
            (with-inline-small-arrays (t)
              (is (eq inline (call-scale! inline scalar)))
              (call-axpy! inline scalar x))
            (with-inline-small-arrays (nil)
              (call-scale! full scalar)
              (call-axpy! full scalar x)))
          (is (equalp inline full)))))))

(test inline-sum-and-dot-match-backend
  (dolist (backend (available-backends))
    (dolist (type '(single-float double-float))
      (dolist (length (small-lengths))
        (let ((left (values-for length type 1))
              (right (values-for length type 2)))
          (with-backend (backend)
            (is (close-enough-p (with-inline-small-arrays (t) (call-sum left))
                                (with-inline-small-arrays (nil) (call-sum left))
                                type))
            (is (close-enough-p (with-inline-small-arrays (t) (call-dot left right))
                                (with-inline-small-arrays (nil) (call-dot left right))
                                type))))))))

(test inline-empty-reductions-return-a-typed-zero
  (with-inline-small-arrays (t)
    (is (eql 0.0f0 (call-sum (fresh 0 'single-float))))
    (is (eql 0.0d0 (call-dot (fresh 0 'double-float) (fresh 0 'double-float))))))

(test inline-path-falls-back-for-other-types
  (with-inline-small-arrays (t)
    (let ((left (make-array 4 :element-type '(unsigned-byte 8) :initial-element 200))
          (right (make-array 4 :element-type '(unsigned-byte 8) :initial-element 100))
          (out (make-array 4 :element-type '(unsigned-byte 8))))
      (call-add! out left right)
      (is (equalp out (make-array 4 :element-type '(unsigned-byte 8) :initial-element 44))))))

(test inline-path-keeps-validating
  (with-inline-small-arrays (t)
    (let ((short (fresh 3 'single-float))
          (long (fresh 4 'single-float))
          (double (fresh 4 'double-float)))
      (signals error (call-add! long long short))
      (signals error (call-add! long long double))
      (signals error (call-scale! long 2d0))
      (signals error (call-scale! long 2))
      (signals error (call-axpy! long 1f0 short))
      (signals error (call-dot long short))
      (signals error (call-dot long double)))))

(test inline-expansion-declines-calls-with-keywords
  (let ((expander (compiler-macro-function 'simd:add!))
        (keyword-form '(simd:add! out left right :start 0))
        (plain-form '(simd:add! out left right)))
    (is (eq keyword-form (funcall expander keyword-form nil)))
    (is (not (eq plain-form (funcall expander plain-form nil))))))

(test inline-path-bypasses-the-backend
  (let ((left (values-for 4 'single-float 1))
        (right (values-for 4 'single-float 2))
        (out (fresh 4 'single-float))
        (simd::*backend* :unavailable))
    (with-inline-small-arrays (t)
      (is (eq out (call-add! out left right))))
    (with-inline-small-arrays (nil)
      (signals condition (call-add! out left right)))))
