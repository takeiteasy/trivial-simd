(eval-when (:compile-toplevel :load-toplevel :execute)
  (ql:quickload :trivial-simd))

(trivial-simd:define-kernel multiply-add (a b c) (+ (* a b) c))

(let ((a (make-array 4 :element-type 'single-float :initial-contents '(1.0 2.0 3.0 4.0)))
      (b (make-array 4 :element-type 'single-float :initial-element 2.0))
      (c (make-array 4 :element-type 'single-float :initial-element 0.5))
      (out (make-array 4 :element-type 'single-float)))
  (multiply-add out a b c)
  (format t "backend: ~A, a*b+c: ~S~%" (trivial-simd:backend) out))

(macrolet ((define-balanced-sum ()
             (labels ((sum-tree (depth)
                        (if (zerop depth) 'a
                            `(+ ,(sum-tree (1- depth)) ,(sum-tree (1- depth))))))
               `(trivial-simd:define-kernel sum-512 (a) ,(sum-tree 9)))))
  (define-balanced-sum))

(let ((a (make-array 257 :element-type 'single-float :initial-element 1.0))
      (out (make-array 257 :element-type 'single-float)))
  (sum-512 out a)
  (assert (every (lambda (value) (= value 512.0)) out)))

(trivial-simd:define-kernel rounded-multiply-add (a b c) (trivial-simd:fma a b c))
(dolist (type '(single-float double-float))
  (let* ((epsilon (coerce (if (eq type 'single-float) (expt 2 -23) (expt 2 -52)) type))
         (a (make-array 3 :element-type type :initial-element (+ (coerce 1 type) epsilon)))
         (b (make-array 3 :element-type type :initial-element (- (coerce 1 type) epsilon)))
         (c (make-array 3 :element-type type :initial-element (coerce -1 type)))
         (out (make-array 3 :element-type type)))
    (rounded-multiply-add out a b c)
    (assert (every (lambda (value) (= value (- (* epsilon epsilon)))) out))))
(trivial-simd:define-kernel magnitude-root (a) (sqrt (abs a)))

(let* ((a (make-array 5 :element-type 'single-float :initial-element -4.0))
       (out (make-array 5 :element-type 'single-float)))
  (magnitude-root out a)
  (assert (every (lambda (value) (= value 2.0)) out))
  (rounded-multiply-add out a a a)
  (assert (every (lambda (value) (= value 12.0)) out)))

(trivial-simd:define-kernel kernel-dot (a b) (trivial-simd:sum (* a b)))
(trivial-simd:define-kernel root-sum (a) (trivial-simd:sum (sqrt (abs a))))

(let ((a (make-array 257 :element-type 'single-float :initial-element -4.0))
      (b (make-array 257 :element-type 'single-float :initial-element 2.0)))
  (assert (= -2056.0 (kernel-dot a b)))
  (assert (= 514.0 (root-sum a)))
  (assert (= -72.0 (kernel-dot a b :end 9 :a-start 11))))

(trivial-simd:define-kernel loudest-difference (a b)
  (trivial-simd:argmax (abs (- a b))))
(trivial-simd:define-kernel difference-norm (a b)
  (trivial-simd:nrm2 (- a b)))

(let ((a (make-array 5 :element-type 'single-float :initial-contents '(1.0 2.0 9.0 4.0 5.0)))
      (b (make-array 5 :element-type 'single-float :initial-element 1.0)))
  (assert (= 2 (loudest-difference a b)))
  (assert (= 4 (loudest-difference a b :start 3 :end 5)))
  (assert (= 9.0 (trivial-simd:maximum a)))
  (assert (< (abs (- (difference-norm a b) (sqrt 90.0))) 1e-5)))

#+ecl
(progn
  (eval '(trivial-simd:define-kernel eval-multiply-add (a b c) (+ (* a b) c)))
  (eval '(trivial-simd:define-kernel eval-multiply-subtract (a b c) (- (* a b) c)))
  (let ((a (make-array 5 :element-type 'single-float :initial-element 2.0))
        (out (make-array 5 :element-type 'single-float)))
    (dotimes (i 2)
      (eval-multiply-add out a a a)
      (assert (every (lambda (value) (= value 6.0)) out)))
    (eval-multiply-subtract out a a a)
    (assert (every (lambda (value) (= value 2.0)) out))))
