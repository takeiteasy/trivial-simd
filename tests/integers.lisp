(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defparameter *integer-elements*
  (loop for bits in '(8 16 32 64)
        append (list (list 'signed-byte bits) (list 'unsigned-byte bits))))

(defun integer-reference (value type)
  (let* ((bits (second type))
         (modulus (ash 1 bits))
         (bits (mod value modulus)))
    (if (and (eq (first type) 'signed-byte) (>= bits (/ modulus 2)))
        (- bits modulus) bits)))

(defun integer-samples (type)
  (let* ((bits (second type))
         (minimum (if (eq (first type) 'signed-byte) (- (ash 1 (1- bits))) 0))
         (maximum (if (eq (first type) 'signed-byte)
                      (1- (ash 1 (1- bits))) (1- (ash 1 bits)))))
    (values (make-array 259 :element-type type
                        :initial-contents
                        (loop for i below 259 collect
                              (nth (mod i 8) (list maximum minimum 1 2 3 7 0 maximum))))
            (make-array 259 :element-type type
                        :initial-contents
                        (loop for i below 259 collect
                              (nth (mod i 8) (list 1 1 2 3 1 5 1 maximum)))))))

(defun integer-expected (operation a b type)
  (integer-reference
   (ecase operation
     (add! (+ a b)) (subtract! (- a b)) (multiply! (* a b))
     (divide! (truncate a b))) type))

(test integer-bulk-across-backends
  (dolist (type *integer-elements*)
    (multiple-value-bind (a b) (integer-samples type)
      (dolist (backend (available-backends))
        (with-backend (backend)
          (dolist (operation '(add! subtract! multiply! divide!))
            (let ((out (make-array 259 :element-type type)))
              (is (eq out (funcall (symbol-function (find-symbol (symbol-name operation) :trivial-simd))
                                   out a b)))
              (dotimes (i 259)
                (is (= (aref out i) (integer-expected operation (aref a i) (aref b i) type))))))
          (is (= (simd:sum a)
                 (integer-reference (loop for x across a sum x) type)))
          (is (= (simd:dot a b)
                 (integer-reference (loop for x across a for y across b sum (* x y)) type)))
          (is (= 0 (simd:sum (make-array 0 :element-type type))))
          (is (= 0 (simd:dot (make-array 0 :element-type type)
                             (make-array 0 :element-type type)))))))))

(simd:define-kernel integer-expression (a b) (max (abs (- a)) (+ (* a b) 3)))
(simd:define-kernel integer-expression-sum (a b)
  (simd:sum (max (abs (- a)) (+ (* a b) 3))))
(simd:define-kernel integer-divide (a b) (/ (- a) b))
(simd:define-kernel integer-out-of-range (a) (+ a 18446744073709551616))

(test integer-kernels-across-backends
  (dolist (type *integer-elements*)
    (multiple-value-bind (a b) (integer-samples type)
      (dolist (backend (available-backends))
        (with-backend (backend)
          (let* ((out (make-array 259 :element-type type :initial-element 0))
                 (expected
                   (loop for x across a for y across b
                         collect (max (integer-reference (abs (integer-reference (- x) type)) type)
                                      (integer-reference (+ (integer-reference (* x y) type) 3) type)))))
            (integer-expression out a b)
            (loop for actual across out for value in expected do (is (= actual value)))
            (is (= (integer-expression-sum a b)
                   (integer-reference (reduce #'+ expected) type)))
            (signals type-error (integer-out-of-range out a))
            (integer-divide out a b)
            (loop for actual across out for x across a for y across b
                  do (is (= actual (integer-reference
                                     (truncate (integer-reference (- x) type) y) type))))))))))

(test integer-slices-and-errors
  (dolist (type *integer-elements*)
    (dolist (backend (available-backends))
      (with-backend (backend)
        (let* ((a (make-array 20 :element-type type :initial-element 3))
               (b (make-array 20 :element-type type :initial-element 2))
               (out (make-array 20 :element-type type :initial-element 9)))
          (simd:add! out a b :start 2 :end 9 :destination-start 10 :left-start 1)
          (loop for i below 20 do (is (= (aref out i) (if (<= 10 i 16) 5 9))))
          (is (= (simd:sum a :start 2 :end 9) 21))
          (is (= (simd:dot a b :start 2 :end 9) 42))
          (setf (aref b 5) 0)
          (signals division-by-zero (simd:divide! out a b))
          (signals division-by-zero (integer-divide out a b))
          (signals error (simd:add! out a (make-array 20 :element-type 'single-float))))))))

(simd:define-kernel integer-pack-add (a b) (+ a b))
(simd:define-kernel integer-pack-sum (a b) (simd:sum (+ a b)))
(simd:define-kernel integer-square-root (a) (sqrt a))
(simd:define-kernel integer-fused (a b c) (simd:fma a b c))
(simd:define-kernel integer-fraction (a) (+ a 1/2))

(test integer-packed-kernels-and-invalid-expressions
  (dolist (type *integer-elements*)
    (multiple-value-bind (a b) (integer-samples type)
      (dolist (backend (available-backends))
        (with-backend (backend)
          (let ((out (make-array 259 :element-type type)))
            (integer-pack-add out a b)
            (dotimes (i 259)
              (is (= (aref out i)
                     (integer-reference (+ (aref a i) (aref b i)) type))))
            (is (= (integer-pack-sum a b)
                   (integer-reference (loop for x across a for y across b sum (+ x y)) type)))
            (signals error (integer-square-root out a))
            (signals error (integer-fused out a b a))
            (signals type-error (integer-fraction out a))))))))

(test integer-signed-division-edge
  (dolist (bits '(8 16 32 64))
    (let* ((type (list 'signed-byte bits))
           (minimum (- (ash 1 (1- bits))))
           (a (make-array 1 :element-type type :initial-element minimum))
           (b (make-array 1 :element-type type :initial-element -1)))
      (dolist (backend (available-backends))
        (with-backend (backend)
          (let ((out (make-array 1 :element-type type)))
            (simd:divide! out a b)
            (is (= minimum (aref out 0)))
            (integer-divide out a b)
            (is (= minimum (aref out 0)))))))))

(defun balanced-integer-expression (depth)
  (if (zerop depth) 'a
      (let ((part (balanced-integer-expression (1- depth))))
        (list '+ part part))))

(test integer-native-spill-and-reduction
  (when simd::*native-available-p*
    (let ((expression (balanced-integer-expression 8)))
      (eval `(simd:define-kernel integer-spill (a) ,expression))
      (eval `(simd:define-kernel integer-spill-sum (a) (simd:sum ,expression)))
      (let* ((a (make-array 257 :element-type '(unsigned-byte 8) :initial-element 3))
             (out (make-array 257 :element-type '(unsigned-byte 8))))
        (dolist (backend '(:native :native-copy))
          (with-backend (backend)
            (integer-spill out a)
            (is (every (lambda (x) (= x 0)) out))
            (is (= 0 (integer-spill-sum a)))))))))

(simd:define-kernel integer-u64-constant (a) (+ a 18446744073709551615))
(simd:define-kernel integer-s64-constant (a) (+ a -9223372036854775808))

(test integer-large-constants-are-exact
  (dolist (backend (available-backends))
    (with-backend (backend)
      (let ((unsigned (make-array 3 :element-type '(unsigned-byte 64)
                                  :initial-contents '(0 1 2)))
            (signed (make-array 3 :element-type '(signed-byte 64)
                                :initial-contents '(0 1 -1)))
            (unsigned-out (make-array 3 :element-type '(unsigned-byte 64)))
            (signed-out (make-array 3 :element-type '(signed-byte 64))))
        (integer-u64-constant unsigned-out unsigned)
        (integer-s64-constant signed-out signed)
        (is (equalp unsigned-out #(18446744073709551615 0 1)))
        (is (equalp signed-out #(-9223372036854775808 -9223372036854775807
                                 9223372036854775807)))))))

(test integer-aliases-and-empty-validation
  (dolist (type *integer-elements*)
    (multiple-value-bind (source right) (integer-samples type)
      (let ((expected (make-array 259 :element-type type)))
        (dotimes (i 259)
          (setf (aref expected i)
                (integer-reference (+ (aref source i) (aref right i)) type)))
        (dolist (backend (available-backends))
          (with-backend (backend)
            (let ((left (copy-seq source))
                  (right-copy (copy-seq right)))
              (simd:add! left left right)
              (simd:add! right-copy source right-copy)
              (is (equalp left expected))
              (is (equalp right-copy expected))))))))
  (dolist (backend (available-backends))
    (with-backend (backend)
      (let ((empty (make-array 0 :element-type '(signed-byte 8))))
        (signals type-error (integer-fraction empty empty))
        (signals error (integer-square-root empty empty))))))

(test integer-native-program-owns-typed-constants
  (let ((release nil)
        (frees 0)
        (free #'simd::free-native-program-buffers))
    (with-function-replaced
        (simd::register-native-program-finalizer
         (lambda (program callback)
           (declare (ignore program))
           (setf release callback)))
      (let ((program (simd::make-native-program '(1 255 0 0)
                                                 '(18446744073709551615) 0 :u64)))
        (is (= 18446744073709551615
               (cffi:mem-ref (simd::native-program-constants program) :uint64)))))
    (with-function-replaced
        (simd::free-native-program-buffers
         (lambda (buffers)
           (incf frees (count-if #'identity buffers))
           (funcall free buffers)))
      (funcall release)
      (funcall release)
      (is (= 2 frees)))))

(simd:define-kernel integer-concurrent (a b) (+ a b))

(test integer-native-concurrent-first-use
  (when (and simd::*native-available-p* bordeaux-threads:*supports-threads-p*)
    (let ((threads
            (loop repeat 4 collect
              (bordeaux-threads:make-thread
               (lambda ()
                 (with-backend (:native)
                   (let ((a (make-array 257 :element-type '(unsigned-byte 64)
                                        :initial-element 18446744073709551615))
                         (b (make-array 257 :element-type '(unsigned-byte 64)
                                        :initial-element 1))
                         (out (make-array 257 :element-type '(unsigned-byte 64))))
                     (integer-concurrent out a b)
                     (every #'zerop out))))))))
      (dolist (thread threads)
        (is (bordeaux-threads:join-thread thread))))))
