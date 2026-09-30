(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(simd:define-kernel mask-kernel-compare (a b) (> a b))
(simd:define-kernel mask-kernel-select (a b) (simd:select (> a b) a b))
(simd:define-kernel mask-kernel-count (a b) (simd:count (> a b)))
(simd:define-kernel mask-kernel-any (a b) (simd:any (> a b)))
(simd:define-kernel mask-kernel-all (a b) (simd:all (> a b)))
(simd:define-kernel mask-kernel-nested (a b c)
  (simd:select (> a b)
               (simd:select (< c a) (+ a c) (* a b))
               (- c b)))
(simd:define-kernel mask-kernel-nested-count (a b c)
  (simd:count (> (simd:select (> a b) a c) b)))
(simd:define-kernel mask-kernel-sum-select (a b)
  (simd:sum (simd:select (> a b) a b)))
(simd:define-kernel mask-kernel-positive (a)
  (simd:select (> a 0) a 0))

(macrolet ((define-spilling-mask-kernel ()
             (labels ((tree (depth on-true)
                        (if (zerop depth)
                            (if on-true 'a 'b)
                            `(simd:select (> a b)
                                          ,(tree (1- depth) t)
                                          ,(tree (1- depth) nil)))))
               `(progn
                  (simd:define-kernel mask-kernel-spilling (a b) ,(tree 7 t))
                  (simd:define-kernel mask-kernel-spilling-count (a b)
                    (simd:count (> ,(tree 7 t) b)))))))
  (define-spilling-mask-kernel))

(test extended-unary-and-bounds
  (dolist (backend (available-backends))
    (with-backend (backend)
      (dolist (type '(single-float double-float (signed-byte 8) (unsigned-byte 8)))
        (let* ((floatp (member type '(single-float double-float)))
               (source (make-array 5 :element-type type
                                   :initial-contents
                                   (mapcar (lambda (x) (coerce x type)) '(0 1 2 3 4))))
               (out (make-array 5 :element-type type))
               (two (coerce 2 type)))
          (simd:negate! out source)
          (is (= (aref out 4) (if floatp -4 (integer-reference -4 type))))
          (when floatp (is (minusp (float-sign (aref out 0)))))
          (simd:abs! out out)
          (is (= (aref out 4) (if (equal type '(unsigned-byte 8)) 252 4)))
          (when floatp (is (plusp (float-sign (aref out 0)))))
          (simd:min! out source two)
          (is (equalp out (make-array 5 :element-type type
                                      :initial-contents (mapcar (lambda (x) (coerce x type))
                                                                '(0 1 2 2 2)))))
          (simd:max! out source two)
          (is (= (aref out 0) 2))
          (simd:clamp! out source (coerce 1 type) (coerce 3 type))
          (is (= (aref out 0) 1))
          (is (= (aref out 4) 3))
          (signals error (simd:clamp! out source (coerce 3 type) (coerce 1 type)))
          (if floatp
              (progn
                (simd:sqrt! out source)
                (is (= (aref out 4) 2))
                (simd:reciprocal! out source :start 1)
                (is (= (aref out 2) (coerce 0.5 type))))
              (progn
                (signals error (simd:sqrt! out source))
                (signals error (simd:reciprocal! out source)))))))))

(test extended-conversions
  (let ((types '(single-float double-float
                 (signed-byte 8) (unsigned-byte 8)
                 (signed-byte 16) (unsigned-byte 16)
                 (signed-byte 32) (unsigned-byte 32)
                 (signed-byte 64) (unsigned-byte 64))))
    (dolist (backend (available-backends))
      (with-backend (backend)
        (dolist (source-type types)
          (dolist (destination-type types)
            (let* ((source (make-array 5 :element-type source-type
                                       :initial-contents (loop for i below 5 collect (coerce i source-type))))
                   (destination (make-array 5 :element-type destination-type)))
              (is (eq destination (simd:convert! destination source)))
              (dotimes (i 5)
                (is (= i (aref destination i)))))))
        (let* ((source (make-array 5 :element-type 'double-float
                                      :initial-contents '(-300d0 -2.5d0 1.5d0 2.5d0 300d0)))
               (destination (make-array 5 :element-type '(signed-byte 8))))
          (simd:convert! destination source)
          (is (equalp destination #(-128 -2 2 2 127)))
          (simd:convert! destination source :rounding :truncate)
          (is (equalp destination #(-128 -2 1 2 127)))
          (simd:convert! destination source :rounding :floor)
          (is (equalp destination #(-128 -3 1 2 127)))
          (simd:convert! destination source :rounding :ceiling)
          (is (equalp destination #(-128 -2 2 3 127))))))))

(test extended-signed-minimum-wraps
  (dolist (backend (available-backends))
    (with-backend (backend)
      (let ((value (make-array 1 :element-type '(signed-byte 8)
                                    :initial-element -128)))
        (simd:negate! value value)
        (is (= -128 (aref value 0)))
        (simd:abs! value value)
        (is (= -128 (aref value 0)))))))

#+sbcl
(test extended-nonfinite-conversions
  (let ((infinities (make-array 2 :element-type 'double-float
                                :initial-contents
                                (list (sb-kernel:make-double-float #x7ff00000 0)
                                      (sb-kernel:make-double-float
                                       (- #xfff00000 #x100000000) 0))))
        (nan (make-array 1 :element-type 'double-float
                         :initial-element (sb-kernel:make-double-float #x7ff80000 0)))
        (destination (make-array 2 :element-type '(signed-byte 8))))
    (dolist (backend (available-backends))
      (with-backend (backend)
        (simd:convert! destination infinities)
        (is (equalp destination #(127 -128)))
        (signals error (simd:convert! destination nan :end 1))))))

(test extended-masks-and-kernels
  (dolist (backend (available-backends))
    (with-backend (backend)
      (dolist (type '(single-float double-float (signed-byte 8) (unsigned-byte 64)))
        (let* ((left (make-array 5 :element-type type
                                      :initial-contents (mapcar (lambda (x) (coerce x type))
                                                                '(1 4 3 8 5))))
               (right (make-array 5 :element-type type
                                       :initial-contents (mapcar (lambda (x) (coerce x type))
                                                                 '(2 2 3 4 6))))
               (mask (make-array 5 :element-type '(unsigned-byte 8)))
               (out (make-array 5 :element-type type)))
          (is (eq mask (simd:compare! mask :gt left right)))
          (is (equalp mask #(0 1 0 1 0)))
          (is (= 2 (simd:count mask)))
          (is (simd:any mask))
          (is (not (simd:all mask)))
          (simd:select! out mask left right)
          (is (equalp out (make-array 5 :element-type type
                                      :initial-contents (mapcar (lambda (x) (coerce x type))
                                                                '(2 4 3 8 6)))))
          (mask-kernel-compare mask left right)
          (is (equalp mask #(0 1 0 1 0)))
          (mask-kernel-select out left right)
          (is (= 8 (aref out 3)))
          (is (= 2 (mask-kernel-count left right)))
          (is (mask-kernel-any left right))
          (is (not (mask-kernel-all left right))))))))

(test extended-slices-and-empty-masks
  (dolist (backend (available-backends))
    (with-backend (backend)
      (let ((input (make-array 4 :element-type 'single-float
                                    :initial-contents '(1.0 2.0 3.0 4.0)))
            (out (make-array 4 :element-type 'single-float :initial-element 9.0))
            (mask (make-array 4 :element-type '(unsigned-byte 8) :initial-element 0)))
        (simd:negate! out input :start 0 :end 2 :input-start 2 :destination-start 1)
        (is (equalp out #(9.0 -3.0 -4.0 9.0)))
        (simd:compare! mask :lt input 3.0 :start 0 :end 2 :left-start 1 :mask-start 2)
        (is (equalp mask #(0 0 1 0)))
        (is (= 0 (simd:count mask :end 0)))
        (is (not (simd:any mask :end 0)))
        (is (simd:all mask :end 0))
        (signals error (simd:convert! out (make-array 2 :element-type 'double-float)))
        (signals error (simd:compare! mask :gt input input :left-start 4))
        (signals error (simd:select! out mask input input :end 4 :mask-start 1))
        (signals error (simd:clamp! out input 3.0 1.0 :end 0))))))

(test extended-large-integers-and-mask-values
  (dolist (backend (available-backends))
    (with-backend (backend)
      (let* ((small (+ (ash 1 60) 1))
             (large (+ (ash 1 60) 2))
             (a (make-array 2 :element-type '(unsigned-byte 64)
                              :initial-contents (list small large)))
             (b (make-array 2 :element-type '(unsigned-byte 64)
                              :initial-contents (list large small)))
             (mask (make-array 2 :element-type '(unsigned-byte 8)))
             (out (make-array 2 :element-type '(unsigned-byte 64)))
             (narrow (make-array 2 :element-type '(unsigned-byte 8))))
        (simd:compare! mask :lt a b)
        (is (equalp mask #(1 0)))
        (simd:compare! mask :ge a b)
        (is (equalp mask #(0 1)))
        (setf (aref mask 0) 255)
        (setf (aref mask 1) 0)
        (simd:select! out mask a b)
        (is (= small (aref out 0)))
        (is (= small (aref out 1)))
        (simd:convert! narrow a)
        (is (equalp narrow #(255 255)))))))

(test extended-comparison-operators
  (dolist (backend (available-backends))
    (with-backend (backend)
      (dolist (type '(single-float (signed-byte 16)))
        (let ((a (make-array 3 :element-type type
                                    :initial-contents (mapcar (lambda (x) (coerce x type))
                                                              '(1 2 3))))
              (b (make-array 3 :element-type type
                                    :initial-contents (mapcar (lambda (x) (coerce x type))
                                                              '(2 2 2))))
              (mask (make-array 3 :element-type '(unsigned-byte 8))))
          (dolist (case '((:eq #(0 1 0)) (:ne #(1 0 1))
                          (:lt #(1 0 0)) (:le #(1 1 0))
                          (:gt #(0 0 1)) (:ge #(0 1 1))))
            (simd:compare! mask (first case) a b)
            (is (equalp mask (second case)))))))))

(test nested-mask-kernels-and-empty-reductions
  (dolist (backend (available-backends))
    (with-backend (backend)
      (dolist (type '(single-float double-float (signed-byte 16)))
        (let* ((a (make-array 259 :element-type type
                                   :initial-contents (loop for i below 259
                                                           collect (coerce (mod i 11) type))))
               (b (make-array 259 :element-type type
                                   :initial-contents (loop for i below 259
                                                           collect (coerce (mod (+ i 4) 9) type))))
               (c (make-array 259 :element-type type
                                   :initial-contents (loop for i below 259
                                                           collect (coerce (mod (+ i 2) 7) type))))
               (out (make-array 259 :element-type type))
               (count 0))
          (mask-kernel-nested out a b c)
          (dotimes (i 259)
            (let* ((x (aref a i)) (y (aref b i)) (z (aref c i))
                   (expected (if (> x y)
                                 (if (< z x) (+ x z) (* x y))
                                 (- z y))))
              (is (= expected (aref out i)))
              (when (> (if (> x y) x z) y) (incf count))))
          (is (= count (mask-kernel-nested-count a b c)))
          (is (= (loop for x across a for y across b sum (max x y))
                 (mask-kernel-sum-select a b)))
          (mask-kernel-positive out a)
          (is (= (aref a 258) (aref out 258)))
          (is (= 0 (mask-kernel-count a b :end 0)))
          (is (not (mask-kernel-any a b :end 0)))
          (is (mask-kernel-all a b :end 0)))))))

(test spilling-mask-kernels
  (dolist (backend (available-backends))
    (with-backend (backend)
      (let ((a (values-for 513 'single-float 1))
            (b (make-array 513 :element-type 'single-float :initial-element 256.0))
            (out (make-array 513 :element-type 'single-float)))
        (mask-kernel-spilling out a b)
        (dotimes (i 513)
          (is (= (max (aref a i) (aref b i)) (aref out i))))
        (is (= 257 (mask-kernel-spilling-count a b)))))))
