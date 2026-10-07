(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defun gemm-addresses (dimensions steps &optional (offset 0))
  (loop for row below (aref dimensions 0) append
        (loop for col below (aref dimensions 1) append
              (loop for batch below (aref dimensions 2)
                    collect (+ offset (* row (aref steps 0))
                               (* col (aref steps 1)) (* batch (aref steps 2)))))))

(test gemm-arithmetic-uniqueness-exhaustive
  (loop for rows from 0 to 3 do
    (loop for cols from 0 to 3 do
      (loop for count from 0 to 3 do
        (loop for rs from -3 to 3 do
          (loop for cs from -3 to 3 do
            (loop for stride from -3 to 3
                  for dimensions = (vector rows cols count)
                  for steps = (vector rs cs stride)
                  for addresses = (gemm-addresses dimensions steps)
                  do (is (eq (= (length addresses) (length (remove-duplicates addresses)))
                             (trivial-simd/blas::gemm-layout-unique-p dimensions steps))))))))))

(test gemm-arithmetic-overlap-exhaustive
  (let* ((data (make-array 64 :element-type 'single-float :initial-element 0f0))
         (layouts (loop for rs from -2 to 2 append
                    (loop for cs from -2 to 2 append
                      (loop for stride from -2 to 2 collect (vector rs cs stride))))))
    (dolist (as layouts)
      (let ((a (trivial-simd/blas:make-matrix-view data 2 2 :offset 24
                                                  :row-stride (aref as 0) :column-stride (aref as 1)))
            (aa (gemm-addresses #(2 2 2) as 24)))
        (dolist (bs layouts)
          (loop for shift from -4 to 4
                for b = (trivial-simd/blas:make-matrix-view
                         data 2 2 :offset (+ 24 shift)
                         :row-stride (aref bs 0) :column-stride (aref bs 1))
                for ba = (gemm-addresses #(2 2 2) bs (+ 24 shift))
                do (is (eq (not (null (intersection aa ba)))
                           (not (null (trivial-simd/blas::matrix-elements-overlap-p
                                       a 2 (aref as 2) b 2 (aref bs 2))))))))))))

(test gemm-arithmetic-seeded-equations
  (let ((*blas-random-seed* 149))
    (dotimes (trial 1000)
      (let* ((rank (1+ (blas-random 6)))
             (steps (map 'vector (lambda (x) (declare (ignore x)) (- (blas-random 15) 7))
                         (make-array rank)))
             (lower (map 'vector (lambda (x) (declare (ignore x)) (- (blas-random 3))) steps))
             (upper (map 'vector (lambda (low) (+ low (blas-random 3))) lower))
             (target (- (blas-random 41) 20)))
        (labels ((enumerate (axis sum)
                   (if (= axis rank)
                       (= sum target)
                       (loop for value from (aref lower axis) to (aref upper axis)
                             thereis (enumerate (1+ axis) (+ sum (* value (aref steps axis))))))))
          (is (eq (enumerate 0 0)
                  (trivial-simd/blas::bounded-affine-solution-p steps lower upper target)))))))
  (is (trivial-simd/blas::gemm-layout-unique-p
       #(2 1000000 1000000) #(1 2000002 2000006)))
  (is (not (trivial-simd/blas::gemm-layout-unique-p
            #(2 1000004 1000002) #(1 2000002 2000006))))
  (is (trivial-simd/blas::bounded-affine-solution-p
       #(1000000007 1000000009) #(0 0) #(1000000000 1000000000) 1000000007))
  (is (not (trivial-simd/blas::bounded-affine-solution-p #(0 2) #(1 0) #(0 3) 2)))
  (let* ((base (ash 1 60))
         (steps (vector (1- base) (- base 3) (- base 7))))
    (is (trivial-simd/blas::bounded-affine-solution-p steps #(0 0 0) #(3 3 3)
                                                        (* 2 (aref steps 0))))
    (is (not (trivial-simd/blas::bounded-affine-solution-p steps #(0 0 0) #(3 3 3)
                                                              (- base 2))))))

(test gemm-interleaved-shared-storage
  (dolist (case '((single-float "s") (double-float "d")
                  ((complex single-float) "c") ((complex double-float) "z")))
    (destructuring-bind (type prefix) case
      (let* ((data (batch-data type (loop repeat 32 collect 7)))
             (a (trivial-simd/blas:make-matrix-view data 3 2 :row-stride 4 :column-stride 6))
             (c (trivial-simd/blas:make-matrix-view data 3 2 :offset 1 :row-stride 4 :column-stride 6))
             (identity (level3-view type '(1 0 0 1)))
             (one (level3-number type 1)) (zero (level3-number type 0))
             (name (concatenate 'string prefix "gemm")))
        (dotimes (row 3)
          (dotimes (col 2)
            (setf (trivial-simd/blas:matrix-ref a row col)
                  (level3-number type (+ 1 (* row 2) col) (- row col)))))
        (is (eq c (level3-call name :no-transpose :no-transpose one a identity zero c)))
        (dotimes (row 3)
          (dotimes (col 2)
            (is (= (trivial-simd/blas:matrix-ref a row col)
                   (trivial-simd/blas:matrix-ref c row col)))))
        (is (= 7 (aref data 31)))
        (let ((before (copy-seq data)))
          (signals error (level3-call name :no-transpose :no-transpose one a identity zero a))
          (is (equalp before data)))))))

(test gemm-interleaved-foreign-validation
  (dolist (type '(single-float double-float))
    (let* ((dtype (if (eq type 'single-float) :f32 :f64))
           (size (if (eq type 'single-float) 4 8))
           (name (if (eq type 'single-float) "sgemm-batch-strided" "dgemm-batch-strided"))
           (one (coerce 1 type)) (zero (coerce 0 type)))
      (cffi:with-foreign-object (pointer :uint8 256)
        (let* ((storage (trivial-simd:make-vector-view pointer dtype 32))
               (other (trivial-simd:make-vector-view (cffi:inc-pointer pointer size) dtype 31))
               (partial (trivial-simd:make-vector-view (cffi:inc-pointer pointer 1) dtype 31))
               (a (trivial-simd/blas:make-matrix-view storage 2 2 :row-stride 8 :column-stride 4))
               (c (trivial-simd/blas:make-matrix-view other 2 2 :row-stride 8 :column-stride 4))
               (bad (trivial-simd/blas:make-matrix-view partial 2 2 :row-stride 8 :column-stride 4))
               (identity (level3-view type '(1 0 0 1))))
          (dotimes (i 32) (setf (trivial-simd::vector-ref storage i) (coerce (mod i 7) type)))
          (check-batch-gemm type a identity c 2 2 0 2)
          (let ((before (loop for i below 256 collect (cffi:mem-aref pointer :uint8 i))))
            (signals error (level3-call name :no-transpose :no-transpose one a identity zero bad
                                       :batch-count 2 :a-stride 2 :b-stride 0 :c-stride 2))
            (signals error (level3-call name :no-transpose :no-transpose one a identity zero c
                                       :batch-count 2 :a-stride 2 :b-stride 0 :c-stride 3))
            (is (equal before (loop for i below 256 collect (cffi:mem-aref pointer :uint8 i)))))
          (is (not (trivial-simd/blas::matrix-elements-overlap-p a 0 0 c 2 2)))
          (is (not (trivial-simd/blas::matrix-elements-overlap-p a 2 2 c 0 0))))))))
