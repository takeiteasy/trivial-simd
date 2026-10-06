(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(simd:define-kernel batched-minimum (x) (simd:minimum x))
(simd:define-kernel batched-maximum (x) (simd:maximum x))
(simd:define-kernel batched-argmin (x) (simd:argmin x))
(simd:define-kernel batched-argmax (x) (simd:argmax x))
(simd:define-kernel batched-product (x) (simd:prod x))
(simd:define-kernel dependent-product (x) (simd:prod (- x (simd:minimum x))))

(test product-types-and-strides
  (dolist (backend (available-backends))
    (dolist (type (append *real-types* '((complex single-float) (complex double-float))))
      (let ((x (make-array 7 :element-type type :initial-contents
                           (mapcar (lambda (v) (coerce v type)) '(2 1 3 1 4 1 1)))))
        (with-backend (backend)
          (is (= 24 (simd:prod x)))
          (is (= 24 (simd:prod x :end 3 :stride 2)))
          (is (= 24 (simd:prod x :input-start 4 :end 3 :stride -2)))
          (is (= 1 (simd:prod x :end 0)))
          (signals error (simd:prod x :end 3 :stride 0))
          (signals error (simd:prod x :end 8))
          (signals error (simd:prod x :input-start 0 :stride -1 :end 2))))))
  (is (= -16 (simd:prod (make-array 2 :element-type '(signed-byte 8) :initial-contents '(20 12))))))

(test row-reducers-types-and-layouts
  (dolist (backend (available-backends))
    (dolist (type *real-types*)
      (let ((x (make-array 12 :element-type type :initial-contents
                             (mapcar (lambda (v) (coerce v type)) '(9 3 1 2 9 9 4 6 5 9 9 9))))
            (out (make-array 4 :element-type type :initial-element (coerce 9 type)))
            (indices (make-array 4 :element-type '(signed-byte 64) :initial-element -1)))
        (with-backend (backend)
          (dolist (spec (list (list #'batched-minimum out '(1 4))
                             (list #'batched-maximum out '(3 6))
                             (list #'batched-argmin indices '(1 0))
                             (list #'batched-argmax indices '(0 1))
                             (list #'batched-product out '(6 120))))
            (destructuring-bind (function destination expected) spec
              (is (eq destination (funcall function x :rows 2 :row-length 3 :x-start 1 :x-row-stride 5
                                                    :destination destination :destination-start 1)))
              (is (equalp expected (coerce (subseq destination 1 3) 'list)))
              (funcall function x :rows 2 :row-length 3 :x-start 6 :x-row-stride -5 :destination destination)
              (is (equalp (reverse expected) (coerce (subseq destination 0 2) 'list)))
              (funcall function x :rows 2 :row-length 3 :x-start 1 :x-row-stride 0 :destination destination)
              (is (= (first expected) (aref destination 0) (aref destination 1)))))
          (is (= 0 (dependent-product x :start 1 :end 4)))
          (signals error (batched-minimum x :rows 2 :row-length 0 :destination out))
          (signals error (batched-argmin x :rows 2 :row-length 3 :destination (make-array 2 :element-type 'single-float)))
          (signals error (batched-minimum x :rows 4 :row-length 3 :x-start 1 :destination out))
          (signals error (batched-product x :rows 2 :row-length 3 :destination x))
          (batched-product x :rows 2 :row-length 0 :destination out)
          (is (= 1 (aref out 0) (aref out 1)))
          (batched-minimum x :rows 0 :row-length 0 :destination out))))))

(test complex-product-rows
  (dolist (backend (available-backends))
    (dolist (type '((complex single-float) (complex double-float)))
      (let ((x (make-array 4 :element-type type :initial-element (coerce #c(1 1) type)))
            (out (make-array 2 :element-type type)))
        (with-backend (backend)
          (batched-product x :rows 2 :row-length 2 :destination out)
          (is (= #c(0 2) (aref out 0) (aref out 1))))))))

(test native-row-reducers-dispatch-and-fallback
  (when (and simd::*native-row-reducers-p* (member :native (available-backends)))
    (let ((original (symbol-function 'simd::call-native-reduction-rows)) (calls 0)
          (x (make-array 6 :element-type 'single-float :initial-contents '(3f0 1f0 2f0 4f0 6f0 5f0)))
          (out (make-array 2 :element-type '(signed-byte 64))))
      (unwind-protect
           (progn
             (setf (symbol-function 'simd::call-native-reduction-rows)
                   (lambda (&rest arguments) (incf calls) (apply original arguments)))
             (with-backend (:native)
               (batched-argmax x :rows 2 :row-length 3 :destination out)
               (is (= 1 calls))
               (is (equalp #(0 1) out))
               (let ((simd::*native-row-reducers-p* nil))
                 (batched-argmax x :rows 2 :row-length 3 :destination out)
                 (is (equalp #(0 1) out))
                 (is (= 1 calls))
                 (is (= 720 (simd:prod x))))))
        (setf (symbol-function 'simd::call-native-reduction-rows) original)))))

(test foreign-row-reducer-output
  (cffi:with-foreign-objects ((input :float 6) (output :int64 2))
    (loop for value in '(3f0 1f0 2f0 4f0 6f0 5f0) for i from 0
          do (setf (cffi:mem-aref input :float i) value))
    (dolist (backend (available-backends))
      (with-backend (backend)
        (batched-argmax (simd:make-vector-view input :f32 6) :rows 2 :row-length 3
                       :destination (simd:make-vector-view output :s64 2))
        (is (= 0 (cffi:mem-aref output :int64 0)))
        (is (= 1 (cffi:mem-aref output :int64 1)))))))

(simd:define-kernel declared-row-arg ((x :type :s8) scale) (simd:argmax (* x scale)))
(simd:define-kernel declared-row-product ((x :type :s8) scale) (simd:prod (* x scale)))
(simd:define-kernel planned-row-arg (x) (simd:argmin (- x (simd:minimum x))))
(simd:define-kernel selected-row-product (x) (simd:prod (simd:select (> x 1) x 1)))

(test declared-and-planned-row-reducers
  (dolist (backend (available-backends))
    (let ((x (make-array 6 :element-type '(signed-byte 8) :initial-contents '(3 1 2 4 6 5)))
          (floats (make-array 6 :element-type 'single-float :initial-contents '(3f0 1f0 2f0 4f0 6f0 5f0)))
          (scale (make-array 3 :element-type 'single-float :initial-element 1f0))
          (indices (make-array 2 :element-type '(signed-byte 64)))
          (out (make-array 2 :element-type 'single-float)))
      (with-backend (backend)
        (declared-row-arg x scale :rows 2 :row-length 3 :scale-row-stride 0 :destination indices)
        (is (equalp #(0 1) indices))
        (declared-row-product x scale :rows 2 :row-length 3 :scale-row-stride 0 :destination out)
        (is (equalp #(6 120) out))
        (planned-row-arg floats :rows 2 :row-length 3 :destination indices)
        (is (equalp #(1 0) indices))
        (selected-row-product floats :rows 2 :row-length 3 :destination out)
        (is (equalp #(6 120) out))))))

(test product-foreign-staging-blocks
  (cffi:with-foreign-object (pointer :float 771)
    (dotimes (i 771) (setf (cffi:mem-aref pointer :float i) 1f0))
    (setf (cffi:mem-aref pointer :float 2) 2f0
          (cffi:mem-aref pointer :float 300) 3f0
          (cffi:mem-aref pointer :float 770) 4f0)
    (dolist (backend (available-backends))
      (with-backend (backend)
        (let ((simd::*view-block-size* 5))
          (is (= 24 (simd:prod (simd:make-vector-view pointer :f32 771)))))))))

(simd:define-kernel complex-selected-product (x) (simd:prod (simd:select (= x 1) x x)))

(test complex-product-foreign-storage
  (dolist (dtype '(:c32 :c64))
    (let ((foreign (if (eq dtype :c32) :float :double)))
      (cffi:with-foreign-object (pointer foreign 1028)
        (dotimes (i 514)
          (setf (cffi:mem-aref pointer foreign (* 2 i)) (if (eq dtype :c32) 1f0 1d0)
                (cffi:mem-aref pointer foreign (1+ (* 2 i))) (if (eq dtype :c32) 0f0 0d0)))
        (dolist (i '(1 300))
          (setf (cffi:mem-aref pointer foreign (1+ (* 2 i))) (if (eq dtype :c32) 1f0 1d0)))
        (let ((view (simd:make-vector-view pointer dtype 514)))
          (dolist (backend (available-backends))
            (with-backend (backend)
              (let ((simd::*view-block-size* 5)
                    (output (make-array 2 :element-type (if (eq dtype :c32) '(complex single-float)
                                                           '(complex double-float)))))
                (is (= #c(0 2) (simd:prod view)))
                (is (= #c(0 2) (complex-selected-product view)))
                (batched-product view :rows 2 :row-length 257 :destination output)
                (is (= #c(1 1) (aref output 0) (aref output 1)))))))))))

(simd:define-kernel nested-product-sum (x) (simd:sum (- x (simd:prod x))))

(test nested-product-reductions
  (dolist (backend (available-backends))
    (with-backend (backend)
      (let ((x (make-array 4 :element-type 'single-float :initial-contents '(1f0 2f0 3f0 4f0)))
            (out (make-array 2 :element-type 'single-float)))
        (is (= -86 (nested-product-sum x)))
        (nested-product-sum x :rows 2 :row-length 2 :destination out)
        (is (equalp #(-1 -17) out))))))
