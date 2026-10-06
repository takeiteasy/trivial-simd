(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defun overlap-vector (type length)
  (make-array length :element-type type
                     :initial-contents (loop for i below length collect (coerce (1+ (mod i 7)) type))))

(defun check-bulk-overlap (function type length mode)
  (with-view-memory
    (let* ((original (overlap-vector type length))
           (expected (copy-seq original))
           (extra (overlap-vector type length))
           (destination (if (eq mode :views) (mirror original) (copy-seq original)))
           (source (if (simd:vector-view-p destination)
                       (simd:make-vector-view (simd:vector-view-pointer destination)
                                              (simd:vector-view-type destination) length)
                       destination)))
      (funcall function expected (copy-seq original) extra)
      (is (eq destination (funcall function destination source
                                  (if (eq mode :mixed) (mirror extra) extra))))
      (is (numerically-close-p (list destination) (list expected))))))

(test bulk-overlap-types-slices-and-strides
  (dolist (type *copy-types*)
    (dolist (count '(0 1 17 257 4101))
      (dolist (mapping (list '(1 0 1 1) '(0 1 1 1) '(0 0 1 1) '(1 0 2 2) '(0 1 2 2)
                            (list 0 (max 0 (1- count)) 1 -1)
                            (list (max 0 (1- count)) 0 -1 1)
                            (list (* 2 count) (max 0 (- (* 2 count) 2)) -2 -2)))
        (destructuring-bind (d-offset i-offset d-stride i-stride) mapping
          (dolist (backend (available-backends))
            (with-backend (backend)
              (dolist (mode '(:arrays :views :mixed))
                (check-bulk-overlap
                 (lambda (destination input extra)
                   (simd:add! destination input extra :end count
                              :destination-start d-offset :left-start i-offset
                              :destination-stride d-stride :left-stride i-stride))
                 type (+ 2 (* 2 count)) mode)))))))))

(test bulk-overlap-operation-families
  (dolist (backend (available-backends))
    (with-backend (backend)
      (dolist (mode '(:arrays :views :mixed))
        (dolist (type '(single-float double-float (complex single-float)
                       (complex double-float) (unsigned-byte 8)))
          (dolist (operation '(simd:subtract! simd:multiply! simd:divide!))
            (check-bulk-overlap
             (lambda (destination input extra)
               (funcall operation destination input extra :end 257 :destination-start 1))
             type 260 mode))
          (check-bulk-overlap
           (lambda (destination input extra)
             (declare (ignore extra))
             (simd:add! destination input (coerce 2 type) :end 257 :destination-start 1))
           type 260 mode)
          (check-bulk-overlap
           (lambda (destination input extra)
             (declare (ignore extra))
             (simd:axpy! destination (coerce 2 type) input :end 257 :y-start 1))
           type 260 mode)
          (dolist (operation '(simd:negate! simd:copy! simd:convert!))
            (check-bulk-overlap
             (lambda (destination input extra)
               (declare (ignore extra))
               (funcall operation destination input :end 257 :destination-start 1))
             type 260 mode)))
        (dolist (operation '(simd:abs! simd:sqrt! simd:reciprocal!))
          (check-bulk-overlap
           (lambda (destination input extra)
             (declare (ignore extra))
             (funcall operation destination input :end 257 :destination-start 1))
           'single-float 260 mode))
        (check-bulk-overlap
         (lambda (destination input extra)
           (simd:fma! destination input extra 1f0 :end 257 :destination-start 1))
         'single-float 260 mode)
        (dolist (operation '(simd:min! simd:max!))
          (check-bulk-overlap
           (lambda (destination input extra)
             (funcall operation destination input extra :end 257 :destination-start 1))
           'single-float 260 mode))
        (check-bulk-overlap
         (lambda (destination input extra)
           (declare (ignore extra))
           (simd:clamp! destination input 2f0 5f0 :end 257 :destination-start 1))
         'single-float 260 mode)
        (check-bulk-overlap
         (lambda (destination input extra)
           (simd:compare! destination :eq input extra :end 257 :mask-start 1))
         '(unsigned-byte 8) 260 mode)
        (check-bulk-overlap
         (lambda (destination input extra)
           (simd:select! destination input extra 0 :end 257 :destination-start 1))
         '(unsigned-byte 8) 260 mode)))))

(test bulk-overlap-opposing-inputs
  (dolist (backend (available-backends))
    (with-backend (backend)
      (dolist (viewp '(nil t))
        (with-view-memory
          (let* ((original (overlap-vector 'single-float 4104))
                 (expected (copy-seq original))
                 (actual (if viewp (mirror original) (copy-seq original))))
            (simd:subtract! expected (copy-seq original) (copy-seq original)
                            :end 4101 :destination-start 1 :right-start 2)
            (simd:subtract! actual actual actual :end 4101 :destination-start 1 :right-start 2)
            (is (numerically-close-p (list actual) (list expected)))))))))

(test bulk-overlap-mixed-strided-blocks
  (let ((simd::*view-block-size* 5))
    (dolist (backend (available-backends))
      (with-backend (backend)
        (check-bulk-overlap
         (lambda (destination input extra)
           (simd:add! destination input extra :end 257 :destination-start 2
                      :destination-stride 2 :left-stride 2))
         'double-float 520 :mixed)))))

(test bulk-overlap-mixed-view-element-sizes
  (dolist (backend (available-backends))
    (with-backend (backend)
      (with-view-memory
        (let* ((pointer (view-memory (* 8 4102)))
               (input (simd:make-vector-view pointer :f32 4102))
               (output (simd:make-vector-view pointer :f64 4101))
               (original (overlap-vector 'single-float 4102))
               (expected (make-array 4101 :element-type 'double-float)))
          (simd::stage-out original input 0 1 4102)
          (simd:convert! expected original :end 4101)
          (simd:convert! output input :end 4101)
          (is (equalp expected (subseq (contents output) 0 4101))))))))

(test bulk-overlap-mixed-view-masks-and-encodings
  (dolist (backend (available-backends))
    (with-backend (backend)
      (dolist (operation '(:compare :select :encode :decode :magnitude))
        (with-view-memory
          (let* ((pointer (view-memory (* 16 260)))
                 (source-type (case operation (:select :u8) (:decode :u16)
                                    (:magnitude :c64) (t :f32)))
                 (target-type (case operation (:compare :u8) (:encode :u16)
                                    (:magnitude :f64) (t :f32)))
                 (input (simd:make-vector-view pointer source-type 260))
                 (output (simd:make-vector-view pointer target-type 257 :offset 1))
                 (source (if (eq operation :decode)
                             (make-array 260 :element-type '(unsigned-byte 16)
                                             :initial-element #x3c00)
                             (overlap-vector (second (simd::numeric-type source-type)) 260)))
                 (expected (make-array 257 :element-type (second (simd::numeric-type target-type)))))
            (simd::stage-out source input 0 1 260)
            (flet ((execute (destination operand)
                     (ecase operation
                       (:compare (simd:compare! destination :gt operand 4f0 :end 257))
                       (:select (simd:select! destination operand 2f0 0f0 :end 257))
                       (:encode (simd:convert! destination operand :end 257 :destination-encoding :f16))
                       (:decode (simd:convert! destination operand :end 257 :input-encoding :f16))
                       (:magnitude (simd:abs! destination operand :end 257)))))
              (execute expected source)
              (execute output input))
            (is (numerically-close-p (list output) (list expected)))))))))

(test swap-overlap-validation
  (dolist (type *copy-types*)
    (dolist (backend (available-backends))
      (with-backend (backend)
        (dolist (viewp '(nil t))
          (with-view-memory
            (let* ((original (overlap-vector type 520))
                   (x (if viewp (mirror original) (copy-seq original)))
                   (y (if viewp
                          (simd:make-vector-view (simd:vector-view-pointer x)
                                                 (simd:vector-view-type x) 520)
                          x)))
              (signals error (simd:swap! x y :end 257 :y-start 1))
              (signals error (simd:swap! x y :end 257 :x-start 256 :x-stride -1))
              (is (equalp (contents x) original))
              (multiple-value-bind (first second) (simd:swap! x y :end 257 :stride -1
                                                              :x-start 256 :y-start 256)
                (is (eq first x))
                (is (eq second y)))
              (is (equalp (contents x) original))
              (simd:swap! x y :end 257 :stride 2 :y-start 1)
              (let ((expected (copy-seq original)))
                (dotimes (i 257)
                  (rotatef (aref expected (* 2 i)) (aref expected (1+ (* 2 i)))))
                (is (equalp (contents x) expected))))))))))
