(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defun conversion-boundary-input (type)
  (let* ((element (second (simd::numeric-type type)))
         (seeds (append '(-65537 -32769 -32768 -129 -128 -1 0 1 127 128 255 256
                          32767 32768 65535 65536 65537)
                        (loop for power in '(24 31 32 53 62 63 64)
                              append (list (1- (ash 1 power)) (ash 1 power) (1+ (ash 1 power))))
                        (list (+ (ash 1 62) (ash 1 38) -1)
                              (+ (ash 1 62) (ash 1 38) 1))))
         (values (if (simd::integer-type-p type)
                     (multiple-value-bind (low high) (simd::integer-limits type)
                       (mapcar (lambda (x) (min high (max low x))) seeds))
                     (append (mapcar (lambda (x) (coerce x element)) seeds)
                             (mapcar (lambda (x) (coerce x element)) '(-5/2 -3/2 -1/2 1/2 3/2 5/2))))))
    (make-array 263 :element-type element
                    :initial-contents (loop for i below 263 collect (nth (mod i (length values)) values)))))

(defun conversion-reference (value type rounding)
  (if (simd::integer-type-p type)
      (multiple-value-bind (low high) (simd::integer-limits type)
        (min high (max low (funcall (ecase rounding
                                    (:nearest-even #'round) (:truncate #'truncate)
                                    (:floor #'floor) (:ceiling #'ceiling))
                                  (if (floatp value) (rational value) value)))))
      (coerce value (second (simd::numeric-type type)))))

(test conversion-boundaries-and-native-fallback
  (dolist (backend (available-backends))
    (with-backend (backend)
      (dolist (mode '(:pointer :copy))
        (let ((simd::*native-array-access* mode) (simd::*view-block-size* 5))
          (dolist (source-type (mapcar #'first simd::*numeric-types*))
            (let ((source (conversion-boundary-input source-type)))
              (dolist (destination-type (mapcar #'first simd::*numeric-types*))
                (dolist (rounding '(:nearest-even :truncate :floor :ceiling))
                  (let* ((element (second (simd::numeric-type destination-type)))
                         (out (make-array 263 :element-type element :initial-element (coerce 7 element)))
                         (expected (copy-seq out)))
                    (dotimes (i 257)
                      (setf (aref expected (+ 3 i))
                            (conversion-reference (aref source (+ 1 i)) destination-type rounding)))
                    (simd:convert! out source :end 257 :input-start 1 :destination-start 3 :rounding rounding)
                    (is (equalp out expected))
                    (with-view-memory
                      (let ((input-view (mirror source)) (output-view (mirror (make-array 263 :element-type element :initial-element (coerce 7 element)))))
                        (simd:convert! output-view input-view :end 257 :input-start 1 :destination-start 3 :rounding rounding)
                        (is (equalp (contents output-view) expected)))))))))))))
  (let* ((source (conversion-boundary-input :s32))
         (out (make-array 263 :element-type 'single-float))
         (simd::*native-available-p* nil)
         (simd::*backend* :native))
    (simd:convert! out source)
    (is (every #'= out (map 'vector (lambda (x) (coerce x 'single-float)) source)))))

#+(and sbcl x86-64)
(test sbcl-packed-mask-branches-preserve-errors
  (simd:define-kernel packed-safe-select (a b) (simd:select (> a b) a (/ a b)))
  (simd:define-kernel packed-safe-any (a b) (simd:any (> (/ a b) 1)))
  (let ((a (make-array 19 :element-type '(signed-byte 8) :initial-element 2))
        (b (make-array 19 :element-type '(signed-byte 8) :initial-element 0))
        (out (make-array 19 :element-type '(signed-byte 8))))
    (with-backend (:sbcl)
      (packed-safe-select out a b)
      (is (every (lambda (x) (= x 2)) out))
      (setf (aref b 0) 1)
      (is (packed-safe-any a b)))))

(test large-numeric-conversion-slices
  (dolist (backend (available-backends))
    (with-backend (backend)
      (dolist (source-type (mapcar #'first simd::*numeric-types*))
        (when (simd::integer-type-p source-type)
          (let* ((small (conversion-boundary-input source-type))
                 (source (make-array 32775 :element-type (array-element-type small)
                                           :initial-contents (loop for i below 32775 collect (aref small (mod i 263))))))
            (dolist (destination-type '(:f32 :f64 :s8 :u8 :s16 :u16))
              (let* ((element (second (simd::numeric-type destination-type)))
                     (out (make-array 32775 :element-type element :initial-element (coerce 7 element)))
                     (expected (copy-seq out)))
                (dotimes (i 32769)
                  (setf (aref expected (+ 3 i))
                        (conversion-reference (aref source (+ 1 i)) destination-type :nearest-even)))
                (dolist (mode '(:pointer :copy))
                  (let ((simd::*native-array-access* mode))
                    (fill out (coerce 7 element))
                    (simd:convert! out source :end 32769 :input-start 1 :destination-start 3)
                    (is (equalp out expected))))))))))))

(test spilling-mask-call-isolation
  (when simd::*native-available-p*
    (dolist (type '(single-float double-float (signed-byte 8) (unsigned-byte 64)))
      (let* ((a (make-array 1025 :element-type type :initial-element (coerce 2 type)))
             (b (make-array 1025 :element-type type :initial-element (coerce 1 type)))
             (workers (loop repeat 4 collect
                            (bordeaux-threads:make-thread
                             (lambda ()
                               (let ((simd::*backend* :native))
                                 (loop repeat 10 always (= 1025 (mask-kernel-spilling-count a b)))))))))
        (dolist (worker workers) (is-true (bordeaux-threads:join-thread worker)))))))

#+(and sbcl x86-64)
(test sbcl-mask-unselected-float-overflow
  (simd:define-kernel mask-safe-float (a b) (simd:select (> a b) a (* a a)))
  (let ((a (make-array 19 :element-type 'single-float :initial-element most-positive-single-float))
        (b (make-array 19 :element-type 'single-float :initial-element 0f0))
        (out (make-array 19 :element-type 'single-float)))
    (with-backend (:sbcl)
      (mask-safe-float out a b)
      (is (equalp out a)))))

(test packed-bulk-mask-boundaries
  (dolist (backend (available-backends))
    (with-backend (backend)
      (dolist (entry simd::*numeric-types*)
        (destructuring-bind (type element foreign bits signed) entry
          (declare (ignore type foreign bits))
          (let ((a (make-array 263 :element-type element
                                  :initial-contents (loop for i below 263 collect
                                                         (coerce (- (mod i 31) (if signed 15 0)) element))))
                (b (make-array 263 :element-type element
                                  :initial-contents (loop for i below 263 collect
                                                         (coerce (- (mod (* i 7) 29) (if signed 14 0)) element)))))
            (dolist (n '(0 1 15 16 17 19 255 256 257 259))
              (dolist (access (if (eq backend :native) '(:pointer :copy) '(:pointer)))
                (let* ((simd::*native-array-access* access)
                       (mask (make-array (+ n 4) :element-type '(unsigned-byte 8) :initial-element 7))
                       (out (make-array (+ n 4) :element-type element :initial-element (coerce 7 element))))
                  (dolist (scalar '(nil t))
                    (let ((left (if scalar (aref a 0) a)))
                      (dolist (op '(:eq :ne :lt :le :gt :ge))
                        (let ((expected (make-array (+ n 4) :element-type '(unsigned-byte 8) :initial-element 7)))
                          (dotimes (i n)
                            (setf (aref expected (+ 1 i))
                                  (if (simd::compare-values op (if scalar left (aref a (+ 1 i)))
                                                           (aref b (+ 2 i))) 1 0)))
                          (fill mask 7)
                          (simd:compare! mask op left b :end n :mask-start 1
                                         :left-start (unless scalar 1) :right-start 2)
                          (is (equalp mask expected))))
                      (let ((expected (copy-seq out)))
                        (fill out (coerce 7 element))
                        (fill expected (coerce 7 element))
                        (dotimes (i n)
                          (setf (aref mask (+ 1 i)) (if (zerop (mod i 4)) 0 (1+ (mod i 255)))
                                (aref expected (+ 2 i))
                                (if (zerop (aref mask (+ 1 i))) (aref b (+ 2 i))
                                    (if scalar left (aref a (+ 1 i))))))
                        (simd:select! out mask left b :end n :destination-start 2 :mask-start 1
                                      :true-start (unless scalar 1) :false-start 2)
                        (is (equalp out expected))
                        (let ((total (loop for i below n count (not (zerop (aref mask (+ 1 i)))))))
                          (is (= total (simd:count mask :end n :mask-start 1)))
                          (is (eq (plusp total) (simd:any mask :end n :mask-start 1)))
                          (is (eq (= total n) (simd:all mask :end n :mask-start 1))))))))))))))))

#+(and sbcl x86-64)
(test sbcl-packed-selection-preserves-bits
  (with-backend (:sbcl)
    (dolist (entry '((single-float bits-f32 f32-bits
                     (#x80000000 #x7f800123 #xffc00045 #x00000001 #x3f800000)
                     (#x00000000 #x7fc06789 #xff800123 #x00800000 #xbf800000))
                    (double-float bits-f64 f64-bits
                     (#x8000000000000000 #x7ff0000000000123 #xfff8000000000045 #x1 #x3ff0000000000000)
                     (#x0 #x7ff8000000006789 #xfff0000000000123 #x0010000000000000 #xbff0000000000000))))
      (destructuring-bind (element decode encode left-bits right-bits) entry
        (let ((a (make-array 19 :element-type element
                                :initial-contents (loop for i below 19 collect
                                                       (funcall decode (nth (mod i 5) left-bits)))))
              (b (make-array 19 :element-type element
                                :initial-contents (loop for i below 19 collect
                                                       (funcall decode (nth (mod i 5) right-bits)))))
              (mask (make-array 19 :element-type '(unsigned-byte 8)
                                   :initial-contents (loop for i below 19 collect (if (oddp i) 255 0))))
              (out (make-array 19 :element-type element)))
          (dolist (scalar '(nil t))
            (simd:select! out mask (if scalar (aref a 1) a) b)
            (dotimes (i 19)
              (is (= (if (zerop (aref mask i)) (nth (mod i 5) right-bits)
                         (if scalar (second left-bits) (nth (mod i 5) left-bits)))
                     (funcall encode (aref out i)))))))))))
