(in-package #:trivial-simd)

(defvar *native-available-p* nil)

(defun native-library-path ()
  (let ((name (case (uiop:operating-system)
                (:macosx #p"build/libtrivial_simd.dylib")
                (:linux #p"build/libtrivial_simd.so")
                ((:windows :win) #p"build/trivial_simd.dll"))))
    (when name
      (merge-pathnames name (asdf:system-source-directory "trivial-simd")))))

(let ((path (native-library-path)))
  (when (and path (probe-file path))
    (cffi:load-foreign-library path)
    (setf *native-available-p* t)))

(cffi:defcfun ("ts_add_f32" %native-add-f32) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(cffi:defcfun ("ts_subtract_f32" %native-subtract-f32) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(cffi:defcfun ("ts_multiply_f32" %native-multiply-f32) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(cffi:defcfun ("ts_divide_f32" %native-divide-f32) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(cffi:defcfun ("ts_sum_f32" %native-sum-f32) :float
  (input :pointer) (length :size))
(cffi:defcfun ("ts_dot_f32" %native-dot-f32) :float
  (left :pointer) (right :pointer) (length :size))

(cffi:defcfun ("ts_add_f64" %native-add-f64) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(cffi:defcfun ("ts_subtract_f64" %native-subtract-f64) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(cffi:defcfun ("ts_multiply_f64" %native-multiply-f64) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(cffi:defcfun ("ts_divide_f64" %native-divide-f64) :void
  (destination :pointer) (left :pointer) (right :pointer) (length :size))
(cffi:defcfun ("ts_sum_f64" %native-sum-f64) :double
  (input :pointer) (length :size))
(cffi:defcfun ("ts_dot_f64" %native-dot-f64) :double
  (left :pointer) (right :pointer) (length :size))

(defun foreign-type (vector)
  (if (typep vector '(simple-array single-float (*))) :float :double))

(defun copy-to-foreign (vector pointer type)
  (dotimes (index (length vector))
    (setf (cffi:mem-aref pointer type index) (aref vector index))))

(defun copy-from-foreign (pointer vector type)
  (dotimes (index (length vector))
    (setf (aref vector index) (cffi:mem-aref pointer type index))))

(defun native-binary (operation destination left right)
  (let* ((length (length destination))
         (type (foreign-type destination)))
    #+(or sbcl ccl ecl)
    (cffi:with-pointer-to-vector-data (output destination)
      (cffi:with-pointer-to-vector-data (a left)
        (cffi:with-pointer-to-vector-data (b right)
          (call-native-binary operation type output a b length))))
    #-(or sbcl ccl ecl)
    (cffi:with-foreign-objects ((output type length) (a type length) (b type length))
      ;; TODO: copying limits throughput; add tested pinning adapters (ticket #2).
      (copy-to-foreign left a type)
      (copy-to-foreign right b type)
      (call-native-binary operation type output a b length)
      (copy-from-foreign output destination type)))
  destination)

(defun call-native-binary (operation type output a b length)
  (funcall (ecase operation
             (:add (if (eq type :float) #'%native-add-f32 #'%native-add-f64))
             (:subtract (if (eq type :float) #'%native-subtract-f32 #'%native-subtract-f64))
             (:multiply (if (eq type :float) #'%native-multiply-f32 #'%native-multiply-f64))
             (:divide (if (eq type :float) #'%native-divide-f32 #'%native-divide-f64)))
           output a b length))

(defun native-sum (input)
  (let* ((length (length input))
         (type (foreign-type input)))
    #+(or sbcl ccl ecl)
    (cffi:with-pointer-to-vector-data (a input)
      (if (eq type :float)
          (%native-sum-f32 a length)
          (%native-sum-f64 a length)))
    #-(or sbcl ccl ecl)
    (cffi:with-foreign-object (a type length)
      (copy-to-foreign input a type)
      (if (eq type :float)
          (%native-sum-f32 a length)
          (%native-sum-f64 a length)))))

(defun native-dot (left right)
  (let* ((length (length left))
         (type (foreign-type left)))
    #+(or sbcl ccl ecl)
    (cffi:with-pointer-to-vector-data (a left)
      (cffi:with-pointer-to-vector-data (b right)
        (if (eq type :float)
            (%native-dot-f32 a b length)
            (%native-dot-f64 a b length))))
    #-(or sbcl ccl ecl)
    (cffi:with-foreign-objects ((a type length) (b type length))
      (copy-to-foreign left a type)
      (copy-to-foreign right b type)
      (if (eq type :float)
          (%native-dot-f32 a b length)
          (%native-dot-f64 a b length)))))
