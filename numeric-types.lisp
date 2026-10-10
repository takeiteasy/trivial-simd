(in-package #:trivial-simd)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *numeric-types*
    '((:f32 single-float :float 32 nil)
      (:f64 double-float :double 64 nil)
      (:s8 (signed-byte 8) :int8 8 t)
      (:u8 (unsigned-byte 8) :uint8 8 nil)
      (:s16 (signed-byte 16) :int16 16 t)
      (:u16 (unsigned-byte 16) :uint16 16 nil)
      (:s32 (signed-byte 32) :int32 32 t)
      (:u32 (unsigned-byte 32) :uint32 32 nil)
      (:s64 (signed-byte 64) :int64 64 t)
      (:u64 (unsigned-byte 64) :uint64 64 nil))))

(defun numeric-type (type)
  (or (assoc type *numeric-types*)
      (case type
        (:c32 '(:c32 (complex single-float) nil 64 nil))
        (:c64 '(:c64 (complex double-float) nil 128 nil)))
      (error "Unknown numeric type ~S" type)))

(defun integer-type-p (type)
  (member type '(:s8 :u8 :s16 :u16 :s32 :u32 :s64 :u64)))

(defun integer-limits (type)
  (destructuring-bind (key element foreign bits signed) (numeric-type type)
    (declare (ignore key element foreign))
    (if signed
        (values (- (ash 1 (1- bits))) (1- (ash 1 (1- bits))))
        (values 0 (1- (ash 1 bits))))))

(defun complex-type-p (type)
  (member type '(:c32 :c64)))

(defun integer-operation-symbol (operation type)
  (intern (format nil "INTEGER-~A-~A" operation type) :trivial-simd))

(defstruct (vector-view (:constructor %make-vector-view (pointer type length))
                        (:copier nil))
  "LENGTH elements of numeric TYPE in foreign memory starting at POINTER."
  (pointer nil :read-only t)
  (type :f32 :type keyword :read-only t)
  (length 0 :type (integer 0 #.most-positive-fixnum) :read-only t))

(defmacro define-vector-type ()
  `(defun vector-type (vector)
     (etypecase vector
       ((simple-array (complex single-float) (*)) :c32)
       ((simple-array (complex double-float) (*)) :c64)
       ,@(loop for (key element) in *numeric-types*
               collect `((simple-array ,element (*)) ,key))
       (vector-view (vector-view-type vector)))))

(define-vector-type)
