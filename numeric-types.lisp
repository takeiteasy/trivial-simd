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
  (or (assoc type *numeric-types*) (error "Unknown numeric type ~S" type)))

(defun integer-type-p (type)
  (not (member type '(:f32 :f64))))

(defun integer-operation-symbol (operation type)
  (intern (format nil "INTEGER-~A-~A" operation type) :trivial-simd))

(defmacro define-vector-type ()
  `(defun vector-type (vector)
     (etypecase vector
       ,@(loop for (key element) in *numeric-types*
               collect `((simple-array ,element (*)) ,key)))))

(define-vector-type)
