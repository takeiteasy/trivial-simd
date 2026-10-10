(in-package #:trivial-simd)

(defmacro define-sbcl-conversions ()
  (let ((convert (find-symbol "F32.4-FROM-S32.4" "SB-SIMD-SSE2")))
    (when (and convert (fboundp convert))
      `(defun %sbcl-convert-s32-f32 (destination input count d-offset i-offset)
         (declare (optimize (speed 3) (safety 0))
                  (type (simple-array single-float (*)) destination)
                  (type (simple-array (signed-byte 32) (*)) input)
                  (type fixnum count d-offset i-offset))
         (let ((i 0))
           (declare (type fixnum i))
           (loop while (<= (+ i 4) count) do
             (setf (sb-simd-sse:f32.4-aref destination (+ d-offset i))
                   (,convert (sb-simd-sse2:s32.4-aref input (+ i-offset i))))
             (incf i 4))
           (loop while (< i count) do
             (setf (aref destination (+ d-offset i)) (coerce (aref input (+ i-offset i)) 'single-float))
             (incf i)))
         destination))))

(define-sbcl-conversions)
