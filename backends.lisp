(in-package #:trivial-simd)

(defvar *sbcl-simd-available-p* nil)

#+(and sbcl x86-64)
(when (ignore-errors (require :sb-simd))
  (when (find-package :sb-simd-sse2)
    (load (asdf:system-relative-pathname "trivial-simd" "sbcl-simd.lisp"))
    (setf *sbcl-simd-available-p* t)))

(defun select-backend ()
  (let ((requested (uiop:getenv "TRIVIAL_SIMD_BACKEND")))
    (cond ((or (null requested) (string= requested "")
               (string-equal requested "auto"))
           (cond (*sbcl-simd-available-p* :sbcl)
                 (*native-available-p* :native)
                 (t :lisp)))
          ((string-equal requested "lisp") :lisp)
          ((and (string-equal requested "native") *native-available-p*) :native)
          ((and (string-equal requested "sbcl") *sbcl-simd-available-p*) :sbcl)
          (t (error "Requested SIMD backend ~S is unavailable" requested)))))

(defvar *backend* (select-backend))

(defun backend ()
  "Return :SBCL, :NATIVE, or :LISP for the active backend."
  *backend*)

(defmacro lisp-loop (scalar-operator)
  `(dotimes (i count)
     (setf (aref destination (+ destination-offset i))
           (,scalar-operator (aref left (+ left-offset i))
                             (aref right (+ right-offset i))))))

(defun lisp-binary (operation destination left right count
                    destination-offset left-offset right-offset)
  (ecase operation
    (:add (lisp-loop +))
    (:subtract (lisp-loop -))
    (:multiply (lisp-loop *))
    (:divide (lisp-loop /)))
  destination)

(defun lisp-sum (input count offset)
  (let ((result (if (typep input '(simple-array single-float (*))) 0.0f0 0.0d0)))
    (dotimes (i count result)
      (incf result (aref input (+ offset i))))))

(defun lisp-dot (left right count left-offset right-offset)
  (let ((result (if (typep left '(simple-array single-float (*))) 0.0f0 0.0d0)))
    (dotimes (i count result)
      (incf result (* (aref left (+ left-offset i))
                      (aref right (+ right-offset i)))))))

(defun sbcl-binary (operation destination left right count
                    destination-offset left-offset right-offset)
  #-(and sbcl x86-64)
  (declare (ignore operation destination left right count
                   destination-offset left-offset right-offset))
  #+(and sbcl x86-64)
  (funcall (ecase operation
             (:add (if (typep destination '(simple-array single-float (*)))
                       #'%sbcl-add-f32 #'%sbcl-add-f64))
             (:subtract (if (typep destination '(simple-array single-float (*)))
                            #'%sbcl-subtract-f32 #'%sbcl-subtract-f64))
             (:multiply (if (typep destination '(simple-array single-float (*)))
                            #'%sbcl-multiply-f32 #'%sbcl-multiply-f64))
             (:divide (if (typep destination '(simple-array single-float (*)))
                          #'%sbcl-divide-f32 #'%sbcl-divide-f64)))
           destination left right count destination-offset left-offset right-offset)
  #-(and sbcl x86-64)
  (error "SBCL SIMD is unavailable on this platform"))

(defun sbcl-sum (input count offset)
  #-(and sbcl x86-64)
  (declare (ignore input count offset))
  #+(and sbcl x86-64)
  (if (typep input '(simple-array single-float (*)))
      (%sbcl-sum-f32 input count offset)
      (%sbcl-sum-f64 input count offset))
  #-(and sbcl x86-64)
  (error "SBCL SIMD is unavailable on this platform"))

(defun sbcl-dot (left right count left-offset right-offset)
  #-(and sbcl x86-64)
  (declare (ignore left right count left-offset right-offset))
  #+(and sbcl x86-64)
  (if (typep left '(simple-array single-float (*)))
      (%sbcl-dot-f32 left right count left-offset right-offset)
      (%sbcl-dot-f64 left right count left-offset right-offset))
  #-(and sbcl x86-64)
  (error "SBCL SIMD is unavailable on this platform"))
