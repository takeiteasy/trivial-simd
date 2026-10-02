(in-package #:trivial-simd)

(defvar *sbcl-simd-available-p* nil)

#+(and sbcl x86-64)
(when (ignore-errors (require :sb-simd))
  (when (find-package :sb-simd-sse2)
    (load (asdf:system-relative-pathname "trivial-simd" "sbcl-simd.lisp"))
    (load (asdf:system-relative-pathname "trivial-simd" "sbcl-integers.lisp"))
    (setf *sbcl-simd-available-p* t)
    (initialize-sbcl-fma)))

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

(defmacro define-lisp-float-binary ()
  `(defun lisp-float-binary (type operation destination left right count
                             destination-offset left-offset right-offset)
     (ecase type
       ,@(loop for (key element) in '((:f32 single-float) (:f64 double-float))
               collect
               `(,key
                 (locally (declare (type (simple-array ,element (*)) destination left right)
                                   (type fixnum count destination-offset left-offset right-offset))
                   (ecase operation
                     ,@(loop for (op scalar-operator) in '((:add +) (:subtract -)
                                                           (:multiply *) (:divide /))
                             collect
                             `(,op (dotimes (i count)
                                     (setf (aref destination (+ destination-offset i))
                                           (,scalar-operator (aref left (+ left-offset i))
                                                             (aref right (+ right-offset i))))))))))))
     destination))

(define-lisp-float-binary)

(defun lisp-binary (operation destination left right count
                    destination-offset left-offset right-offset)
  (let ((type (vector-type destination)))
    (if (integer-type-p type)
        (integer-binary type operation destination left right count
                        destination-offset left-offset right-offset)
        (lisp-float-binary type operation destination left right count
                           destination-offset left-offset right-offset))))

(defmacro define-lisp-bulk-loops ()
  `(progn
     ,@(loop for (key element) in *numeric-types*
             for scalar-name = (intern (format nil "%LISP-SCALAR-BINARY-~A" key))
             for scale-name = (intern (format nil "%LISP-SCALE-~A" key))
             for axpy-name = (intern (format nil "%LISP-AXPY-~A" key))
             collect
             `(progn
                (defun ,scalar-name (operation destination left right count d-offset l-offset r-offset)
                  (declare (type (simple-array ,element (*)) destination)
                           (type fixnum count d-offset))
                  (let ((left-vector (vectorp left)) (right-vector (vectorp right)))
                    (dotimes (i count)
                      (let ((a (if left-vector (aref left (+ l-offset i)) left))
                            (b (if right-vector (aref right (+ r-offset i)) right)))
                        (setf (aref destination (+ d-offset i))
                              (ecase operation
                                ,@(loop for op in '(:add :subtract :multiply :divide)
                                        for function = (if (integer-type-p key)
                                                           (integer-operation-symbol op key)
                                                           (case op (:add '+) (:subtract '-)
                                                                 (:multiply '*) (:divide '/)))
                                        collect `(,op (,function a b))))))))
                  destination)
                (defun ,scale-name (x a count offset)
                  (declare (type (simple-array ,element (*)) x)
                           (type ,element a) (type fixnum count offset))
                  (dotimes (i count)
                    (setf (aref x (+ offset i))
                          (,(if (integer-type-p key) (integer-operation-symbol :multiply key) '*)
                           a (aref x (+ offset i)))))
                  x)
                (defun ,axpy-name (y a x count y-offset x-offset)
                  (declare (type (simple-array ,element (*)) y x)
                           (type ,element a) (type fixnum count y-offset x-offset))
                  (dotimes (i count)
                    (setf (aref y (+ y-offset i))
                          (,(if (integer-type-p key) (integer-operation-symbol :add key) '+)
                           (,(if (integer-type-p key) (integer-operation-symbol :multiply key) '*)
                            a (aref x (+ x-offset i)))
                           (aref y (+ y-offset i)))))
                  y)))))

(define-lisp-bulk-loops)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun lisp-bulk-symbol (prefix type)
    (intern (format nil "~A~A" prefix type) :trivial-simd)))

(defun lisp-bulk-function (prefix vector)
  (lisp-bulk-symbol prefix (vector-type vector)))

(define-compiler-macro lisp-bulk-function (&whole form prefix vector)
  (if (stringp prefix)
      `(ecase (vector-type ,vector)
         ,@(loop for key in (append (mapcar #'first *numeric-types*) '(:c32 :c64))
                 collect `(,key ',(lisp-bulk-symbol prefix key))))
      form))

(defun lisp-scalar-binary (operation destination left right count d-offset l-offset r-offset)
  (funcall (lisp-bulk-function "%LISP-SCALAR-BINARY-" destination)
           operation destination left right count d-offset l-offset r-offset))

(defun lisp-scale (x a count offset)
  (funcall (lisp-bulk-function "%LISP-SCALE-" x) x a count offset))

(defun lisp-axpy (y a x count y-offset x-offset)
  (funcall (lisp-bulk-function "%LISP-AXPY-" y) y a x count y-offset x-offset))

(defun lisp-bulk-fma (destination x y z count d-offset x-offset y-offset z-offset)
  (dotimes (i count)
    (setf (aref destination (+ d-offset i))
          (fma (if x-offset (aref x (+ x-offset i)) x)
               (if y-offset (aref y (+ y-offset i)) y)
               (if z-offset (aref z (+ z-offset i)) z))))
  destination)

(defun sbcl-scalar-binary (operation destination left right count d-offset l-offset r-offset)
  #-(and sbcl x86-64)
  (declare (ignore operation destination left right count d-offset l-offset r-offset))
  #+(and sbcl x86-64)
  (if (integer-type-p (vector-type destination))
      (lisp-scalar-binary operation destination left right count d-offset l-offset r-offset)
      (funcall (if (eq (vector-type destination) :f32)
                   #'%sbcl-scalar-binary-f32 #'%sbcl-scalar-binary-f64)
               operation destination left right count d-offset l-offset r-offset))
  #-(and sbcl x86-64)
  (error "SBCL SIMD is unavailable on this platform"))

(defun sbcl-scale (x a count offset)
  (sbcl-scalar-binary :multiply x x a count offset offset nil))

(defun sbcl-axpy (y a x count y-offset x-offset)
  #-(and sbcl x86-64)
  (declare (ignore y a x count y-offset x-offset))
  #+(and sbcl x86-64)
  (if (integer-type-p (vector-type y))
      (lisp-axpy y a x count y-offset x-offset)
      (funcall (if (eq (vector-type y) :f32) #'%sbcl-axpy-f32 #'%sbcl-axpy-f64)
               y a x count y-offset x-offset))
  #-(and sbcl x86-64)
  (error "SBCL SIMD is unavailable on this platform"))

(defun sbcl-bulk-fma (destination x y z count d-offset x-offset y-offset z-offset)
  #-(and sbcl x86-64)
  (declare (ignore destination x y z count d-offset x-offset y-offset z-offset))
  #+(and sbcl x86-64)
  (lisp-bulk-fma destination x y z count d-offset x-offset y-offset z-offset)
  #-(and sbcl x86-64)
  (error "SBCL SIMD is unavailable on this platform"))

(defmacro define-lisp-float-reductions ()
  `(progn
     (defun lisp-sum (input count offset)
       (let ((type (vector-type input)))
         (when (integer-type-p type)
           (return-from lisp-sum (integer-sum type input count offset)))
         (ecase type
           ,@(loop for (key element) in '((:f32 single-float) (:f64 double-float))
                   collect
                   `(,key
                     (locally (declare (type (simple-array ,element (*)) input)
                                       (type fixnum count offset))
                       (let ((result (coerce 0 ',element)))
                         (declare (type ,element result))
                         (dotimes (i count result)
                           (incf result (aref input (+ offset i)))))))))))
     (defun lisp-dot (left right count left-offset right-offset)
       (let ((type (vector-type left)))
         (when (integer-type-p type)
           (return-from lisp-dot
             (integer-dot type left right count left-offset right-offset)))
         (ecase type
           ,@(loop for (key element) in '((:f32 single-float) (:f64 double-float))
                   collect
                   `(,key
                     (locally (declare (type (simple-array ,element (*)) left right)
                                       (type fixnum count left-offset right-offset))
                       (let ((result (coerce 0 ',element)))
                         (declare (type ,element result))
                         (dotimes (i count result)
                           (incf result (* (aref left (+ left-offset i))
                                           (aref right (+ right-offset i))))))))))))))

(define-lisp-float-reductions)

(defun sbcl-binary (operation destination left right count
                    destination-offset left-offset right-offset)
  #-(and sbcl x86-64)
  (declare (ignore operation destination left right count
                   destination-offset left-offset right-offset))
  #+(and sbcl x86-64)
  (when (integer-type-p (vector-type destination))
    (return-from sbcl-binary
      (sbcl-integer-binary (vector-type destination) operation destination left right count
                           destination-offset left-offset right-offset)))
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
  (when (integer-type-p (vector-type input))
    (return-from sbcl-sum (sbcl-integer-sum (vector-type input) input count offset)))
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
  (when (integer-type-p (vector-type left))
    (return-from sbcl-dot (integer-dot (vector-type left) left right count left-offset right-offset)))
  #+(and sbcl x86-64)
  (if (typep left '(simple-array single-float (*)))
      (%sbcl-dot-f32 left right count left-offset right-offset)
      (%sbcl-dot-f64 left right count left-offset right-offset))
  #-(and sbcl x86-64)
  (error "SBCL SIMD is unavailable on this platform"))
