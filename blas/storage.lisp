(in-package #:trivial-simd/blas)

;; AREF is shadowed so that kernels can rebind it locally for vector views.
(defmacro aref (array &rest subscripts)
  `(cl:aref ,array ,@subscripts))

(defun storage-element-type (storage)
  "Element type of a BLAS vector or matrix storage: a simple float or complex
float vector, or a vector view of one of those types; NIL for anything else."
  (cond ((typep storage '(simple-array single-float (*))) 'single-float)
        ((typep storage '(simple-array double-float (*))) 'double-float)
        ((typep storage '(simple-array (complex single-float) (*))) '(complex single-float))
        ((typep storage '(simple-array (complex double-float) (*))) '(complex double-float))
        ((trivial-simd:vector-view-p storage)
         (case (trivial-simd:vector-view-type storage)
           (:f32 'single-float) (:f64 'double-float)
           (:c32 '(complex single-float)) (:c64 '(complex double-float))))))

(declaim (inline element (setf element)))
(defun element (storage index)
  "Element INDEX of a BLAS storage vector or view, for untyped loops."
  (trivial-simd::vector-ref storage index))

(defun (setf element) (value storage index)
  (setf (trivial-simd::vector-ref storage index) value))

(defun storage-length (storage)
  (trivial-simd::vector-length storage))

(defun storage-origin (storage)
  "Return an object identifying STORAGE's memory and the byte address of its
first element there, so that slices of two storages can be compared."
  (if (trivial-simd:vector-view-p storage)
      (values :foreign (cffi:pointer-address (trivial-simd:vector-view-pointer storage)))
      (values storage 0)))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *storage-types*
    '((single-float :float) (double-float :double)
      ((complex single-float) :float) ((complex double-float) :double)))

  (defun storage-accessor (type)
    (intern (format nil "STORAGE-REF/~A"
                    (if (consp type) (format nil "COMPLEX-~A" (second type)) type))
            :trivial-simd/blas)))

(defmacro define-storage-accessors ()
  `(progn
     ,@(loop for (type foreign) in *storage-types*
             for name = (storage-accessor type)
             for complexp = (consp type)
             collect
             `(progn
                (declaim (inline ,name (setf ,name)))
                (defun ,name (view-p array pointer index)
                  (declare (type (simple-array ,type (*)) array) (type fixnum index)
                           (optimize (speed 3) (safety 0) (debug 0)))
                  (if view-p
                      ,(if complexp
                           `(complex (cffi:mem-aref pointer ,foreign (* 2 index))
                                     (cffi:mem-aref pointer ,foreign (1+ (* 2 index))))
                           `(cffi:mem-aref pointer ,foreign index))
                      (cl:aref array index)))
                (defun (setf ,name) (value view-p array pointer index)
                  (declare (type ,type value) (type (simple-array ,type (*)) array)
                           (type fixnum index) (optimize (speed 3) (safety 0) (debug 0)))
                  (if view-p
                      ,(if complexp
                           `(setf (cffi:mem-aref pointer ,foreign (* 2 index)) (realpart value)
                                  (cffi:mem-aref pointer ,foreign (1+ (* 2 index))) (imagpart value))
                           `(setf (cffi:mem-aref pointer ,foreign index) value))
                      (setf (cl:aref array index) value))
                  value)))))

(define-storage-accessors)

(defun storage-parameters (body type)
  "Split BODY's leading declarations: return the variables declared as simple
vectors of TYPE and the declarations without those type specifiers."
  (let ((array `(simple-array ,type (*))) (storage '()) (declarations '()))
    (loop while (and (consp (first body)) (eq (first (first body)) 'declare))
          do (push `(declare
                     ,@(loop for specifier in (rest (pop body))
                             if (and (consp specifier) (eq (first specifier) 'type)
                                     (equal (second specifier) array))
                               do (setf storage (append storage (cddr specifier)))
                             else collect specifier))
                   declarations))
    (values storage (nreverse declarations) body)))

(defun view-kernel-body (type storage declarations forms)
  "Body of a kernel variant whose STORAGE parameters may each be a vector view:
every (AREF parameter index) reads through the view's pointer or the array."
  (let* ((table (loop for variable in storage
                      collect (list variable (gensym "VIEW-P") (gensym "ARRAY")
                                    (gensym "POINTER"))))
         (accessor (storage-accessor type))
         (expander (gensym "SUBSCRIPTS")))
    `(,@declarations
      (let* ,(loop for (variable view-p array pointer) in table
                   append `((,view-p (trivial-simd:vector-view-p ,variable))
                            (,array (if ,view-p
                                        (load-time-value (make-array 0 :element-type ',type))
                                        ,variable))
                            (,pointer (if ,view-p
                                          (trivial-simd:vector-view-pointer ,variable)
                                          (cffi:null-pointer)))))
        (declare (type (simple-array ,type (*)) ,@(mapcar #'third table))
                 (ignorable ,@(loop for entry in table append (rest entry))))
        (macrolet ((aref (array &rest ,expander)
                     (let ((entry (assoc array ',table)))
                       (cond ((null entry) `(cl:aref ,array ,@,expander))
                             ((/= 1 (length ,expander))
                              (error "A BLAS storage vector takes one subscript"))
                             (t `(,',accessor ,@(rest entry) ,@,expander))))))
          ,@forms)))))
