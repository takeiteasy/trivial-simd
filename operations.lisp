(in-package #:trivial-simd)

(defun float-vector-type (vector)
  (cond ((typep vector '(simple-array single-float (*))) :f32)
        ((typep vector '(simple-array double-float (*))) :f64)
        (t (error 'type-error :datum vector
                  :expected-type '(or (simple-array single-float (*))
                                      (simple-array double-float (*)))))))

(defun validate-vectors (&rest vectors)
  (let ((type (float-vector-type (first vectors)))
        (length (length (first vectors))))
    (dolist (vector (rest vectors))
      (unless (eq type (float-vector-type vector))
        (error "All vectors must have the same element type"))
      (unless (= length (length vector))
        (error "All vectors must have the same length")))
    type))

(defun binary-operation (operation destination left right)
  (validate-vectors destination left right)
  (ecase *backend*
    (:sbcl (sbcl-binary operation destination left right))
    (:native (native-binary operation destination left right))
    (:lisp (lisp-binary operation destination left right))))

(defun add! (destination left right)
  "Add LEFT and RIGHT elementwise into DESTINATION and return it."
  (binary-operation :add destination left right))

(defun subtract! (destination left right)
  "Subtract RIGHT from LEFT elementwise into DESTINATION and return it."
  (binary-operation :subtract destination left right))

(defun multiply! (destination left right)
  "Multiply LEFT and RIGHT elementwise into DESTINATION and return it."
  (binary-operation :multiply destination left right))

(defun divide! (destination left right)
  "Divide LEFT by RIGHT elementwise into DESTINATION and return it."
  (binary-operation :divide destination left right))

(defun sum (input)
  "Return the sum of INPUT, or a zero of its element type if empty."
  (float-vector-type input)
  (ecase *backend*
    (:sbcl (sbcl-sum input))
    (:native (native-sum input))
    (:lisp (lisp-sum input))))

(defun dot (left right)
  "Return the dot product of LEFT and RIGHT."
  (validate-vectors left right)
  (ecase *backend*
    (:sbcl (sbcl-dot left right))
    (:native (native-dot left right))
    (:lisp (lisp-dot left right))))
