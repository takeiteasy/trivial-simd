(in-package #:trivial-simd)

(defun zero-offsets (count)
  (case count
    (1 '(0))
    (2 '(0 0))
    (3 '(0 0 0))
    (t (make-list count :initial-element 0))))

(defun resolve-whole (vectors)
  (let* ((first (first vectors))
         (type (vector-type first))
         (length (length first)))
    (dolist (vector (rest vectors))
      (unless (eq type (vector-type vector))
        (error "All vectors must have the same element type"))
      (unless (= length (length vector))
        (error "All vectors must have the same length")))
    (values type length (zero-offsets (length vectors)))))

(defun resolve-slice (vectors starts start end)
  "Validate VECTORS and return the element type, count, and per-vector offsets.
STARTS holds each vector's own start or NIL to use START."
  (when (and (null start) (null end) (dolist (own starts t) (when own (return nil))))
    (return-from resolve-slice (resolve-whole vectors)))
  (let* ((type (vector-type (first vectors)))
         (length (length (first vectors)))
         (start (or start 0)))
    (dolist (vector (rest vectors))
      (unless (eq type (vector-type vector))
        (error "All vectors must have the same element type")))
    (unless end
      (dolist (vector (rest vectors))
        (unless (= length (length vector))
          (error "All vectors must have the same length without :END"))))
    (let ((end (or end length)))
      (unless (and (integerp start) (integerp end) (<= 0 start end))
        (error "Invalid slice: :START ~S, :END ~S" start end))
      (let* ((count (- end start))
             (offsets (mapcar (lambda (offset) (or offset start)) starts)))
        (loop for vector in vectors
              for offset in offsets
              do (unless (and (integerp offset) (<= 0 offset)
                              (<= (+ offset count) (length vector)))
                   (error "Invalid slice of ~D elements at offset ~S for vector length ~D"
                          count offset (length vector))))
        (values type count offsets)))))

(defun binary-operation (operation destination left right
                         start end destination-start left-start right-start)
  (unless (or (vectorp left) (vectorp right))
    (error "A binary operation needs a vector operand"))
  (multiple-value-bind (type count offsets)
      (resolve-bulk-operands destination (list left right)
                             (list left-start right-start) start end destination-start)
    (destructuring-bind (destination-offset left-offset right-offset) offsets
      (cond ((complex-type-p type)
             (complex-binary operation destination left right count
                             destination-offset left-offset right-offset))
            ((and (vectorp left) (vectorp right))
             (ecase *backend*
               (:sbcl (sbcl-binary operation destination left right count
                                   destination-offset left-offset right-offset))
               (:native (native-binary operation destination left right count
                                       destination-offset left-offset right-offset))
               (:lisp (lisp-binary operation destination left right count
                                   destination-offset left-offset right-offset))))
            (t
             (ecase *backend*
               (:sbcl (sbcl-scalar-binary operation destination left right count
                                          destination-offset left-offset right-offset))
               (:native (native-scalar-binary operation destination left right count
                                            destination-offset left-offset right-offset))
               (:lisp (lisp-scalar-binary operation destination left right count
                                          destination-offset left-offset right-offset)))))))
  destination)

(defun resolve-bulk-operands (destination operands starts start end destination-start)
  (let ((type (vector-type destination))
        (vectors (list destination))
        (vector-starts (list destination-start)))
    (loop for operand in operands for offset in starts do
      (if (vectorp operand)
          (progn (push operand vectors) (push offset vector-starts))
          (progn
            (when offset (error "A scalar operand has no start offset"))
            (unless (typep operand (second (numeric-type type)))
              (error 'type-error :datum operand
                     :expected-type (second (numeric-type type)))))))
    (multiple-value-bind (resolved count offsets)
        (resolve-slice (nreverse vectors) (nreverse vector-starts) start end)
      (let ((remaining (rest offsets)) (result (list (first offsets))))
        (dolist (operand operands)
          (push (when (vectorp operand) (pop remaining)) result))
        (values resolved count (nreverse result))))))

(defmacro define-binary-operation (name operation verb)
  `(defun ,name (destination left right &key start end
                                          destination-start left-start right-start)
     ,(format nil "~A LEFT and RIGHT elementwise into DESTINATION and return it.
The slice is :START to :END (default the whole vector); the ...-START keywords
override the start for one vector." verb)
     (binary-operation ,operation destination left right
                       start end destination-start left-start right-start)))

(define-binary-operation add! :add "Add")
(define-binary-operation subtract! :subtract "Subtract RIGHT from")
(define-binary-operation multiply! :multiply "Multiply")
(define-binary-operation divide! :divide "Divide LEFT by RIGHT,")

(defun scale! (x a &key start end)
  "Multiply X by scalar A in place and return X."
  (when (vectorp a) (error "SCALE! needs a scalar multiplier"))
  (multiple-value-bind (type count offsets)
      (resolve-bulk-operands x (list a) (list nil) start end nil)
    (let ((offset (first offsets)))
      (if (complex-type-p type)
          (complex-scale x a count offset)
          (ecase *backend*
            (:sbcl (sbcl-scale x a count offset))
            (:native (native-scale x a count offset))
            (:lisp (lisp-scale x a count offset))))))
  x)

(defun axpy! (y a x &key start end y-start x-start)
  "Set Y to A*X+Y in place with separate multiplication and addition."
  (when (vectorp a) (error "AXPY! needs a scalar multiplier"))
  (unless (vectorp x) (error "AXPY! needs a vector input"))
  (multiple-value-bind (type count offsets)
      (resolve-bulk-operands y (list a x) (list nil x-start) start end y-start)
    (destructuring-bind (y-offset scalar-offset x-offset) offsets
      (declare (ignore scalar-offset))
      (if (complex-type-p type)
          (complex-axpy y a x count y-offset x-offset)
          (ecase *backend*
            (:sbcl (sbcl-axpy y a x count y-offset x-offset))
            (:native (native-axpy y a x count y-offset x-offset))
            (:lisp (lisp-axpy y a x count y-offset x-offset))))))
  y)

(defun fma! (destination x y z &key start end destination-start x-start y-start z-start)
  "Set DESTINATION to fused X*Y+Z elementwise and return it."
  (unless (member (vector-type destination) '(:f32 :f64))
    (error "FMA requires float vectors"))
  (multiple-value-bind (type count offsets)
      (resolve-bulk-operands destination (list x y z) (list x-start y-start z-start)
                             start end destination-start)
    (declare (ignore type))
    (destructuring-bind (d-offset x-offset y-offset z-offset) offsets
      (ecase *backend*
        (:sbcl (sbcl-bulk-fma destination x y z count d-offset x-offset y-offset z-offset))
        (:native (native-bulk-fma destination x y z count d-offset x-offset y-offset z-offset))
        (:lisp (lisp-bulk-fma destination x y z count d-offset x-offset y-offset z-offset)))))
  destination)

(defun sum (input &key start end input-start)
  "Return the sum of the :START to :END slice of INPUT, or a zero if empty."
  (multiple-value-bind (type count offsets)
      (resolve-slice (list input) (list input-start) start end)
    (let ((offset (first offsets)))
      (if (complex-type-p type)
          (complex-sum input count offset)
          (ecase *backend*
            (:sbcl (sbcl-sum input count offset))
            (:native (native-sum input count offset))
            (:lisp (lisp-sum input count offset)))))))

(defun dot (left right &key start end left-start right-start)
  "Return the dot product of the :START to :END slices of LEFT and RIGHT."
  (multiple-value-bind (type count offsets)
      (resolve-slice (list left right) (list left-start right-start) start end)
    (destructuring-bind (left-offset right-offset) offsets
      (if (complex-type-p type)
          (complex-dot left right count left-offset right-offset nil)
          (ecase *backend*
            (:sbcl (sbcl-dot left right count left-offset right-offset))
            (:native (native-dot left right count left-offset right-offset))
            (:lisp (lisp-dot left right count left-offset right-offset)))))))

(defun dotc (left right &key start end left-start right-start)
  "Return the dot product with LEFT conjugated."
  (multiple-value-bind (type count offsets)
      (resolve-slice (list left right) (list left-start right-start) start end)
    (unless (complex-type-p type)
      (error "DOTC requires complex vectors"))
    (complex-dot left right count (first offsets) (second offsets) t)))
