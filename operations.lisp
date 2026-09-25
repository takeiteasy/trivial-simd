(in-package #:trivial-simd)

(defun float-vector-type (vector)
  (cond ((typep vector '(simple-array single-float (*))) :f32)
        ((typep vector '(simple-array double-float (*))) :f64)
        (t (error 'type-error :datum vector
                  :expected-type '(or (simple-array single-float (*))
                                      (simple-array double-float (*)))))))

(defun zero-offsets (count)
  (case count
    (1 '(0))
    (2 '(0 0))
    (3 '(0 0 0))
    (t (make-list count :initial-element 0))))

(defun resolve-whole (vectors)
  (let* ((first (first vectors))
         (type (float-vector-type first))
         (length (length first)))
    (dolist (vector (rest vectors))
      (unless (eq type (float-vector-type vector))
        (error "All vectors must have the same element type"))
      (unless (= length (length vector))
        (error "All vectors must have the same length")))
    (values type length (zero-offsets (length vectors)))))

(defun resolve-slice (vectors starts start end)
  "Validate VECTORS and return the element type, count, and per-vector offsets.
STARTS holds each vector's own start or NIL to use START."
  (when (and (null start) (null end) (dolist (own starts t) (when own (return nil))))
    (return-from resolve-slice (resolve-whole vectors)))
  (let* ((type (float-vector-type (first vectors)))
         (length (length (first vectors)))
         (start (or start 0)))
    (dolist (vector (rest vectors))
      (unless (eq type (float-vector-type vector))
        (error "All vectors must have the same element type")))
    (unless end
      (dolist (vector (rest vectors))
        (unless (= length (length vector))
          (error "All vectors must have the same length without :END"))))
    (let ((end (or end length)))
      (unless (<= 0 start end)
        (error "Invalid slice: :START ~S, :END ~S" start end))
      (let* ((count (- end start))
             (offsets (mapcar (lambda (offset) (or offset start)) starts)))
        (loop for vector in vectors
              for offset in offsets
              do (unless (and (<= 0 offset) (<= (+ offset count) (length vector)))
                   (error "Slice of ~D elements at offset ~D exceeds vector length ~D"
                          count offset (length vector))))
        (values type count offsets)))))

(defun binary-operation (operation destination left right
                         start end destination-start left-start right-start)
  (let ((vectors (list destination left right))
        (starts (list destination-start left-start right-start)))
    (declare (dynamic-extent vectors starts))
    (multiple-value-bind (type count offsets) (resolve-slice vectors starts start end)
      (declare (ignore type))
      (destructuring-bind (destination-offset left-offset right-offset) offsets
      (ecase *backend*
        (:sbcl (sbcl-binary operation destination left right count
                            destination-offset left-offset right-offset))
        (:native (native-binary operation destination left right count
                                destination-offset left-offset right-offset))
        (:lisp (lisp-binary operation destination left right count
                            destination-offset left-offset right-offset))))))
  destination)

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

(defun sum (input &key start end input-start)
  "Return the sum of the :START to :END slice of INPUT, or a zero if empty."
  (multiple-value-bind (type count offsets)
      (resolve-slice (list input) (list input-start) start end)
    (declare (ignore type))
    (let ((offset (first offsets)))
      (ecase *backend*
        (:sbcl (sbcl-sum input count offset))
        (:native (native-sum input count offset))
        (:lisp (lisp-sum input count offset))))))

(defun dot (left right &key start end left-start right-start)
  "Return the dot product of the :START to :END slices of LEFT and RIGHT."
  (multiple-value-bind (type count offsets)
      (resolve-slice (list left right) (list left-start right-start) start end)
    (declare (ignore type))
    (destructuring-bind (left-offset right-offset) offsets
      (ecase *backend*
        (:sbcl (sbcl-dot left right count left-offset right-offset))
        (:native (native-dot left right count left-offset right-offset))
        (:lisp (lisp-dot left right count left-offset right-offset))))))
