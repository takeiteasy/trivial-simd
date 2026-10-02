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

(defun spread-strides (operands strides)
  "Place the strides of the vectors among OPERANDS, after the destination's, with NIL for scalars."
  (let ((remaining (rest strides)) (result (list (first strides))))
    (dolist (operand operands (nreverse result))
      (push (when (vectorp operand) (pop remaining)) result))))

(declaim (inline resolve-strided-slice))
(defun resolve-strided-slice (vectors starts start end strides)
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
              for index from 0
              for stride = (if strides (nth index strides) 1)
              do (unless (span-valid-p offset count stride (length vector))
                   (error "Invalid slice of ~D elements at offset ~S, stride ~D, for vector length ~D"
                          count offset stride (length vector))))
        (values type count offsets strides)))))

(defun resolve-slice (vectors starts start end &optional own-strides stride)
  "Validate VECTORS and return the element type, count, per-vector offsets and
per-vector strides. STARTS and OWN-STRIDES hold each vector's own start or
stride, or NIL to use START or STRIDE (default 1)."
  (let ((strides (resolve-strides own-strides stride (length vectors))))
    (when (and (null start) (null end) (null strides)
               (dolist (own starts t) (when own (return nil))))
      (multiple-value-bind (type length offsets) (resolve-whole vectors)
        (return-from resolve-slice (values type length offsets strides))))
    (resolve-strided-slice vectors starts start end strides)))

(defun dispatch-binary (type operation destination left right count
                        destination-offset left-offset right-offset)
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
                                      destination-offset left-offset right-offset))))))

(defun binary-operation (operation destination left right
                         start end destination-start left-start right-start
                         stride destination-stride left-stride right-stride)
  (unless (or (vectorp left) (vectorp right))
    (error "A binary operation needs a vector operand"))
  (when (and (vectorp left) (vectorp right)
             (null start) (null end) (null destination-start) (null left-start)
             (null right-start) (null stride) (null destination-stride)
             (null left-stride) (null right-stride))
    (let ((type (vector-type destination))
          (count (length destination)))
      (when (and (eq type (vector-type left)) (eq type (vector-type right))
                 (= count (length left) (length right)))
        (dispatch-binary type operation destination left right count 0 0 0)
        (return-from binary-operation destination))))
  (when (and (null start) (null end) (null destination-start) (null left-start)
             (null right-start) (null stride) (null destination-stride)
             (null left-stride) (null right-stride))
    (let* ((type (vector-type destination))
           (vector (if (vectorp left) left right))
           (scalar (if (vectorp left) right left)))
      (when (and (not (vectorp scalar)) (not (complex-type-p type))
                 (eq type (vector-type vector))
                 (= (length destination) (length vector))
                 (typep scalar (second (numeric-type type))))
        (dispatch-binary type operation destination left right (length destination) 0
                         (and (vectorp left) 0) (and (vectorp right) 0))
        (return-from binary-operation destination))))
  (multiple-value-bind (type count offsets strides)
      (resolve-bulk-operands destination (list left right)
                             (list left-start right-start) start end destination-start
                             (list left-stride right-stride) stride destination-stride)
    (destructuring-bind (destination-offset left-offset right-offset) offsets
      (destructuring-bind (&optional destination-stride left-stride right-stride) strides
        (with-gathered ((destination destination-offset destination-stride :out)
                        (left left-offset left-stride)
                        (right right-offset right-stride))
            count
          (dispatch-binary type operation destination left right count
                           destination-offset left-offset right-offset)))))
  destination)

(defun resolve-bulk-operands (destination operands starts start end destination-start
                              &optional operand-strides stride destination-stride)
  "Like RESOLVE-SLICE for DESTINATION and the vectors among OPERANDS. The offsets
and strides cover DESTINATION then every operand, with NIL for scalars."
  (let* ((type (vector-type destination))
         (strided (or stride destination-stride (some #'identity operand-strides)))
         (vectors (list destination))
         (vector-starts (list destination-start))
         (vector-strides (and strided (list destination-stride))))
    (loop for operand in operands
          for offset in starts
          for index from 0
          for own-stride = (nth index operand-strides)
          do (if (vectorp operand)
                 (progn (push operand vectors) (push offset vector-starts)
                        (when strided (push own-stride vector-strides)))
                 (progn
                   (when offset (error "A scalar operand has no start offset"))
                   (when own-stride (error "A scalar operand has no stride"))
                   (unless (typep operand (second (numeric-type type)))
                     (error 'type-error :datum operand
                            :expected-type (second (numeric-type type)))))))
    (multiple-value-bind (resolved count offsets strides)
        (resolve-slice (nreverse vectors) (nreverse vector-starts) start end
                       (nreverse vector-strides) stride)
      (let ((remaining-offsets (rest offsets)) (result-offsets (list (first offsets))))
        (dolist (operand operands)
          (push (when (vectorp operand) (pop remaining-offsets)) result-offsets))
        (values resolved count (nreverse result-offsets)
                (and strides (spread-strides operands strides)))))))

(defmacro define-binary-operation (name operation verb)
  `(defun ,name (destination left right &key start end
                                          destination-start left-start right-start
                                          stride destination-stride left-stride right-stride)
     ,(format nil "~A LEFT and RIGHT elementwise into DESTINATION and return it.
The slice is :START to :END (default the whole vector) at :STRIDE (default 1);
the ...-START and ...-STRIDE keywords override one vector." verb)
     (binary-operation ,operation destination left right
                       start end destination-start left-start right-start
                       stride destination-stride left-stride right-stride)))

(define-binary-operation add! :add "Add")
(define-binary-operation subtract! :subtract "Subtract RIGHT from")
(define-binary-operation multiply! :multiply "Multiply")
(define-binary-operation divide! :divide "Divide LEFT by RIGHT,")

(defun dispatch-scale (x a count offset)
  (ecase *backend*
    (:sbcl (sbcl-scale x a count offset))
    (:native (native-scale x a count offset))
    (:lisp (lisp-scale x a count offset))))

(defun scale! (x a &key start end stride)
  "Multiply X by scalar A in place and return X."
  (when (vectorp a) (error "SCALE! needs a scalar multiplier"))
  (when (and (null start) (null end) (null stride))
    (let ((type (vector-type x)))
      (when (and (not (complex-type-p type)) (typep a (second (numeric-type type))))
        (dispatch-scale x a (length x) 0)
        (return-from scale! x))))
  (multiple-value-bind (type count offsets strides)
      (resolve-bulk-operands x (list a) (list nil) start end nil nil stride)
    (let ((offset (first offsets)))
      (with-gathered ((x offset (first strides) :in-out)) count
        (if (complex-type-p type)
            (complex-scale x a count offset)
            (dispatch-scale x a count offset)))))
  x)

(defun dispatch-axpy (y a x count y-offset x-offset)
  (ecase *backend*
    (:sbcl (sbcl-axpy y a x count y-offset x-offset))
    (:native (native-axpy y a x count y-offset x-offset))
    (:lisp (lisp-axpy y a x count y-offset x-offset))))

(defun axpy! (y a x &key start end y-start x-start stride y-stride x-stride)
  "Set Y to A*X+Y in place with separate multiplication and addition."
  (when (vectorp a) (error "AXPY! needs a scalar multiplier"))
  (unless (vectorp x) (error "AXPY! needs a vector input"))
  (when (and (null start) (null end) (null y-start) (null x-start)
             (null stride) (null y-stride) (null x-stride))
    (let ((type (vector-type y)))
      (when (and (not (complex-type-p type)) (eq type (vector-type x))
                 (= (length y) (length x))
                 (typep a (second (numeric-type type))))
        (dispatch-axpy y a x (length y) 0 0)
        (return-from axpy! y))))
  (multiple-value-bind (type count offsets strides)
      (resolve-bulk-operands y (list a x) (list nil x-start) start end y-start
                             (list nil x-stride) stride y-stride)
    (destructuring-bind (y-offset scalar-offset x-offset) offsets
      (declare (ignore scalar-offset))
      (with-gathered ((y y-offset (first strides) :in-out)
                      (x x-offset (third strides)))
          count
        (if (complex-type-p type)
            (complex-axpy y a x count y-offset x-offset)
            (dispatch-axpy y a x count y-offset x-offset)))))
  y)

(defun fma! (destination x y z &key start end destination-start x-start y-start z-start
                                   stride destination-stride x-stride y-stride z-stride)
  "Set DESTINATION to fused X*Y+Z elementwise and return it."
  (unless (member (vector-type destination) '(:f32 :f64))
    (error "FMA requires float vectors"))
  (multiple-value-bind (type count offsets strides)
      (resolve-bulk-operands destination (list x y z) (list x-start y-start z-start)
                             start end destination-start
                             (list x-stride y-stride z-stride) stride destination-stride)
    (declare (ignore type))
    (destructuring-bind (d-offset x-offset y-offset z-offset) offsets
      (destructuring-bind (&optional d-stride x-stride y-stride z-stride) strides
        (with-gathered ((destination d-offset d-stride :out)
                        (x x-offset x-stride) (y y-offset y-stride) (z z-offset z-stride))
            count
          (ecase *backend*
            (:sbcl (sbcl-bulk-fma destination x y z count d-offset x-offset y-offset z-offset))
            (:native (native-bulk-fma destination x y z count d-offset x-offset y-offset z-offset))
            (:lisp (lisp-bulk-fma destination x y z count d-offset x-offset y-offset z-offset)))))))
  destination)

(declaim (ftype function lisp-sum-wide lisp-dot-wide))

(defun wide-accumulation-p (type accumulate)
  "True when :ACCUMULATE widens TYPE's partial sums to double precision."
  (cond ((null accumulate) nil)
        ((not (eq accumulate :f64))
         (error "Unknown :ACCUMULATE ~S; only :F64 is supported" accumulate))
        ((integer-type-p type)
         (error ":ACCUMULATE does not apply to integer vectors"))
        (t (and (member type '(:f32 :c32)) t))))

(defun dispatch-sum (input count offset)
  (ecase *backend*
    (:sbcl (sbcl-sum input count offset))
    (:native (native-sum input count offset))
    (:lisp (lisp-sum input count offset))))

(defun sum (input &key start end input-start accumulate stride input-stride)
  "Return the sum of the :START to :END slice of INPUT at :STRIDE, or a zero if empty.
With :ACCUMULATE :F64, single-float and complex single-float sums accumulate and
return double precision."
  (when (and (null start) (null end) (null input-start) (null accumulate)
             (null stride) (null input-stride)
             (not (complex-type-p (vector-type input))))
    (return-from sum (dispatch-sum input (length input) 0)))
  (multiple-value-bind (type count offsets strides)
      (resolve-slice (list input) (list input-start) start end (list input-stride) stride)
    (let ((offset (first offsets))
          (wide (wide-accumulation-p type accumulate)))
      (with-gathered ((input offset (first strides))) count
        (cond ((complex-type-p type) (complex-sum input count offset wide))
              (wide (if (eq *backend* :native)
                        (native-sum-acc input count offset)
                        (lisp-sum-wide input count offset)))
              (t (dispatch-sum input count offset)))))))

(defun dispatch-dot (left right count left-offset right-offset)
  (ecase *backend*
    (:sbcl (sbcl-dot left right count left-offset right-offset))
    (:native (native-dot left right count left-offset right-offset))
    (:lisp (lisp-dot left right count left-offset right-offset))))

(defun dot (left right &key start end left-start right-start accumulate
                         stride left-stride right-stride)
  "Return the dot product of the :START to :END slices of LEFT and RIGHT at :STRIDE.
With :ACCUMULATE :F64, single-float and complex single-float products accumulate
and return double precision."
  (when (and (null start) (null end) (null left-start) (null right-start)
             (null accumulate) (null stride) (null left-stride) (null right-stride))
    (let ((type (vector-type left)))
      (when (and (not (complex-type-p type)) (eq type (vector-type right))
                 (= (length left) (length right)))
        (return-from dot (dispatch-dot left right (length left) 0 0)))))
  (multiple-value-bind (type count offsets strides)
      (resolve-slice (list left right) (list left-start right-start) start end
                     (list left-stride right-stride) stride)
    (destructuring-bind (left-offset right-offset) offsets
      (let ((wide (wide-accumulation-p type accumulate)))
        (with-gathered ((left left-offset (first strides))
                        (right right-offset (second strides)))
            count
          (cond ((complex-type-p type)
                 (complex-dot left right count left-offset right-offset nil wide))
                (wide (if (eq *backend* :native)
                          (native-dot-acc left right count left-offset right-offset)
                          (lisp-dot-wide left right count left-offset right-offset)))
                (t (dispatch-dot left right count left-offset right-offset))))))))

(defun dotc (left right &key start end left-start right-start accumulate
                          stride left-stride right-stride)
  "Return the dot product with LEFT conjugated."
  (multiple-value-bind (type count offsets strides)
      (resolve-slice (list left right) (list left-start right-start) start end
                     (list left-stride right-stride) stride)
    (unless (complex-type-p type)
      (error "DOTC requires complex vectors"))
    (destructuring-bind (left-offset right-offset) offsets
      (with-gathered ((left left-offset (first strides))
                      (right right-offset (second strides)))
          count
        (complex-dot left right count left-offset right-offset t
                     (wide-accumulation-p type accumulate))))))
