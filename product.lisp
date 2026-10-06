(in-package #:trivial-simd)

(define-kernel %product (input) (prod input))

(defun prod (input &key start end input-start stride input-stride)
  "Return the product of the logical input slice, or typed one when empty."
  (multiple-value-bind (type count offsets strides)
      (resolve-slice (list input) (list input-start) start end (list input-stride) stride)
    (let ((offset (first offsets)))
      (with-staged ((input offset (first strides)))
          (count :direct (and (eq *backend* :native) *native-row-reducers-p*
                              (not (complex-type-p type)))
                 :combine (product-combiner type))
        (%product input :input-start offset :end count)))))
