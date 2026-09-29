(in-package #:trivial-simd)

(defmacro define-integer-arithmetic ()
  `(progn
     ,@(loop for (key element foreign bits signed) in *numeric-types*
             when (integer-type-p key)
             append
             (let* ((mask (1- (ash 1 bits)))
                    (sign (ash 1 (1- bits)))
                    (modulus (ash 1 bits))
                    (wrap (integer-operation-symbol :wrap key)))
               `((declaim (inline ,wrap))
                 (defun ,wrap (value)
                   (let ((value (logand value ,mask)))
                     ,(if signed `(if (>= value ,sign) (- value ,modulus) value) 'value)))
                 ,@(loop for (operation operator arity)
                         in '((:add + 2) (:subtract - 2) (:multiply * 2)
                              (:divide truncate 2) (:negate - 1) (:abs abs 1)
                              (:min min 2) (:max max 2))
                         for name = (integer-operation-symbol operation key)
                         collect `(progn
                                    (declaim (inline ,name))
                                    (defun ,name (a ,@(when (= arity 2) '(b)))
                                      (declare (type ,element a ,@(when (= arity 2) '(b))))
                                      ,@(when (eq operation :divide)
                                          `((when (zerop b)
                                              (error 'division-by-zero :operation 'truncate
                                                     :operands (list a b)))))
                                      (,wrap (,operator a ,@(when (= arity 2) '(b))))))))))))

(define-integer-arithmetic)

(defmacro define-integer-loops ()
  `(progn
     (defun integer-binary (type operation destination left right count
                           destination-offset left-offset right-offset)
       (ecase type
         ,@(loop for (key element) in *numeric-types*
                 when (integer-type-p key)
                 collect
                 `(,key
                   (locally
                       (declare (type (simple-array ,element (*)) destination left right))
                     (ecase operation
                       ,@(loop for operation in '(:add :subtract :multiply :divide)
                               collect
                               `(,operation
                                 (dotimes (i count)
                                   (setf (aref destination (+ destination-offset i))
                                         (,(integer-operation-symbol operation key)
                                          (aref left (+ left-offset i))
                                          (aref right (+ right-offset i))))))))))))
       destination)
     (defun integer-sum (type input count offset)
       (ecase type
         ,@(loop for (key element) in *numeric-types*
                 when (integer-type-p key)
                 collect
                 `(,key
                   (locally
                       (declare (type (simple-array ,element (*)) input))
                     (let ((result 0))
                       (declare (type ,element result))
                       (dotimes (i count result)
                         (setf result (,(integer-operation-symbol :add key)
                                       result (aref input (+ offset i)))))))))))
     (defun integer-dot (type left right count left-offset right-offset)
       (ecase type
         ,@(loop for (key element) in *numeric-types*
                 when (integer-type-p key)
                 collect
                 `(,key
                   (locally
                       (declare (type (simple-array ,element (*)) left right))
                     (let ((result 0))
                       (declare (type ,element result))
                       (dotimes (i count result)
                         (setf result (,(integer-operation-symbol :add key)
                                       result (,(integer-operation-symbol :multiply key)
                                               (aref left (+ left-offset i))
                                               (aref right (+ right-offset i))))))))))))))

(define-integer-loops)
