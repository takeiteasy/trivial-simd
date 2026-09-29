(in-package #:trivial-simd)

(defun sbcl-integer-pack (type)
  (let* ((bits (fourth (numeric-type type)))
         (width (/ 128 bits))
         (prefix (format nil "~A~D.~D" (if (fifth (numeric-type type)) "S" "U") bits width))
         (package "SB-SIMD-SSE2"))
    (values package prefix width)))

(defun sbcl-integer-packable-p (tree type)
  (multiple-value-bind (package prefix) (sbcl-integer-pack type)
    (labels ((walk (node)
               (case (first node)
                 ((:argument :constant) t)
                 (:negate (walk (second node)))
                 (:operation
                  (and (member (second node) '(:add :subtract :multiply))
                       (let ((symbol (find-symbol
                                      (concatenate 'string prefix
                                                   (case (second node)
                                                     (:add "+") (:subtract "-")
                                                     (:multiply "*"))) package)))
                         (and symbol (fboundp symbol)))
                       (walk (third node)) (walk (fourth node)))))))
      (walk tree))))

(defun sbcl-integer-kernel-form (tree type destination arguments d-offset offsets count)
  (if (sbcl-integer-packable-p tree type)
      (sbcl-kernel-form tree type destination arguments d-offset offsets count)
      (lisp-kernel-form tree type destination arguments d-offset offsets count)))

(defmacro define-sbcl-integer-loops ()
  `(progn
     ,@(loop for (key element) in *numeric-types*
             when (integer-type-p key)
             collect
             (multiple-value-bind (package prefix width) (sbcl-integer-pack key)
               (let ((aref (find-symbol (concatenate 'string prefix "-AREF") package))
                     (add (find-symbol (concatenate 'string prefix "+") package))
                     (subtract (find-symbol (concatenate 'string prefix "-") package))
                     (binary (intern (format nil "%SBCL-INTEGER-BINARY-~A" key)))
                     (sum (intern (format nil "%SBCL-INTEGER-SUM-~A" key))))
                 `(progn
                    (defun ,binary (operation destination left right count
                                     destination-offset left-offset right-offset)
                      (declare (type (simple-array ,element (*)) destination left right)
                               (type fixnum count destination-offset left-offset right-offset))
                      (let ((i 0))
                        (declare (type fixnum i))
                        (when (member operation '(:add :subtract))
                          (loop while (<= (+ i ,width) count) do
                            (setf (,aref destination (+ destination-offset i))
                                  (if (eq operation :add)
                                      (,add (,aref left (+ left-offset i))
                                            (,aref right (+ right-offset i)))
                                      (,subtract (,aref left (+ left-offset i))
                                                 (,aref right (+ right-offset i)))))
                            (incf i ,width)))
                        (loop while (< i count) do
                          (setf (aref destination (+ destination-offset i))
                                (funcall (ecase operation
                                           (:add #',(integer-operation-symbol :add key))
                                           (:subtract #',(integer-operation-symbol :subtract key))
                                           (:multiply #',(integer-operation-symbol :multiply key))
                                           (:divide #',(integer-operation-symbol :divide key)))
                                         (aref left (+ left-offset i))
                                         (aref right (+ right-offset i))))
                          (incf i)))
                      destination)
                    (defun ,sum (input count offset)
                      (declare (type (simple-array ,element (*)) input)
                               (type fixnum count offset))
                      (let ((i 0) (result 0)
                            (pack (,(find-symbol prefix package) 0)))
                        (declare (type fixnum i) (type ,element result)
                                 (type ,(find-symbol prefix package) pack))
                        (loop while (<= (+ i ,width) count) do
                          (setf pack (,add pack (,aref input (+ offset i))))
                          (incf i ,width))
                        (multiple-value-call
                            (lambda (&rest values)
                              (dolist (value values)
                                (setf result (,(integer-operation-symbol :add key) result value))))
                          (,(find-symbol (concatenate 'string prefix "-VALUES") package) pack))
                        (loop while (< i count) do
                          (setf result (,(integer-operation-symbol :add key)
                                        result (aref input (+ offset i))))
                          (incf i))
                        result))))))))

(define-sbcl-integer-loops)

(defun sbcl-integer-binary (type operation destination left right count destination-offset left-offset right-offset)
  (funcall (ecase type
             (:s8 #'%sbcl-integer-binary-s8)
             (:u8 #'%sbcl-integer-binary-u8)
             (:s16 #'%sbcl-integer-binary-s16)
             (:u16 #'%sbcl-integer-binary-u16)
             (:s32 #'%sbcl-integer-binary-s32)
             (:u32 #'%sbcl-integer-binary-u32)
             (:s64 #'%sbcl-integer-binary-s64)
             (:u64 #'%sbcl-integer-binary-u64))
           operation destination left right count destination-offset left-offset right-offset))

(defun sbcl-integer-sum (type input count offset)
  (funcall (ecase type
             (:s8 #'%sbcl-integer-sum-s8)
             (:u8 #'%sbcl-integer-sum-u8)
             (:s16 #'%sbcl-integer-sum-s16)
             (:u16 #'%sbcl-integer-sum-u16)
             (:s32 #'%sbcl-integer-sum-s32)
             (:u32 #'%sbcl-integer-sum-u32)
             (:s64 #'%sbcl-integer-sum-s64)
             (:u64 #'%sbcl-integer-sum-u64))
           input count offset))
