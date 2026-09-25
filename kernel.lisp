(in-package #:trivial-simd)

(defconstant +kernel-registers+ 8)

(defparameter *kernel-opcodes*
  '((:copy . 0) (:constant . 1) (:add . 2) (:subtract . 3)
    (:multiply . 4) (:divide . 5) (:negate . 6)))

(defconstant +kernel-output+ #xFF)
(defconstant +kernel-input-base+ +kernel-registers+)
(defconstant +kernel-max-arguments+ (- +kernel-output+ +kernel-input-base+))

(defparameter *kernel-operators*
  '((+ . :add) (- . :subtract) (* . :multiply) (/ . :divide)))

(defun parse-kernel-expression (expression arguments)
  "Parse EXPRESSION into (:argument i), (:constant x), (:negate node), or
(:operation kind left right)."
  (cond ((and (symbolp expression) (position expression arguments))
         (list :argument (position expression arguments)))
        ((realp expression) (list :constant expression))
        ((and (consp expression) (assoc (first expression) *kernel-operators*))
         (let ((kind (cdr (assoc (first expression) *kernel-operators*)))
               (operands (mapcar (lambda (operand)
                                   (parse-kernel-expression operand arguments))
                                 (rest expression))))
           (cond ((null operands)
                  (error "Kernel operator ~S needs an operand" (first expression)))
                 ((rest operands)
                  (reduce (lambda (left right) (list :operation kind left right))
                          operands))
                 ((eq kind :subtract) (list :negate (first operands)))
                 ((eq kind :divide)
                  (list :operation :divide (list :constant 1) (first operands)))
                 (t (first operands)))))
        (t (error "Unsupported kernel expression ~S" expression))))

(defun register-need (node)
  (ecase (first node)
    (:argument 0)
    (:constant 1)
    (:negate (max 1 (register-need (second node))))
    (:operation (let ((left (register-need (third node)))
                      (right (register-need (fourth node))))
                  (max 1 (if (= left right) (1+ left) (max left right)))))))

;; TODO: expressions needing more registers are rejected; spill to memory (ticket #35).
(defun lower-kernel (tree)
  "Return (values instructions constants). Each instruction is (op dst a b);
operands below +KERNEL-INPUT-BASE+ are registers, the rest are inputs."
  (let ((free (loop for register below +kernel-registers+ collect register))
        (instructions '())
        (constants '()))
    (labels ((allocate ()
               (or (pop free)
                   (error "Kernel expression needs more than ~D registers"
                          +kernel-registers+)))
             (register-p (operand) (< operand +kernel-input-base+))
             (release (operand) (when (register-p operand) (push operand free)))
             (emit (op destination a b)
               (push (list op destination a b) instructions))
             (constant-index (value)
               (or (position value constants)
                   (progn (setf constants (append constants (list value)))
                          (1- (length constants)))))
             (destination-for (top-p &rest operands)
               (cond (top-p +kernel-output+)
                     ((find-if #'register-p operands))
                     (t (allocate))))
             (generate (node top-p)
               (ecase (first node)
                 (:argument
                  (let ((operand (+ +kernel-input-base+ (second node))))
                    (if top-p
                        (progn (emit :copy +kernel-output+ operand 0) +kernel-output+)
                        operand)))
                 (:constant
                  (let ((destination (destination-for top-p)))
                    (emit :constant destination (constant-index (second node)) 0)
                    destination))
                 (:negate
                  (let* ((operand (generate (second node) nil))
                         (destination (destination-for top-p operand)))
                    (emit :negate destination operand 0)
                    (unless (eql destination operand) (release operand))
                    destination))
                 (:operation
                  (destructuring-bind (kind left right) (rest node)
                    (let (left-operand right-operand)
                      (if (>= (register-need left) (register-need right))
                          (setf left-operand (generate left nil)
                                right-operand (generate right nil))
                          (setf right-operand (generate right nil)
                                left-operand (generate left nil)))
                      (let ((destination (destination-for top-p left-operand right-operand)))
                        (emit kind destination left-operand right-operand)
                        (dolist (operand (list left-operand right-operand))
                          (unless (eql operand destination) (release operand)))
                        destination)))))))
      (generate tree t))
    (values (nreverse instructions) constants)))

(defun kernel-bytes (instructions)
  (loop for (op destination a b) in instructions
        append (list (cdr (assoc op *kernel-opcodes*)) destination a b)))

(defun kernel-element-type (type)
  (ecase type (:f32 'single-float) (:f64 'double-float)))

(defun scalar-kernel-form (node type arguments offsets index)
  (let ((element (kernel-element-type type)))
    (labels ((walk (node)
               (ecase (first node)
                 (:argument `(aref ,(nth (second node) arguments)
                                   (+ ,(nth (second node) offsets) ,index)))
                 (:constant (coerce (second node) element))
                 (:negate `(- ,(walk (second node))))
                 (:operation
                  `(,(car (rassoc (second node) *kernel-operators*))
                    ,(walk (third node)) ,(walk (fourth node)))))))
      (walk node))))

(defun lisp-kernel-form (tree type destination arguments d-offset offsets count)
  (let ((index (gensym "I"))
        (variables (cons destination arguments)))
    `(let ,(mapcar (lambda (variable) (list variable variable)) variables)
       (declare (type (simple-array ,(kernel-element-type type) (*)) ,@variables)
                (type fixnum ,count ,d-offset ,@offsets))
       (dotimes (,index ,count)
         (setf (aref ,destination (+ ,d-offset ,index))
               ,(scalar-kernel-form tree type arguments offsets index))))))

#+(and sbcl x86-64)
(defun sbcl-kernel-form (tree type destination arguments d-offset offsets count)
  (destructuring-bind (package width prefix)
      (ecase type (:f32 '("SB-SIMD-SSE" 4 "F32.4")) (:f64 '("SB-SIMD-SSE2" 2 "F64.2")))
    (flet ((symbol-for (suffix)
             (or (find-symbol (concatenate 'string prefix suffix) package)
                 (error "sb-simd symbol ~A~A is missing" prefix suffix))))
      (let ((index (gensym "I"))
            (variables (cons destination arguments))
            (aref (symbol-for "-AREF"))
            (cast (symbol-for "")))
        (labels ((walk (node)
                   (ecase (first node)
                     (:argument `(,aref ,(nth (second node) arguments)
                                        (+ ,(nth (second node) offsets) ,index)))
                     (:constant (coerce (second node) (kernel-element-type type)))
                     (:negate `(,(symbol-for "-") (,cast 0) ,(walk (second node))))
                     (:operation
                      `(,(symbol-for (string (car (rassoc (second node) *kernel-operators*))))
                        ,(walk (third node)) ,(walk (fourth node)))))))
          `(let ,(mapcar (lambda (variable) (list variable variable)) variables)
             (declare (type (simple-array ,(kernel-element-type type) (*)) ,@variables)
                      (type fixnum ,count ,d-offset ,@offsets))
             (let ((,index 0))
               (declare (type fixnum ,index))
               (loop while (<= (+ ,index ,width) ,count) do
                 (setf (,aref ,destination (+ ,d-offset ,index)) (,cast ,(walk tree)))
                 (incf ,index ,width))
               (loop while (< ,index ,count) do
                 (setf (aref ,destination (+ ,d-offset ,index))
                       ,(scalar-kernel-form tree type arguments offsets index))
                 (incf ,index)))))))))

(defun native-kernel-form (program type destination arguments d-offset offsets count)
  (let ((pointers (loop for nil in arguments collect (gensym "POINTER")))
        (output (gensym "OUTPUT"))
        (table (gensym "TABLE"))
        (foreign (gensym "FOREIGN")))
    `(let ((,foreign (if (eq ,type :f32) :float :double)))
       (with-native-vectors (,foreign ((,output ,destination)
                                       ,@(mapcar #'list pointers arguments))
                             :outputs (,output))
         (cffi:with-foreign-object (,table :pointer ,(max 1 (length arguments)))
           ,@(loop for pointer in pointers
                   for offset in offsets
                   for index from 0
                   collect `(setf (cffi:mem-aref ,table :pointer ,index)
                                  (element-pointer ,pointer ,foreign ,offset)))
           (call-native-kernel ,program ,foreign ,table
                               (element-pointer ,output ,foreign ,d-offset)
                               ,count))))))

(defmacro define-kernel (name (&rest arguments) expression)
  "Define NAME as (NAME destination ,@ARGUMENTS &key start end ...) computing
EXPRESSION elementwise from ARGUMENTS into DESTINATION. Experimental."
  (when (> (length arguments) +kernel-max-arguments+)
    (error "A kernel takes at most ~D arguments" +kernel-max-arguments+))
  (let ((tree (parse-kernel-expression expression arguments)))
    (multiple-value-bind (instructions constants) (lower-kernel tree)
      (let* ((destination (gensym "DESTINATION"))
             (count (gensym "COUNT"))
             (d-offset (gensym "D-OFFSET"))
             (offsets (loop for nil in arguments collect (gensym "OFFSET")))
             (program (gensym "PROGRAM"))
             (type (gensym "TYPE"))
             (offsets-variable (gensym "OFFSETS"))
             (starts (mapcar (lambda (argument)
                               (intern (format nil "~A-START" argument)))
                             arguments))
             (bytes (kernel-bytes instructions)))
        `(let ((,program nil))
           (defun ,name (,destination ,@arguments
                         &key start end destination-start ,@starts)
             (multiple-value-bind (,type ,count ,offsets-variable)
                 (resolve-slice (list ,destination ,@arguments)
                                (list destination-start ,@starts) start end)
               (destructuring-bind (,d-offset ,@offsets) ,offsets-variable
                 (declare (type fixnum ,d-offset ,@offsets))
                 (ecase *backend*
                   (:lisp (if (eq ,type :f32)
                              ,(lisp-kernel-form tree :f32 destination arguments
                                                 d-offset offsets count)
                              ,(lisp-kernel-form tree :f64 destination arguments
                                                 d-offset offsets count)))
                   (:sbcl
                    ,(if (and (boundp '*sbcl-simd-available-p*) *sbcl-simd-available-p*)
                         #+(and sbcl x86-64)
                         `(if (eq ,type :f32)
                              ,(sbcl-kernel-form tree :f32 destination arguments
                                                 d-offset offsets count)
                              ,(sbcl-kernel-form tree :f64 destination arguments
                                                 d-offset offsets count))
                         #-(and sbcl x86-64)
                         '(error "SBCL SIMD is unavailable on this platform")
                         '(error "SBCL SIMD is unavailable on this platform")))
                   (:native
                    (let ((,program (or ,program
                                        (setf ,program
                                              (make-native-program ',bytes ',constants)))))
                      ,(native-kernel-form program type destination arguments
                                           d-offset offsets count))))))
             ,destination))))))
