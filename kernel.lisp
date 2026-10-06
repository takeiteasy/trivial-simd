(in-package #:trivial-simd)

(defconstant +kernel-registers+ 8)

(defparameter *kernel-opcodes*
  '((:copy . 0) (:constant . 1) (:add . 2) (:subtract . 3)
    (:multiply . 4) (:divide . 5) (:negate . 6) (:spill . 7) (:reload . 8)
    (:sqrt . 9) (:abs . 10) (:min . 11) (:max . 12) (:fma . 13)
    (:eq . 14) (:ne . 15) (:lt . 16) (:le . 17) (:gt . 18) (:ge . 19)
    (:select . 20) (:exp . 21) (:sin . 22) (:cos . 23)))

(defconstant +kernel-output+ #xFF)
(defconstant +kernel-input-base+ +kernel-registers+)
(defconstant +kernel-max-arguments+ (- +kernel-output+ +kernel-input-base+))

(defparameter *kernel-operators*
  '((+ . :add) (- . :subtract) (* . :multiply) (/ . :divide)
    (sqrt . :sqrt) (abs . :abs) (min . :min) (max . :max) (fma . :fma)
    (exp . :exp) (sin . :sin) (cos . :cos)))

(defun parse-kernel-expression (expression arguments)
  "Parse an elementwise expression into a typed operator tree."
  (cond ((and (symbolp expression) (position expression arguments))
         (list :argument (position expression arguments)))
        ((or (realp expression) (kernel-scalar-p expression)) (list :constant expression))
        ((and (consp expression) (assoc (first expression) *kernel-operators*))
         (let ((kind (cdr (assoc (first expression) *kernel-operators*)))
               (operands (mapcar (lambda (operand)
                                   (parse-kernel-expression operand arguments))
                                 (rest expression))))
           (cond ((null operands)
                  (error "Kernel operator ~S needs an operand" (first expression)))
                 ((member kind '(:sqrt :abs :exp :sin :cos))
                  (unless (= 1 (length operands))
                    (error "Kernel operator ~S needs one operand" (first expression)))
                  (list :unary kind (first operands)))
                 ((eq kind :fma)
                  (unless (= 3 (length operands)) (error "Kernel FMA needs three operands"))
                  (cons :fma operands))
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
    (:unary (max 1 (register-need (third node))))
    (:fma (loop for need in (sort (mapcar #'register-need (rest node)) #'>)
                for held from 0 maximize (max 3 (+ held need))))
    (:select (loop for need in (sort (mapcar #'register-need (rest node)) #'>)
                   for held from 0 maximize (max 3 (+ held need))))
    (:operation (let ((left (register-need (third node)))
                      (right (register-need (fourth node))))
                  (max 1 (if (= left right) (1+ left) (max left right)))))))

;; TODO: 16-bit scratch indexes cap live slots at 65,536; widen bytecode (#50).
(defun kernel-scratch-index (index)
  (unless (typep index '(unsigned-byte 16))
    (error "Kernel scratch index ~D exceeds the 65,536-slot limit" index))
  index)

(defun lower-kernel (tree)
  "Return instructions, constants, and scratch count. Instructions are (op dst a b).
SPILL/RELOAD use a register and slot; other operands name registers or inputs."
  (let ((free (loop for register below +kernel-registers+ collect register))
        (instructions '())
        (constants '())
        (scratch-free '())
        (scratch-count 0))
    (labels ((allocate ()
               (or (pop free)
                   (error "Internal kernel register allocation exhausted")))
             (register-p (operand) (< operand +kernel-input-base+))
             (release (operand) (when (register-p operand) (push operand free)))
             (emit (op destination a b)
               (push (list op destination a b) instructions))
             (scratch-slot ()
               (or (pop scratch-free)
                   (prog1 (kernel-scratch-index scratch-count)
                     (incf scratch-count))))
             (generate-pair (first second)
               (let* ((operand (generate first nil))
                      (slot (when (and (register-p operand)
                                       (> (register-need second) (length free)))
                              (let ((slot (scratch-slot)))
                                (emit :spill operand slot 0)
                                (release operand)
                                slot)))
                      (other (generate second nil)))
                 (when slot
                   (setf operand (allocate))
                   (emit :reload operand slot 0)
                   (push slot scratch-free))
                 (values operand other)))
             (generate-many (nodes)
               (let ((operands (make-array (length nodes))) (held '()) (spilled '()))
                 (dolist (index (stable-sort (loop for i below (length nodes) collect i)
                                            #'> :key (lambda (i) (register-need (nth i nodes)))))
                   (loop while (and held (> (register-need (nth index nodes)) (length free)))
                         for previous = (pop held)
                         do (let ((slot (scratch-slot)))
                              (emit :spill (aref operands previous) slot 0)
                              (release (aref operands previous))
                              (push (cons previous slot) spilled)))
                   (setf (aref operands index) (generate (nth index nodes) nil))
                   (when (register-p (aref operands index)) (push index held)))
                 (dolist (spill spilled)
                   (let ((register (allocate)))
                     (emit :reload register (cdr spill) 0)
                     (setf (aref operands (car spill)) register)
                     (push (cdr spill) scratch-free)))
                 (coerce operands 'list)))
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
                 (:unary
                  (let* ((operand (generate (third node) nil))
                         (destination (destination-for top-p operand)))
                    (emit (second node) destination operand 0)
                    (unless (eql destination operand) (release operand))
                    destination))
                 (:fma
                  (destructuring-bind (a b c) (generate-many (rest node))
                    (let ((destination (if (register-p c) c (allocate))))
                      (unless (eql destination c) (emit :copy destination c 0))
                      (emit :fma destination a b)
                      (dolist (operand (list a b)) (release operand))
                      (if top-p
                          (progn (emit :copy +kernel-output+ destination 0)
                                 (release destination) +kernel-output+)
                          destination))))
                 (:select
                  (destructuring-bind (mask on-true on-false) (generate-many (rest node))
                    (let ((destination (if (register-p on-false) on-false (allocate))))
                      (unless (eql destination on-false)
                        (emit :copy destination on-false 0))
                      (emit :select destination mask on-true)
                      (dolist (operand (list mask on-true))
                        (unless (eql operand destination) (release operand)))
                      (if top-p
                          (progn (emit :copy +kernel-output+ destination 0)
                                 (release destination) +kernel-output+)
                          destination))))
                 (:operation
                  (destructuring-bind (kind left right) (rest node)
                    (let (left-operand right-operand)
                      (if (>= (register-need left) (register-need right))
                          (multiple-value-setq (left-operand right-operand)
                            (generate-pair left right))
                          (multiple-value-setq (right-operand left-operand)
                            (generate-pair right left)))
                      (let ((destination (destination-for top-p left-operand right-operand)))
                        (emit kind destination left-operand right-operand)
                        (dolist (operand (list left-operand right-operand))
                          (unless (eql operand destination) (release operand)))
                        destination)))))))
      (generate tree t))
    (values (nreverse instructions) constants scratch-count)))

(defun kernel-bytes (instructions)
  (loop for (op destination a b) in instructions
        append (if (member op '(:spill :reload))
                   (let ((index (kernel-scratch-index a)))
                     (list (cdr (assoc op *kernel-opcodes*)) destination
                           (ldb (byte 8 0) index) (ldb (byte 8 8) index)))
                   (list (cdr (assoc op *kernel-opcodes*)) destination a b))))

(defun kernel-element-type (type)
  (second (numeric-type type)))

(defun scalar-kernel-form (node type arguments offsets index &optional (fma-operator 'fma))
  (let ((element (kernel-element-type type))
        (inputs (loop for nil in arguments collect (gensym "ELEMENT")))
        (temporaries '()) (free '()) (assignments '()))
    (labels ((allocate ()
               (or (pop free)
                   (let ((name (gensym "VALUE"))) (push name temporaries) name)))
             (operation (operator operands)
               (let ((result (or (find-if (lambda (operand) (member operand temporaries)) operands)
                                 (allocate))))
                 (let ((operator (if (integer-type-p type)
                                     (integer-operation-symbol
                                      (case operator
                                        (kernel-min :min) (kernel-max :max)
                                        (- (if (= (length operands) 1) :negate :subtract))
                                        (t (cdr (assoc operator *kernel-operators*)))) type)
                                     operator)))
                   (push `(setf ,result (,operator ,@operands)) assignments))
                 (dolist (operand operands)
                   (when (and (member operand temporaries) (not (eq operand result)))
                     (push operand free)))
                 result))
             (walk (node)
               (ecase (first node)
                 (:argument (nth (second node) inputs))
                 (:constant (kernel-constant-form (second node) type))
                 (:negate (operation '- (list (walk (second node)))))
                 (:unary (operation (if (eq (second node) :sqrt) 'kernel-sqrt
                                     (car (rassoc (second node) *kernel-operators*)))
                                    (list (walk (third node)))))
                 (:fma (operation fma-operator (mapcar #'walk (rest node))))
                 (:operation
                  (operation (case (second node)
                               (:min 'kernel-min) (:max 'kernel-max)
                               (t (car (rassoc (second node) *kernel-operators*))))
                             (list (walk (third node)) (walk (fourth node))))))))
      (let ((result (walk node)))
        `(let ,(loop for input in inputs for argument in arguments for offset in offsets
                     collect `(,input (aref ,argument (+ ,offset ,index))))
           ,@(when inputs `((declare (type ,element ,@inputs) (ignorable ,@inputs))))
           (let ,(loop for temporary in temporaries collect `(,temporary ,(coerce 0 element)))
             ,@(when temporaries `((declare (type ,element ,@temporaries))))
             ,@(nreverse assignments)
             ,result))))))

(defun reduction-loop-form (reducer type count index value)
  "Scalar loop reducing COUNT values; VALUE computes the one at INDEX.
Evaluates to the reduction, or for ARGMIN and ARGMAX the index within the slice.
Empty slices give NIL for the extrema and a zero otherwise."
  (let* ((element (kernel-element-type type))
         (real (case type (:c32 'single-float) (:c64 'double-float) (t element)))
         (integerp (integer-type-p type))
         (sum (gensym "SUM")) (best (gensym "BEST"))
         (best-index (gensym "BEST-INDEX")) (current (gensym "VALUE")))
    (ecase reducer
      ((sum asum)
       (let ((accumulator (if (eq reducer 'asum) real element))
             (term (cond ((eq reducer 'sum) value)
                         (integerp `(,(integer-operation-symbol :abs type) ,value))
                         (t `(abs ,value)))))
         `(let ((,sum ,(coerce 0 accumulator)))
            (declare (type ,accumulator ,sum))
            (dotimes (,index ,count ,sum)
              ,(if integerp
                   `(setf ,sum (,(integer-operation-symbol :add type) ,sum ,term))
                   `(incf ,sum ,term))))))
      (prod
       `(let ((,sum ,(coerce 1 element)))
          (declare (type ,element ,sum))
          (dotimes (,index ,count ,sum)
            (setf ,sum ,(if integerp
                           `(,(integer-operation-symbol :multiply type) ,sum ,value)
                           `(* ,sum ,value))))))
      (nrm2
       (when integerp (error "NRM2 requires float or complex vectors"))
       (let* ((real-part (gensym "REAL")) (imaginary-part (gensym "IMAGINARY"))
              (square (if (complex-type-p type)
                          `(let* ((,current ,value)
                                  (,real-part (coerce (realpart ,current) 'double-float))
                                  (,imaginary-part (coerce (imagpart ,current) 'double-float)))
                             (+ (* ,real-part ,real-part) (* ,imaginary-part ,imaginary-part)))
                          `(let ((,current (coerce ,value 'double-float)))
                             (* ,current ,current)))))
         `(let ((,sum 0d0))
            (declare (type double-float ,sum))
            ,(if (member type '(:f64 :c64))
                 `(if (and (handler-case (progn (dotimes (,index ,count) (incf ,sum ,square)) t)
                             (arithmetic-error () nil))
                           (<= +nrm2-fast-lower+ ,sum most-positive-double-float))
                      (sqrt ,sum)
                      (scaled-norm-function ,count (lambda (,index) ,value)))
                 `(progn (dotimes (,index ,count) (incf ,sum ,square))
                         (coerce (sqrt ,sum) 'single-float))))))
      ((minimum maximum argmin argmax)
       `(if (zerop ,count)
            nil
            (let ((,best (let ((,index 0))
                           (declare (type fixnum ,index))
                           ,value))
                  (,best-index 0))
              (declare (type ,element ,best) (type fixnum ,best-index))
              (loop for ,index of-type fixnum from 1 below ,count
                    do (let ((,current ,value))
                         (declare (type ,element ,current))
                         (when (,(if (member reducer '(minimum argmin)) '< '>) ,current ,best)
                           (setf ,best ,current ,best-index ,index))))
              ,(if (member reducer '(minimum maximum)) best best-index)))))))

(defun lisp-kernel-form (tree type destination arguments d-offset offsets count
                       &optional (fma-operator 'fma) reducer)
  (let ((index (gensym "I"))
        (element (kernel-element-type type))
        (variables (if destination (cons destination arguments) arguments)))
    `(let ,(mapcar (lambda (variable) (list variable variable)) variables)
       (declare (type (simple-array ,element (*)) ,@variables)
                (type fixnum ,count ,@(when destination (list d-offset)) ,@offsets))
       ,(if destination
            `(dotimes (,index ,count)
               (setf (aref ,destination (+ ,d-offset ,index))
                     ,(scalar-kernel-form tree type arguments offsets index fma-operator)))
            (reduction-loop-form
             (or reducer 'sum) type count index
             (scalar-kernel-form tree type arguments offsets index fma-operator))))))

(defun selected-lisp-kernel-form (tree type destination arguments d-offset offsets count
                                  &optional reducer)
  (let ((fallback (lisp-kernel-form tree type destination arguments d-offset offsets count
                                    'fma reducer)))
    (labels ((fma-p (node)
               (and (consp node) (or (eq (first node) :fma) (some #'fma-p node)))))
      (if (and *arm64-fma-compiler-p* (fma-p tree))
          `(if (arm64-fma-enabled-p ',(kernel-element-type type))
               ,(lisp-kernel-form tree type destination arguments d-offset offsets count
                                  (ecase type (:f32 'arm64-fma-f32) (:f64 'arm64-fma-f64))
                                  reducer)
               ,fallback)
          fallback))))

#+(and sbcl x86-64)
;; TODO: min/max/arg reducers use scalar loops (#92); f32 nrm2 too (#95).
(defun sbcl-kernel-form (tree type destination arguments d-offset offsets count &optional reducer)
  (when (or (kernel-transcendental-p tree)
            (member reducer '(minimum maximum argmin argmax prod))
            (and (eq reducer 'nrm2) (eq type :f32)))
    (return-from sbcl-kernel-form
      (lisp-kernel-form tree type destination arguments d-offset offsets count 'fma reducer)))
  (destructuring-bind (package width prefix)
      (if (integer-type-p type)
          (multiple-value-bind (package prefix width) (sbcl-integer-pack type)
            (list package width prefix))
          (ecase type (:f32 '("SB-SIMD-SSE" 4 "F32.4")) (:f64 '("SB-SIMD-SSE2" 2 "F64.2"))))
    (flet ((symbol-for (suffix)
             (or (find-symbol (concatenate 'string prefix suffix) package)
                 (error "sb-simd symbol ~A~A is missing" prefix suffix))))
      (let ((index (gensym "I"))
            (hardware-fma
              (when (and *sbcl-fma-f32* *sbcl-fma-f64*
                         (labels ((contains-fma (node)
                                    (and (consp node)
                                         (or (eq (first node) :fma)
                                             (some #'contains-fma (rest node))))))
                           (contains-fma tree)))
                (gensym "HARDWARE-FMA")))
            (variables (if destination (cons destination arguments) arguments))
            (aref (symbol-for "-AREF"))
            (cast (symbol-for ""))
            (assignments '())
            (temporaries '())
            (free '())
            (inputs (make-array (length arguments) :initial-element nil))
            (sum (gensym "SUM")) (sum-pack (gensym "SUM-PACK")))
        (labels ((allocate ()
                   (or (pop free)
                       (let ((name (gensym "PACK")))
                         (push name temporaries)
                         name)))
                 (emit (name form)
                   (push `(setf ,name ,form) assignments)
                   name)
                 (operation (operator operands)
                   (let ((result (or (find-if (lambda (operand) (member operand temporaries))
                                             operands)
                                     (allocate))))
                     (emit result (cons operator operands))
                     (dolist (operand operands)
                       (when (and (member operand temporaries) (not (eq operand result)))
                         (push operand free)))
                     result))
                 (pack-math (kind operands)
                   (let* ((result (or (find-if (lambda (operand) (member operand temporaries)) operands)
                                      (allocate)))
                          (packed (mapcar (lambda (operand) `(,cast ,operand)) operands))
                          (form
                            (cond
                              ((eq kind :abs) `(,(symbol-for "-ANDC1") (,cast ,(- (coerce 0 (kernel-element-type type))))
                                      ,(first packed)))
                              ((member kind '(:min :max)) `(,(symbol-for (if (eq kind :min) "-MIN" "-MAX"))
                                            ,(second packed) ,(first packed)))
                              (t
                               (let ((lanes (loop for nil in operands
                                                  collect (loop repeat width collect (gensym "LANE")))))
                                 (labels ((bind-lanes (remaining packs body)
                                            (if remaining
                                                `(multiple-value-bind ,(first remaining)
                                                     (,(symbol-for "-VALUES") ,(first packs))
                                                   ,(bind-lanes (rest remaining) (rest packs) body))
                                                body)))
                                   (bind-lanes
                                    lanes packed
                                    (if (eq kind :sqrt)
                                        `(progn
                                           (when (or ,@(mapcar (lambda (lane) `(minusp ,lane)) (first lanes)))
                                             (error "Negative kernel square root operand"))
                                           (,(symbol-for "-SQRT") ,(first packed)))
                                        `(,(find-symbol (concatenate 'string "MAKE-" prefix) package)
                                          ,@(loop for i below width
                                                  collect `(fma ,@(mapcar (lambda (group) (nth i group)) lanes))))))))))))
                     (emit result
                           (if (and (eq kind :fma) hardware-fma)
                               `(if ,hardware-fma
                                    (,(find-symbol (concatenate 'string prefix "-FMADD") "SB-SIMD-FMA")
                                     ,@packed)
                                    ,form)
                               form))
                     (dolist (operand operands)
                       (when (and (member operand temporaries) (not (eq operand result)))
                         (push operand free)))
                     result))
                 (walk (node)
                   (ecase (first node)
                     (:argument
                      (let ((argument (second node)))
                        (or (aref inputs argument)
                            (setf (aref inputs argument)
                                  (emit (gensym "INPUT")
                                        `(,aref ,(nth argument arguments)
                                                (+ ,(nth argument offsets) ,index)))))))
                     (:constant (kernel-constant-form (second node) type))
                     (:negate (operation (symbol-for "-") (list `(,cast 0) (walk (second node)))))
                     (:unary (pack-math (second node) (list (walk (third node)))))
                     (:fma (pack-math :fma (mapcar #'walk (rest node))))
                     (:operation
                      (if (member (second node) '(:min :max))
                          (pack-math (second node) (list (walk (third node)) (walk (fourth node))))
                          (operation (symbol-for (string (car (rassoc (second node) *kernel-operators*))))
                                 (list (walk (third node)) (walk (fourth node)))))))))
          ;; A long LET* also exhausts SBCL's compiler stack; use flat assignments.
          (let* ((result (walk tree))
                 (packs (append (remove nil (coerce inputs 'list)) temporaries))
                 (lanes (loop repeat width collect (gensym "LANE")))
                 (reducer (or reducer 'sum))
                 (square (gensym "SQUARE")) (term (gensym "TERM"))
                 (accumulated
                   (ecase reducer
                     (sum `(,cast ,result))
                     (asum `(,(symbol-for "-ANDC1")
                             (,cast ,(- (coerce 0 (kernel-element-type type)))) (,cast ,result)))
                     (nrm2 `(let ((,square (,cast ,result)))
                              (,(symbol-for "*") ,square ,square)))))
                 (tail-term
                   (let ((value (scalar-kernel-form tree type arguments offsets index)))
                     (ecase reducer
                       (sum value)
                       (asum `(abs ,value))
                       (nrm2 `(let ((,term ,value)) (* ,term ,term))))))
                 (accumulate
                   (unless destination
                     `((loop while (<= (+ ,index ,width) ,count) do
                         (let ,(loop for name in packs collect `(,name (,cast 0)))
                           ,@(when packs `((declare (type ,cast ,@packs))))
                           ,@(reverse assignments)
                           (setf ,sum-pack (,(symbol-for "+") ,sum-pack ,accumulated)))
                         (incf ,index ,width))
                       (multiple-value-bind ,lanes (,(symbol-for "-VALUES") ,sum-pack)
                         (setf ,sum ,(if (integer-type-p type)
                                         (reduce (lambda (value lane)
                                                   `(,(integer-operation-symbol :add type) ,value ,lane))
                                                 lanes :initial-value sum)
                                         `(+ ,@lanes))))
                       (loop while (< ,index ,count) do
                         ,(if (integer-type-p type)
                              `(setf ,sum (,(integer-operation-symbol :add type) ,sum ,tail-term))
                              `(incf ,sum ,tail-term))
                         (incf ,index))))))
            `(let ,(mapcar (lambda (variable) (list variable variable)) variables)
               (declare (type (simple-array ,(kernel-element-type type) (*)) ,@variables)
                        (type fixnum ,count ,@(when destination (list d-offset)) ,@offsets))
               (let ((,index 0)
                     ,@(when hardware-fma `((,hardware-fma (sbcl-fma-enabled-p))))
                     ,@(unless destination
                         `((,sum ,(coerce 0 (kernel-element-type type))) (,sum-pack (,cast 0)))))
                 (declare (type fixnum ,index)
                          ,@(unless destination
                              `((type ,(kernel-element-type type) ,sum) (type ,cast ,sum-pack))))
                 ,@(if destination
                       `((loop while (<= (+ ,index ,width) ,count) do
                           (let ,(loop for name in packs collect `(,name (,cast 0)))
                             ,@(when packs `((declare (type ,cast ,@packs))))
                             ,@(nreverse assignments)
                             (setf (,aref ,destination (+ ,d-offset ,index)) (,cast ,result)))
                           (incf ,index ,width))
                         (loop while (< ,index ,count) do
                           (setf (aref ,destination (+ ,d-offset ,index))
                                 ,(scalar-kernel-form tree type arguments offsets index))
                           (incf ,index)))
                       (if (eq reducer 'nrm2)
                           `((if (and (handler-case (progn ,@accumulate t)
                                        (arithmetic-error () nil))
                                      (<= +nrm2-fast-lower+ ,sum most-positive-double-float))
                                 (sqrt ,sum)
                                 (scaled-norm-function
                                  ,count (lambda (,index)
                                           ,(scalar-kernel-form tree type arguments offsets index)))))
                           `(,@accumulate ,sum)))))))))))

;; TODO: reducers other than sum evaluate a block, then reduce it; fuse into the VM (#96).
;; Reducer codes match ts_kernel_reduction_*.
(defun native-reduction-form (program foreign table count reducer input-count)
  (let ((value (gensym "VALUE")) (wide (gensym "WIDE")) (index (gensym "INDEX"))
        (scale (gensym "SCALE")) (sum (gensym "SUM"))
        (integerp (integer-type-p (first (find foreign *numeric-types* :key #'third)))))
    (flet ((call (code &optional (scale-form 0d0))
             `(call-native-kernel-reduction ,program ,foreign ,table ,input-count ,count
                                           ,code ,scale-form ,value ,wide ,index))
           (read-wide () `(cffi:mem-ref ,wide :double)))
      `(cffi:with-foreign-objects ((,value ,foreign) (,wide :double) (,index :size))
         ;; Each reducer uses a subset of the output cells; touching all avoids warnings.
         ,value ,wide ,index
         ,(ecase reducer
            ((minimum maximum)
             `(when (plusp ,count)
                ,(call (if (eq reducer 'minimum) 1 2))
                ,(if integerp
                     `(native-integer-result ,value ,foreign)
                     `(cffi:mem-ref ,value ,foreign))))
            ((argmin argmax)
             `(when (plusp ,count)
                ,(call (if (eq reducer 'argmin) 1 2))
                (cffi:mem-ref ,index :size)))
            (prod
             `(progn ,(call 7)
                     ,(if integerp
                          `(native-integer-result ,value ,foreign)
                          `(cffi:mem-ref ,value ,foreign))))
            (asum
             `(progn ,(call 3)
                     ,(if integerp
                          `(native-integer-result ,value ,foreign)
                          `(cffi:mem-ref ,value ,foreign))))
            (nrm2
             (if integerp
                 '(error "NRM2 requires float or complex vectors")
                 (if (eq foreign :double)
                 `(progn
                    ,(call 4)
                    (let ((,sum ,(read-wide)))
                      (if (<= +nrm2-fast-lower+ ,sum most-positive-double-float)
                          (sqrt ,sum)
                          (progn
                            ,(call 5)
                            (let ((,scale ,(read-wide)))
                              (if (zerop ,scale)
                                  0d0
                                  (progn ,(call 6 scale)
                                         (* ,scale (sqrt ,(read-wide))))))))))
                 `(progn ,(call 4)
                         (coerce (sqrt ,(read-wide)) 'single-float))))))))))

;; TODO: pointer/output setup dominates short kernels; specialize runners if worthwhile (#59).
(defun native-kernel-form (program foreign destination arguments d-offset offsets count
                           &optional reducer)
  (let ((pointers (loop for nil in arguments collect (gensym "POINTER")))
        (output (gensym "OUTPUT")) (table (gensym "TABLE")))
    `(with-native-vectors (,foreign (,@(when destination `((,output ,destination ,d-offset)))
                                    ,@(mapcar #'list pointers arguments offsets))
                          :outputs ,(when destination (list output))
                          :count ,count)
       (cffi:with-foreign-object (,table :pointer ,(max 1 (length arguments)))
         ,@(loop for pointer in pointers for index from 0
                 collect `(setf (cffi:mem-aref ,table :pointer ,index)
                                ,pointer))
         ,(cond (destination
                 `(call-native-kernel ,program ,foreign ,table
                                      ,output ,count))
                ((member reducer '(nil sum))
                 `(cffi:with-foreign-object (,output ,foreign)
                    (call-native-kernel ,program ,foreign ,table ,output ,count :sum-p t)
                    ,(if (integer-type-p (first (find foreign *numeric-types* :key #'third)))
                         `(native-integer-result ,output ,foreign)
                         `(cffi:mem-ref ,output ,foreign))))
                (t (native-reduction-form program foreign table count reducer
                                          (length arguments))))))))

#+ecl
(declaim (notinline compile-native-kernel-runner))
#+ecl
(defun compile-native-kernel-runner (form fallback &optional (compiler #'compile))
  (handler-case
      (let ((*compile-verbose* nil) (*compile-print* nil))
        (multiple-value-bind (function warnings failure) (funcall compiler nil form)
          (declare (ignore warnings))
          (if failure fallback function)))
    (error () fallback)))

#+ecl
(defvar *native-kernel-runners* (make-hash-table :test 'eql))

#+(and ecl threads)
;; TODO: cold compilation is serialized; evaluate per-signature locks (#65).
(defvar *native-kernel-runner-lock* (mp:make-lock :name "native kernel runners"))

#+ecl
(defun ensure-native-kernel-runner (program signature form fallback)
  (or (native-program-runner program)
      (flet ((initialize ()
               (or (native-program-runner program)
                   (let ((runner (or (gethash signature *native-kernel-runners*)
                                     (setf (gethash signature *native-kernel-runners*)
                                           (or (compile-native-kernel-runner form nil) :failed)))))
                     (setf (native-program-runner program)
                           (if (eq runner :failed) fallback runner))))))
        #+threads (mp:with-lock (*native-kernel-runner-lock*) (initialize))
        #-threads (initialize))))

(defun validate-integer-kernel (tree type &optional reducer)
  (when (eq reducer 'nrm2) (error "NRM2 requires float or complex vectors"))
  (let ((element (kernel-element-type type)))
    (labels ((walk (node)
               (case (first node)
                 (:constant
                  (unless (or (kernel-scalar-p (second node)) (typep (second node) element))
                    (error 'type-error :datum (second node) :expected-type element)))
                 (:argument nil)
                 (:fma (error "FMA requires float vectors"))
                 (:unary
                  (when (member (second node) '(:sqrt :exp :sin :cos))
                    (error "~A requires float vectors" (second node)))
                  (walk (third node)))
                 (:negate (walk (second node)))
                 (:operation (walk (third node)) (walk (fourth node))))))
      (walk tree))))

(defun validate-complex-kernel (tree &optional reducer)
  (when (member reducer '(minimum maximum argmin argmax))
    (error "Complex values have no ordering"))
  (labels ((walk (node)
             (case (first node)
               ((:argument :constant) nil)
               (:negate (walk (second node)))
               (:unary
                (when (member (second node) '(:abs :exp :sin :cos))
                  (error "~A requires real float vectors" (second node)))
                (walk (third node)))
               (:fma (error "FMA requires real float vectors"))
               (:select
                (walk (second node))
                (walk (third node))
                (walk (fourth node)))
               (:operation
                (when (member (second node) '(:min :max :lt :le :gt :ge))
                  (error "Complex values have no ordering"))
                (walk (third node))
                (walk (fourth node))))))
    (walk tree)))

(defun integer-kernel-function (cache tree type backend destination arguments count d-offset offsets
                                &optional reducer)
  (let ((slot (+ (* 2 (- (position type *numeric-types* :key #'first) 2))
                 (if (eq backend :sbcl) 1 0))))
    (or (aref cache slot)
        (setf (aref cache slot)
              (#+ecl eval #-ecl compile
               #-ecl nil
                       `(lambda (,@(when destination (list destination)) ,@arguments ,count
                                 ,@(when destination (list d-offset)) ,@offsets)
                          ,(if (and (eq backend :sbcl) (member reducer '(nil sum)))
                               #+(and sbcl x86-64)
                               (sbcl-integer-kernel-form tree type destination arguments d-offset offsets count)
                               #-(and sbcl x86-64)
                               '(error "SBCL SIMD is unavailable on this platform")
                               (lisp-kernel-form tree type destination arguments d-offset offsets count
                                                 'fma reducer))))))))

(defun kernel-combine-form (reducer type)
  "Form folding the block results of a REDUCER kernel over vectors of TYPE."
  (ecase reducer
    ((nil) nil)
    ((sum asum) `(sum-combiner ,type))
    (prod `(product-combiner ,type))
    (nrm2 '#'norm-combiner)
    ((minimum maximum) `(extremum-combiner ,(eq reducer 'maximum)))
    ((argmin argmax) `(index-combiner ,(eq reducer 'argmax)))
    (count '#'count-combiner)
    (any '#'any-combiner)
    (all '#'all-combiner)))

(defun kernel-tree-value (tree type vectors offsets index &optional masked)
  "Value of the kernel expression TREE at INDEX of the Lisp VECTORS (element type
TYPE) whose slices start at OFFSETS. It runs the operations of the compiled
scalar forms, so a call needs no per-type expansion. MASKED selects the MIN and MAX
of mask kernels."
  (let ((integerp (integer-type-p type)))
    (labels ((integer-call (kind &rest operands)
               (apply (integer-operation-symbol kind type) operands))
             (walk (node)
               (ecase (first node)
                 (:argument (let ((position (second node)))
                              (aref (nth position vectors) (+ (nth position offsets) index))))
                 (:constant (kernel-constant-value (second node) type))
                 (:negate (if integerp
                              (integer-call :negate (walk (second node)))
                              (- (walk (second node)))))
                 (:unary (let ((value (walk (third node))))
                           (cond ((eq (second node) :sqrt) (kernel-sqrt value))
                                 (integerp (integer-call :abs value))
                                 (t (funcall (car (rassoc (second node) *kernel-operators*)) value)))))
                 (:fma (apply #'fma (mapcar #'walk (rest node))))
                 (:select (if (walk (second node)) (walk (third node)) (walk (fourth node))))
                 (:operation
                  (let ((kind (second node)) (left (walk (third node))) (right (walk (fourth node))))
                    (case kind
                      (:eq (= left right)) (:ne (/= left right)) (:lt (< left right))
                      (:le (<= left right)) (:gt (> left right)) (:ge (>= left right))
                      (:min (cond (integerp (integer-call :min left right))
                                  (masked (kernel-min-left left right))
                                  (t (kernel-min left right))))
                      (:max (cond (integerp (integer-call :max left right))
                                  (masked (kernel-max-left left right))
                                  (t (kernel-max left right))))
                      (t (if integerp
                             (integer-call kind left right)
                             (funcall (car (rassoc kind *kernel-operators*)) left right)))))))))
      (walk tree))))

(defun staged-kernel-form (name reducer type destination d-offset arguments offsets starts
                           count direct body &optional value declarations)
  "Wrap BODY, the computation of kernel NAME over the slices, to stage vector
views: each block calls NAME again on Lisp buffers, passing the offsets as the
start keywords in STARTS (the destination's first). The result is DESTINATION,
when there is one, or the reduction. For ARGMIN and ARGMAX, VALUE is a function
of an index variable returning the form computing the element there, which the
block results are compared by. DECLARATIONS apply to BODY's variables."
  (let* ((bindings `(,@(when destination `((,destination ,d-offset nil :out)))
                     ,@(mapcar (lambda (argument offset) (list argument offset nil))
                               arguments offsets)))
         (call `(,name ,@(when destination (list destination)) ,@arguments :end ,count
                       ,@(loop for start in starts
                               for offset in (if destination (cons d-offset offsets) offsets)
                               append (list (intern (symbol-name start) :keyword) offset))))
         (index (gensym "INDEX"))
         (staged `(with-staged ,bindings
                      (,count :direct ,direct :combine ,(kernel-combine-form reducer type)
                              :declarations ,declarations
                              :block-form
                              ,(if (member reducer '(argmin argmax))
                                   `(let ((,index ,call))
                                      (and ,index (cons ,index ,(funcall value index))))
                                   call))
                    ,body)))
    (cond ((member reducer '(argmin argmax)) `(staged-index ,staged))
          (destination `(progn ,staged ,destination))
          (t staged))))

(defmacro define-single-row-kernel (name (&rest arguments) expression)
  "Define an elementwise destination kernel or a scalar reduction kernel such as
(SUM expression), (ASUM expression), (NRM2 expression), (MINIMUM expression),
(MAXIMUM expression), (ARGMIN expression) or (ARGMAX expression). Both accept START,
END, and per-input start keywords. Experimental."
  (when (mask-kernel-expression-p expression)
    (return-from define-single-row-kernel (mask-kernel-expansion name arguments expression)))
  (when (> (length arguments) +kernel-max-arguments+)
    (error "A kernel takes at most ~D arguments" +kernel-max-arguments+))
  (let* ((reducer (and (consp expression) (find (first expression) *kernel-reducers*)))
         (reduction-p (and reducer t))
         (destination (unless reduction-p (gensym "DESTINATION"))))
    (when reduction-p
      (unless (and arguments (= 2 (length expression)))
        (error "A ~(~A~) kernel needs input vectors and one expression" reducer)))
    (let ((tree (parse-kernel-expression (if reduction-p (second expression) expression) arguments)))
      (multiple-value-bind (instructions constants scratch-count) (lower-kernel tree)
        (let* ((count (gensym "COUNT"))
               (d-offset (when destination (gensym "D-OFFSET")))
               (offsets (loop for nil in arguments collect (gensym "OFFSET")))
               (program (gensym "PROGRAM")) (programs (gensym "PROGRAMS"))
               (integer-functions (gensym "INTEGER-FUNCTIONS")) (type (gensym "TYPE"))
               (complex-kernel (gensym "COMPLEX-KERNEL"))
               (offsets-variable (gensym "OFFSETS"))
               (starts (mapcar (lambda (argument) (intern (format nil "~A-START" argument))) arguments))
               (vectors (if destination (cons destination arguments) arguments))
               (all-starts (if destination (cons 'destination-start starts) starts))
               (all-offsets (if destination (cons d-offset offsets) offsets))
               (bytes (kernel-bytes instructions))
               (native-form `(ecase ,type
                               ,@(loop for (key element foreign) in *numeric-types*
                                       collect `(,key ,(native-kernel-form program foreign destination arguments
                                                                           d-offset offsets count reducer)))))
               (integer-form
                 `(funcall (integer-kernel-function ,integer-functions ',tree ,type *backend*
                                                   ',destination ',arguments ',count ',d-offset ',offsets
                                                   ',reducer)
                           ,@vectors ,count ,@all-offsets))
               #+ecl (runner (gensym "RUNNER"))
               #+ecl (runner-arguments (append (list program type count) vectors all-offsets))
               #+ecl (runner-form `(lambda ,runner-arguments ,native-form)))
          `(let ((,programs (make-array ,(length *numeric-types*) :initial-element nil))
                 (,integer-functions (make-array 16 :initial-element nil))
                 (,complex-kernel (make-complex-kernel ',arguments ',tree :numeric ',reducer))
                 #+ecl (,runner ,runner-form))
             (defun ,name (,@vectors &key start end ,@all-starts)
               ,(if reduction-p
                    (format nil "Return the ~(~A~) of the kernel expression over the input slice."
                            reducer)
                    "Compute the kernel expression into destination and return it.")
               ,(argument-index-result reducer (gensym "RESULT")
                `(multiple-value-bind (,type ,count ,offsets-variable)
                   (resolve-slice (list ,@vectors) (list ,@all-starts) start end)
                 (when (integer-type-p ,type) (validate-integer-kernel ',tree ,type ',reducer))
                 (when (complex-type-p ,type)
                   (validate-complex-kernel (complex-kernel-tree ,complex-kernel) ',reducer))
                 ,@(when (eq reducer 'prod)
                     `((unless (or (not (eq *backend* :native)) *native-row-reducers-p*)
                         (return-from ,name
                           (let ((*backend* :lisp))
                             (,name ,@vectors :start start :end end
                                    ,@(loop for s in all-starts append (list (intern (symbol-name s) :keyword) s))))))))
                 (destructuring-bind ,all-offsets ,offsets-variable
                   (declare (type fixnum ,@all-offsets))
                   ,@(unshifted-kernel-inputs-forms destination d-offset arguments offsets count)
                   ,(staged-kernel-form
                     name reducer type destination d-offset arguments offsets all-starts count
                     `(and (eq *backend* :native)
                           (or (not (complex-type-p ,type))
                               (and (native-complex-kernel-p) ,(not (eq reducer 'prod)))))
                     `(if (complex-type-p ,type)
                       (run-complex-expression ,complex-kernel ,type ,destination
                                               (list ,@arguments) (list ,@offsets) ,(or d-offset 0) ,count)
                       (ecase *backend*
                     (:lisp (if (integer-type-p ,type) ,integer-form
                                (if (eq ,type :f32)
                                    ,(selected-lisp-kernel-form tree :f32 destination arguments d-offset offsets count reducer)
                                    ,(selected-lisp-kernel-form tree :f64 destination arguments d-offset offsets count reducer))))
                     (:sbcl
                      ,(if (and (boundp '*sbcl-simd-available-p*) *sbcl-simd-available-p*)
                           #+(and sbcl x86-64)
                           `(if (integer-type-p ,type) ,integer-form
                                (if (eq ,type :f32)
                                    ,(sbcl-kernel-form tree :f32 destination arguments d-offset offsets count reducer)
                                    ,(sbcl-kernel-form tree :f64 destination arguments d-offset offsets count reducer)))
                           #-(and sbcl x86-64)
                           '(error "SBCL SIMD is unavailable on this platform")
                           '(error "SBCL SIMD is unavailable on this platform")))
                     (:native
                      (let* ((slot (position ,type *numeric-types* :key #'first))
                             (,program (or (aref ,programs slot)
                                           (setf (aref ,programs slot)
                                                 (make-native-program ',bytes ',constants ,scratch-count ,type)))))
                        #+ecl (funcall (ensure-native-kernel-runner
                                       ,program ,(+ (* 8 (length arguments)) (position reducer '(nil sum asum nrm2 minimum maximum argmin argmax prod)))
                                       ',runner-form ,runner)
                                       ,@runner-arguments)
                        #-ecl ,native-form))))
                     (lambda (index)
                       `(kernel-tree-value ',tree ,type (list ,@arguments) (list ,@offsets)
                                           ,index))
                     `((type fixnum ,@all-offsets))))
                 ,@(when destination (list destination)))))))))))
