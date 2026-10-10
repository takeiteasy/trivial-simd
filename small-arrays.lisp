(in-package #:trivial-simd)

;; TODO: one limit for every operation; the measured crossover with the native
;; backend is ~80 elements for dot, ~110 for add! and higher for min!. Per-operation
;; limits would gain a little (#50).
(defconstant +small-array-limit+ 64
  "Arrays up to this length run an inline loop instead of a backend call.")

(defvar *inline-small-arrays* t
  "When false, calls compiled with an inline expansion use the active backend.")

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun small-array-expansion (name arguments forms each reduce result guard)
    "Expand a call to NAME into guarded inline loops for float vectors, falling
back to the full call. ARGUMENTS pairs each parameter with :VECTOR or :SCALAR."
    (let* ((names (mapcar #'first arguments))
           (vectors (loop for (argument kind) in arguments
                          when (eq kind :vector) collect argument))
           (first-vector (first vectors)))
      `(let ,(mapcar #'list names forms)
         (cond
           ,@(loop for element in '(single-float double-float)
                   collect
                   `((and *inline-small-arrays*
                          ,@(loop for (argument kind) in arguments
                                  collect (if (eq kind :vector)
                                              `(typep ,argument '(simple-array ,element (*)))
                                              `(typep ,argument ',element)))
                          ,@(loop for other in (rest vectors)
                                  collect `(= (length ,first-vector) (length ,other)))
                          (<= (length ,first-vector) +small-array-limit+)
                          ,@(when guard
                              `((locally (declare (type ,element ,@(loop for (argument kind) in arguments
                                                                         when (eq kind :scalar) collect argument)))
                                  ,guard))))
                     (locally
                         (declare ,@(loop for (argument kind) in arguments
                                          collect (if (eq kind :vector)
                                                      `(type (simple-array ,element (*)) ,argument)
                                                      `(type ,element ,argument)))
                                  (optimize (speed 1) (safety 0)))
                       ,(if reduce
                            `(let ((accumulator (coerce 0 ',element)))
                               (declare (type ,element accumulator))
                               (dotimes (i (length ,first-vector))
                                 (setf accumulator ,reduce))
                               ,result)
                            `(progn
                               (dotimes (i (length ,first-vector))
                                 ,each)
                               ,result)))))
           (t (locally (declare (notinline ,name))
                (,name ,@names))))))))

(defmacro define-small-array-expansion (name arguments &key each reduce result guard)
  "Define a compiler macro for NAME that runs EACH (or REDUCE into ACCUMULATOR)
over the vectors in ARGUMENTS when a call without keywords has float vectors of
one length up to +SMALL-ARRAY-LIMIT+ and GUARD, when given, is true."
  `(define-compiler-macro ,name (&whole form &rest forms)
     (if (= (length forms) ,(length arguments))
         (small-array-expansion ',name ',arguments forms ',each ',reduce ',result ',guard)
         form)))

(define-small-array-expansion add! ((destination :vector) (left :vector) (right :vector))
  :each (setf (aref destination i) (+ (aref left i) (aref right i)))
  :result destination)

(define-small-array-expansion subtract! ((destination :vector) (left :vector) (right :vector))
  :each (setf (aref destination i) (- (aref left i) (aref right i)))
  :result destination)

(define-small-array-expansion multiply! ((destination :vector) (left :vector) (right :vector))
  :each (setf (aref destination i) (* (aref left i) (aref right i)))
  :result destination)

(define-small-array-expansion divide! ((destination :vector) (left :vector) (right :vector))
  :each (setf (aref destination i) (/ (aref left i) (aref right i)))
  :result destination)

(define-small-array-expansion scale! ((x :vector) (a :scalar))
  :each (setf (aref x i) (* a (aref x i)))
  :result x)

(define-small-array-expansion axpy! ((y :vector) (a :scalar) (x :vector))
  :each (setf (aref y i) (+ (* a (aref x i)) (aref y i)))
  :result y)

(define-small-array-expansion sum ((input :vector))
  :reduce (+ accumulator (aref input i))
  :result accumulator)

(define-small-array-expansion dot ((left :vector) (right :vector))
  :reduce (+ accumulator (* (aref left i) (aref right i)))
  :result accumulator)

(define-small-array-expansion negate! ((destination :vector) (input :vector))
  :each (setf (aref destination i) (- (aref input i)))
  :result destination)

(define-small-array-expansion abs! ((destination :vector) (input :vector))
  :each (setf (aref destination i) (abs (aref input i)))
  :result destination)

(define-small-array-expansion reciprocal! ((destination :vector) (input :vector))
  :each (setf (aref destination i) (/ (aref input i)))
  :result destination)

(define-small-array-expansion min! ((destination :vector) (left :vector) (right :vector))
  :each (let ((a (aref left i)) (b (aref right i)))
          (setf (aref destination i) (if (< b a) b a)))
  :result destination)

(define-small-array-expansion max! ((destination :vector) (left :vector) (right :vector))
  :each (let ((a (aref left i)) (b (aref right i)))
          (setf (aref destination i) (if (> b a) b a)))
  :result destination)

(define-small-array-expansion clamp! ((destination :vector) (input :vector)
                                      (lower :scalar) (upper :scalar))
  :guard (<= lower upper)
  :each (let ((bounded (if (> lower (aref input i)) lower (aref input i))))
          (setf (aref destination i) (if (< upper bounded) upper bounded)))
  :result destination)
