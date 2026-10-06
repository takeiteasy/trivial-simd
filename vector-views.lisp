(in-package #:trivial-simd)

(defun vector-element-size (type)
  "Bytes per element of numeric TYPE."
  (floor (fourth (numeric-type type)) 8))

(defun make-vector-view (pointer type length &key (offset 0))
  "Return a view of LENGTH elements of numeric TYPE (:F32, :S8, :C64 and so on)
in foreign memory, starting OFFSET elements after POINTER. The memory is not
copied, owned or freed: it must stay valid while the view is used."
  (unless (cffi:pointerp pointer)
    (error 'type-error :datum pointer :expected-type 'cffi:foreign-pointer))
  (numeric-type type)
  (unless (typep length '(integer 0 #.most-positive-fixnum))
    (error "Invalid vector view length ~S" length))
  (unless (typep offset '(integer 0 #.most-positive-fixnum))
    (error "Invalid vector view offset ~S" offset))
  (when (and (cffi:null-pointer-p pointer) (plusp length))
    (error "A nonempty vector view needs a non-null pointer"))
  (%make-vector-view (if (zerop offset)
                         pointer
                         (cffi:inc-pointer pointer (* offset (vector-element-size type))))
                     type length))

(defmethod print-object ((view vector-view) stream)
  (print-unreadable-object (view stream :type t :identity t)
    (format stream "~S ~D" (vector-view-type view) (vector-view-length view))))

(declaim (inline operand-vector-p vector-length))
(defun operand-vector-p (value)
  "True for a Lisp vector or a vector view, false for a scalar operand."
  (or (vectorp value) (vector-view-p value)))

(defun vector-length (vector)
  (if (vector-view-p vector) (vector-view-length vector) (length vector)))

(defun view-address (view index)
  (+ (cffi:pointer-address (vector-view-pointer view))
     (* index (vector-element-size (vector-view-type view)))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *view-element-types*
    (append (loop for (key element foreign) in *numeric-types*
                  collect (list key element foreign))
            '((:c32 (complex single-float) :float)
              (:c64 (complex double-float) :double))))

  (defun view-read-form (key foreign pointer index)
    (if (complex-type-p key)
        `(complex (cffi:mem-aref ,pointer ,foreign (* 2 ,index))
                  (cffi:mem-aref ,pointer ,foreign (1+ (* 2 ,index))))
        `(cffi:mem-aref ,pointer ,foreign ,index)))

  (defun view-write-form (key foreign pointer index value)
    (if (complex-type-p key)
        (let ((number (gensym "VALUE")))
          `(let ((,number ,value))
             (setf (cffi:mem-aref ,pointer ,foreign (* 2 ,index)) (realpart ,number)
                   (cffi:mem-aref ,pointer ,foreign (1+ (* 2 ,index))) (imagpart ,number))))
        `(setf (cffi:mem-aref ,pointer ,foreign ,index) ,value))))

(defmacro define-view-access ()
  `(progn
     (defun view-ref (view index)
       (let ((pointer (vector-view-pointer view)))
         (ecase (vector-view-type view)
           ,@(loop for (key nil foreign) in *view-element-types*
                   collect `(,key ,(view-read-form key foreign 'pointer 'index))))))
     (defun (setf view-ref) (value view index)
       (let ((pointer (vector-view-pointer view)))
         (ecase (vector-view-type view)
           ,@(loop for (key element foreign) in *view-element-types*
                   collect `(,key ,(view-write-form key foreign 'pointer 'index
                                                    `(coerce value ',element))))))
       value)))

(define-view-access)

(defun vector-ref (vector index)
  "Element INDEX of a Lisp vector or a vector view."
  (if (vector-view-p vector) (view-ref vector index) (aref vector index)))

(defun (setf vector-ref) (value vector index)
  (if (vector-view-p vector)
      (setf (view-ref vector index) value)
      (setf (aref vector index) value)))

;; Callers validate every span before these unchecked transfers.
(defmacro define-staging-transfers ()
  `(progn
     (defun stage-in (source offset stride count buffer)
       "Copy the COUNT elements of SOURCE (a vector or view) from OFFSET at STRIDE
into BUFFER from index 0."
       (declare (type fixnum offset stride count))
       (etypecase buffer
         ,@(loop for (key element foreign) in *view-element-types*
                 collect
                 `((simple-array ,element (*))
                   (let ((index offset))
                     (declare (type fixnum index) (optimize (speed 3) (safety 0)))
                     (if (vector-view-p source)
                         (let ((pointer (vector-view-pointer source)))
                           (dotimes (i count)
                             (setf (aref buffer i) ,(view-read-form key foreign 'pointer 'index))
                             (incf index stride)))
                         (let ((source source))
                           (declare (type (simple-array ,element (*)) source))
                           (dotimes (i count)
                             (setf (aref buffer i) (aref source index))
                             (incf index stride)))))))))
     (defun stage-out (buffer target offset stride count)
       "Store the first COUNT elements of BUFFER into TARGET (a vector or view)
from OFFSET at STRIDE."
       (declare (type fixnum offset stride count))
       (etypecase buffer
         ,@(loop for (key element foreign) in *view-element-types*
                 collect
                 `((simple-array ,element (*))
                   (let ((index offset))
                     (declare (type fixnum index) (optimize (speed 3) (safety 0)))
                     (if (vector-view-p target)
                         (let ((pointer (vector-view-pointer target)))
                           (dotimes (i count)
                             ,(view-write-form key foreign 'pointer 'index '(aref buffer i))
                             (incf index stride)))
                         (let ((target target))
                           (declare (type (simple-array ,element (*)) target))
                           (dotimes (i count)
                             (setf (aref target index) (aref buffer i))
                             (incf index stride)))))))))
     (defun staging-buffer (vector count)
       "Return an uninitialised Lisp vector of COUNT elements of VECTOR's type."
       (declare (type fixnum count))
       (ecase (vector-type vector)
         ,@(loop for (key element) in *view-element-types*
                 collect `(,key (make-array count :element-type ',element)))))))

(define-staging-transfers)

(defun slice-overlap (left left-offset left-stride right right-offset right-stride count)
  "Return :SAME for identical byte mappings, :OVERLAP for intersecting slices,
or NIL. Lisp vectors share storage by identity; views share it by address."
  (unless (and (plusp count) (operand-vector-p left) (operand-vector-p right)
               (or (eq left right)
                   (and (vector-view-p left) (vector-view-p right))))
    (return-from slice-overlap nil))
  (let* ((left-size (vector-element-size (vector-type left)))
         (right-size (vector-element-size (vector-type right)))
         (left-first (if (vector-view-p left) (view-address left left-offset)
                         (* left-offset left-size)))
         (right-first (if (vector-view-p right) (view-address right right-offset)
                          (* right-offset right-size)))
         (left-step (* (or left-stride 1) left-size))
         (right-step (* (or right-stride 1) right-size)))
    (when (and (= left-first right-first) (= left-size right-size)
               (or (= count 1) (= left-step right-step)))
      (return-from slice-overlap :same))
    (let* ((left-low (min left-first (+ left-first (* (1- count) left-step))))
           (right-low (min right-first (+ right-first (* (1- count) right-step))))
           (left-step (abs left-step)) (right-step (abs right-step))
           (left-high (+ left-low (* (1- count) left-step) left-size))
           (right-high (+ right-low (* (1- count) right-step) right-size)))
      (when (and (< left-low right-high) (< right-low left-high))
        (loop with a = left-low with b = right-low
              with i = 0 with j = 0
              while (and (< i count) (< j count))
              do (cond ((<= (+ a left-size) b) (incf a left-step) (incf i))
                       ((<= (+ b right-size) a) (incf b right-step) (incf j))
                       (t (return-from slice-overlap :overlap))))))))

(defun unshifted-bulk-input (destination d-offset d-stride input offset stride count)
  (if (eq :overlap (slice-overlap destination d-offset d-stride input offset stride count))
      (let ((copy (staging-buffer input count)))
        (stage-in input offset (or stride 1) count copy)
        (values copy 0 1))
      (values input offset stride)))

(defmacro with-bulk-staged ((destination &rest inputs) options &body body)
  "Snapshot shifted inputs before staging a single-destination bulk call."
  (destructuring-bind (output output-offset output-stride role) destination
    (let* ((count (first options)) (d-stride (gensym "OUTPUT-STRIDE"))
           (bindings (list (list output output-offset d-stride role)))
           (wrappers nil))
      (dolist (input inputs)
        (destructuring-bind (vector offset stride &optional (role :in)) input
          (let ((i-stride (gensym "INPUT-STRIDE")))
            (push (list vector offset i-stride role) bindings)
            (push `(multiple-value-bind (,vector ,offset ,i-stride)
                       (unshifted-bulk-input ,output ,output-offset ,d-stride
                                             ,vector ,offset ,stride ,count)) wrappers))))
      (let ((form `(with-staged ,(nreverse bindings) ,options ,@body)))
        (dolist (wrapper wrappers)
          (setf form `(,@wrapper ,form)))
        `(let ((,d-stride ,output-stride)) ,form)))))

(defun unshifted-kernel-input (destination d-offset input offset count)
  "Return INPUT and OFFSET, or a Lisp copy of INPUT's COUNT-element slice and 0
when that slice shares memory with DESTINATION's slice other than element for
element, so a kernel reads every input element before writing any output (#67)."
  (declare (type fixnum d-offset offset count))
  (let ((input-size (vector-element-size (vector-type input)))
        (output-size (vector-element-size (vector-type destination))))
    (multiple-value-bind (input-low output-low)
        (cond ((eq input destination) (values (* offset input-size) (* d-offset output-size)))
              ((and (vector-view-p input) (vector-view-p destination))
               (values (view-address input offset) (view-address destination d-offset)))
              (t (return-from unshifted-kernel-input (values input offset))))
      (if (and (or (/= input-low output-low) (/= input-size output-size))
               (< input-low (+ output-low (* count output-size)))
               (< output-low (+ input-low (* count input-size))))
          (let ((copy (staging-buffer input count)))
            (stage-in input offset 1 count copy)
            (values copy 0))
          (values input offset)))))

(defvar *view-block-size* 4096
  "Elements per block when a backend reads a vector view through a Lisp buffer.")

(defun staging-order (specs count)
  "Return :FORWARD, :REVERSE or :WHOLE: the block order that lets every output
view read the input views it overlaps before overwriting them, or :WHOLE when
no block order can (the slices then move through buffers as one block)."
  (let ((order :forward) (constrained nil))
    (flet ((span (view offset stride)
             (let ((first (view-address view offset))
                   (last (view-address view (+ offset (* (1- count) stride)))))
               (values (min first last)
                       (+ (max first last) (vector-element-size (vector-view-type view)))
                       first))))
      (loop for (output output-offset output-stride role) in specs
            for output-index from 0
            when (and (vector-view-p output) (member role '(:out :in-out)))
              do (multiple-value-bind (output-low output-high output-first)
                     (span output output-offset (or output-stride 1))
                   (loop for (input input-offset input-stride input-role) in specs
                         for input-index from 0
                         when (and (/= input-index output-index) (vector-view-p input)
                                   (member input-role '(:in :in-out)))
                           do (multiple-value-bind (input-low input-high input-first)
                                  (span input input-offset (or input-stride 1))
                                (when (and (< output-low input-high) (< input-low output-high))
                                  (let ((stride (or output-stride 1))
                                        (shift (- output-first input-first)))
                                    (cond ((/= stride (or input-stride 1))
                                           (return-from staging-order :whole))
                                          ((zerop shift))
                                          (t (let ((wanted (if (plusp (* shift stride))
                                                               :reverse :forward)))
                                               (when (and constrained (not (eq wanted order)))
                                                 (return-from staging-order :whole))
                                               (setf order wanted constrained t)))))))))))
    order))

(defstruct (staging (:constructor %make-staging))
  "Progress of a call through blocks of Lisp buffers; see WITH-STAGED."
  specs buffers combine (count 0 :type fixnum) (size 0 :type fixnum)
  (step 1 :type fixnum) (start 0 :type fixnum) (length 0 :type fixnum) result)

(defun make-staging (count specs combine)
  "Start staging the COUNT-element slices in SPECS, a list of (VECTOR OFFSET
STRIDE ROLE). Views and strided vectors move through Lisp buffers; contiguous
Lisp vectors and scalars pass through."
  (let* ((order (if (plusp count) (staging-order specs count) :forward))
         (size (if (eq order :whole) count (min count *view-block-size*)))
         (last (if (zerop count) 0 (* size (floor (1- count) size)))))
    (%make-staging
     :specs specs :combine combine :count count :size size
     :step (if (eq order :reverse) (- size) size)
     :start (if (eq order :reverse) last 0)
     :buffers (loop for (vector nil stride) in specs
                    collect (and (operand-vector-p vector)
                                 (or (vector-view-p vector) (not (member stride '(nil 1))))
                                 (staging-buffer vector size))))))

(defun staging-arguments (staging)
  "Fill the buffers for the current block and return a list of each operand and
offset to use for it, followed by the block length."
  (let* ((start (staging-start staging))
         (length (min (staging-size staging) (- (staging-count staging) start)))
         (arguments '()))
    (setf (staging-length staging) length)
    (loop for (vector offset stride role) in (staging-specs staging)
          for buffer in (staging-buffers staging)
          do (cond ((null buffer)
                    (push vector arguments)
                    (push (and offset (+ offset start)) arguments))
                   (t
                    (unless (eq role :out)
                      (let ((stride (or stride 1)))
                        (stage-in vector (+ offset (* start stride)) stride length buffer)))
                    (push buffer arguments)
                    (push 0 arguments))))
    (nreverse (cons length arguments))))

(defun staging-finish-block (staging value)
  "Store the current block's outputs and fold VALUE, its result, with COMBINE,
which receives the result so far (NIL before the first block), the block's
first index within the slice, and VALUE. Return true after the last block."
  (let ((start (staging-start staging)) (length (staging-length staging)))
    (loop for (vector offset stride role) in (staging-specs staging)
          for buffer in (staging-buffers staging)
          when (and buffer (not (eq role :in)))
            do (let ((stride (or stride 1)))
                 (stage-out buffer vector (+ offset (* start stride)) stride length)))
    (setf (staging-result staging)
          (let ((combine (staging-combine staging)))
            (if combine (funcall combine (staging-result staging) start value) value)))
    (let ((next (+ start (staging-step staging))))
      (setf (staging-start staging) next)
      (not (< -1 next (staging-count staging))))))

(defun run-staged (count specs function combine)
  "Call FUNCTION once for each block of the COUNT-element slices in SPECS (see
MAKE-STAGING) with each operand and offset to use, then the block length, and
fold the block results with COMBINE (see STAGING-FINISH-BLOCK)."
  (let ((staging (make-staging count specs combine)))
    (loop until (staging-finish-block
                 staging (apply function (staging-arguments staging))))
    (staging-result staging)))

(defmacro with-staged ((&rest bindings)
                       (count &key direct combine (staged (gensym "STAGED")) declarations
                                   (block-form nil block-form-p))
                       &body body)
  "Run BODY like WITH-GATHERED when no operand in BINDINGS is a vector view. With
a view, run BODY directly on it when DIRECT is true and every view is contiguous
(the body then reaches the view's memory itself), and otherwise once for each
block of Lisp buffers through RUN-STAGED, folding block results with COMBINE.
COUNT must be a variable; BODY sees it, and each binding's vector and offset,
rebound, and sees the variable STAGED true for a block. BLOCK-FORM, when given,
runs for each block instead of BODY. DECLARATIONS are declaration specifiers
for BODY's variables."
  (check-type count symbol)
  (let ((function (gensym "BODY")) (block (gensym "BLOCK"))
        (parameters (loop for (vector offset) in bindings append (list vector offset))))
    `(flet ((,function (,@parameters ,count ,staged)
              (declare (ignorable ,@parameters ,staged) ,@declarations)
              ,@body))
       (declare (dynamic-extent #',function))
       (if (and (or ,@(loop for (vector) in bindings collect `(vector-view-p ,vector)))
                ,(if direct
                     `(not (and ,direct
                                ,@(loop for (vector nil stride) in bindings
                                        collect `(or (not (vector-view-p ,vector))
                                                     (member ,stride '(nil 1))))))
                     t))
           (flet ((,block (,@parameters ,count)
                    (declare (ignorable ,@parameters))
                    ,(if block-form-p
                         block-form
                         `(,function ,@parameters ,count t))))
             (declare (dynamic-extent #',block))
             (run-staged ,count
                         (list ,@(loop for (vector offset stride role) in bindings
                                       collect `(list ,vector ,offset ,stride ,(or role :in))))
                         #',block ,combine))
           (with-gathered ,bindings ,count (,function ,@parameters ,count nil))))))

(defun sum-combiner (type)
  "Combine block sums of numeric TYPE, wrapping integer sums like the backends."
  (let ((add (if (integer-type-p type)
                 (symbol-function (integer-operation-symbol :add type))
                 #'+)))
    (lambda (accumulated start value)
      (declare (ignore start))
      (if accumulated (funcall add accumulated value) value))))

(defun count-combiner (accumulated start value)
  (declare (ignore start))
  (+ (or accumulated 0) value))

(defun any-combiner (accumulated start value)
  (declare (ignore start))
  (or accumulated value))

(defun all-combiner (accumulated start value)
  ;; Reductions read inputs only, so their first block starts at 0.
  (if (zerop start) value (and accumulated value)))

(defun norm-combiner (accumulated start value)
  "Combine block Euclidean norms without overflow, in VALUE's float type."
  (declare (ignore start))
  (if (null accumulated)
      value
      (let* ((a (abs (coerce accumulated 'double-float)))
             (b (abs (coerce value 'double-float)))
             (large (max a b)) (small (min a b)))
        (coerce (cond ((not (and (<= a most-positive-double-float)
                                 (<= b most-positive-double-float)))
                       ;; Infinities and NaNs propagate.
                       (+ a b))
                      ;; Beyond 500 binary orders the smaller norm adds nothing,
                      ;; and squaring their ratio could underflow.
                      ((or (zerop small)
                           (> (- (nth-value 1 (decode-float large))
                                 (nth-value 1 (decode-float small)))
                              500))
                       large)
                      (t (* large (sqrt (+ 1 (expt (/ small large) 2))))))
                (type-of-float value)))))

(defun type-of-float (value)
  (if (typep value 'single-float) 'single-float 'double-float))

(declaim (inline staged-index))
(defun staged-index (result)
  "The slice index from an argument-extremum WITH-STAGED result: the INDEX-COMBINER
cons of a staged call, or the plain index (or NIL) of a direct one."
  (if (consp result) (car result) result))

(defun extremum-combiner (maximum-p)
  "Combine block minima or maxima (NIL for an empty slice); ties keep the first."
  (lambda (accumulated start value)
    (declare (ignore start))
    (cond ((null accumulated) value)
          ((null value) accumulated)
          ((if maximum-p (> value accumulated) (< value accumulated)) value)
          (t accumulated))))

(defun index-combiner (maximum-p)
  "Combine block (INDEX . VALUE) extrema whose INDEX is relative to the block
into one relative to the slice; ties keep the first. A block without values
gives NIL."
  (lambda (accumulated start value)
    (cond ((null value) accumulated)
          ((or (null accumulated)
               (if maximum-p (> (cdr value) (cdr accumulated)) (< (cdr value) (cdr accumulated))))
           (cons (+ start (car value)) (cdr value)))
          (t accumulated))))

(defun product-combiner (type)
  (let ((multiply (if (integer-type-p type)
                      (symbol-function (integer-operation-symbol :multiply type))
                      #'*)))
    (lambda (accumulated start value)
      (declare (ignore start))
      (if accumulated (funcall multiply accumulated value) value))))
