(in-package #:trivial-simd)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun normalize-kernel-inputs (arguments)
    (unless (and arguments (<= (length arguments) +kernel-max-arguments+))
      (error "A declared-input kernel needs 1 to ~D inputs" +kernel-max-arguments+))
    (let ((names nil) (specs nil))
      (dolist (argument arguments)
        (let* ((declaration (if (consp argument) argument (list argument)))
               (name (first declaration)) (options (rest declaration))
               (source nil) (repeat 1) (seen nil))
          (unless (and (symbolp name) name (not (keywordp name)) (not (eq name t))
                       (not (member name names)) (evenp (length options)))
            (error "Invalid kernel input declaration ~S" argument))
          (loop for (key value) on options by #'cddr
                do (when (or (not (member key '(:type :repeat))) (member key seen))
                     (error "Invalid kernel input option ~S" key))
                   (push key seen)
                   (ecase key
                     (:type
                      (unless (assoc value *numeric-types*)
                        (error "Invalid kernel input type ~S" value))
                      (setf source value))
                     (:repeat
                      (unless (typep value `(integer 1 ,most-positive-fixnum))
                        (error "Invalid kernel repetition factor ~S" value))
                      (setf repeat value))))
          (push name names)
          (push (list source repeat) specs)))
      (unless (some (lambda (spec) (= 1 (second spec))) specs)
        (error "A kernel needs a non-repeated input to establish its length"))
      (values (nreverse names) (nreverse specs)))))

(cffi:defcstruct kernel-input
  (data :pointer) (length :size) (start :size) (type :size)
  (repeat :size) (phase :size) (stride :int64))

(macrolet ((define-input-calls ()
             `(progn
                ,@(loop for suffix in '(f32 f64)
                        collect
                        `(define-native (,(format nil "ts_kernel_inputs_~(~A~)" suffix)
                                         ,(intern (format nil "%NATIVE-KERNEL-INPUTS-~A" suffix))) :int
                           (code :pointer) (code-length :size) (constants :pointer)
                           (inputs :pointer) (input-count :size) (output :pointer)
                           (rows :size) (count :size) (scratch-count :size)
                           (mode :unsigned-int) (scale :double) (wide :pointer) (index :pointer))))))
  (define-input-calls))

(defvar *native-kernel-inputs-available-p*
  (and *native-available-p*
       (every (lambda (name) (ignore-errors (cffi:foreign-symbol-pointer name)))
              '("ts_kernel_inputs_f32" "ts_kernel_inputs_f64"))))

(defun declared-kernel-precision (inputs specs destination kind)
  (let ((type (if (and destination (eq kind :numeric))
                  (vector-type destination)
                  (loop for input in inputs for (source repeat) in specs
                        when (and (= repeat 1) (not (integer-type-p source)))
                          return (vector-type input)))))
    (unless (member type '(:f32 :f64))
      (error "Declared-input kernels require an ordinary float input or numeric destination"))
    (loop for input in inputs for (source nil) in specs
          do (unless (eq (vector-type input) (or source type))
               (error "Kernel input has the wrong declared type"))
             (when (and source (not (integer-type-p source)) (not (eq source type)))
               (error "Kernel float inputs must match the computation precision")))
    (when destination
      (unless (eq (vector-type destination) (if (eq kind :mask) :u8 type))
        (error "Kernel destination has the wrong type")))
    type))

(defun declared-kernel-limit (type)
  (min most-positive-fixnum
       (floor (1- (ash 1 (1- (* 8 (cffi:foreign-type-size :pointer)))))
              (vector-element-size type))))

(defun declared-kernel-span (input offset count repeat phase)
  (let* ((length (vector-length input))
         (span (if (zerop count) 0 (1+ (floor (+ phase count -1) repeat))))
         (limit (declared-kernel-limit (vector-type input))))
    (unless (and (typep offset `(integer 0 ,limit)) (<= (+ offset span) length limit))
      (error "Kernel input span is out of bounds"))
    (when (and (vector-view-p input)
               (> (+ (cffi:pointer-address (vector-view-pointer input))
                     (* (+ offset span) (vector-element-size (vector-type input))))
                  (1- (ash 1 (* 8 (cffi:foreign-type-size :pointer))))))
      (error "Kernel view span exceeds the address range"))
    (list offset (+ offset span))))

(defun check-declared-kernel-overlap (inputs spans destination destination-start count)
  (when (and destination (plusp count))
    (labels ((check (pointers)
               (let* ((output (car (last pointers)))
                      (bytes (vector-element-size (vector-type destination)))
                      (low (+ (if output (cffi:pointer-address output) 0)
                              (* (if output bytes 1) destination-start)))
                      (high (+ low (* (if output bytes 1) count))))
                 (loop for input in inputs for span in spans for pointer in pointers
                       when (and span (or (and pointer output) (eq input destination)))
                         do (destructuring-bind (first last) span
                              (let* ((scale (if pointer (vector-element-size (vector-type input)) 1))
                                     (base (if pointer (cffi:pointer-address pointer) 0)))
                                (when (and (< first last)
                                           (< low (+ base (* scale last)))
                                           (< (+ base (* scale first)) high))
                                  (error "Kernel destination overlaps a converted or repeated input"))))))))
      #+(or sbcl ccl ecl)
      (call-with-row-pointers (append inputs (list destination)) #'check)
      #-(or sbcl ccl ecl)
      (check (mapcar (lambda (vector) (when (vector-view-p vector) (vector-view-pointer vector)))
                     (append inputs (list destination)))))))

(defun resolve-declared-kernel (inputs specs destination kind start end starts destination-start)
  (let* ((type (declared-kernel-precision inputs specs destination kind))
         (length (vector-length (nth (position 1 specs :key #'second) inputs)))
         (whole-p (null end))
         (start (or start 0)) (end (or end length))
         (offsets nil) (phases nil) (spans nil))
    (unless (and (integerp start) (integerp end)
                 (<= 0 start end (declared-kernel-limit type)))
      (error "Invalid kernel slice"))
    (let ((count (- end start)))
      (loop for input in inputs for (source repeat) in specs for own-start in starts
            do (when (and whole-p (= repeat 1) (/= length (vector-length input)))
                 (error "Non-repeated kernel inputs must have matching lengths"))
               (let* ((offset (if (= repeat 1) (or own-start start)
                                  (+ (or own-start 0) (floor start repeat))))
                      (phase (if (= repeat 1) 0 (mod start repeat))))
                 (push offset offsets)
                 (push phase phases)
                 (push (declared-kernel-span input offset count repeat phase) spans)))
      (let ((output-start (or destination-start start)))
        (when destination
          (when (and whole-p (/= length (vector-length destination)))
            (error "Kernel destination must match the logical input length"))
          (declared-kernel-span destination output-start count 1 0))
        (check-declared-kernel-overlap
         inputs
         (loop for span in (reverse spans) for (source repeat) in specs
               collect (when (or (integer-type-p source) (> repeat 1)) span))
         destination output-start count)
        (values type count (nreverse offsets) (nreverse phases) (nreverse spans) output-start)))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  ;; TODO: scalar SBCL declarations; integrate packed mixed-input execution (#129).
  (defun declared-kernel-loop (arguments specs expression kind reducer type)
    (let* ((offsets (loop for nil in arguments collect (gensym "OFFSET")))
           (phases (loop for nil in arguments collect (gensym "PHASE")))
           (values (loop for nil in arguments collect (gensym "VALUE")))
           (index (gensym "INDEX")) (count (gensym "COUNT"))
           (destination (gensym "DESTINATION")) (output-start (gensym "OUTPUT-START"))
           (body (if reducer (second expression) expression))
           (element (kernel-element-type type))
           (value `(let ,(loop for argument in arguments for value in values
                               for offset in offsets for phase in phases for (source repeat) in specs
                               collect `(,value
                                         ,(let ((load `(aref ,argument
                                                            (+ ,offset ,(if (= repeat 1) index
                                                                            `(floor (+ ,phase ,index) ,repeat))))))
                                            (if (integer-type-p source) `(coerce ,load ',element) load))))
                     (declare (type ,element ,@values) (ignorable ,@values))
                     ,(mask-kernel-scalar-form body arguments values type)))
           (loop-form
             (case reducer
               (count `(loop for ,index below ,count count ,value))
               (any `(loop for ,index below ,count thereis ,value))
               (all `(loop for ,index below ,count always ,value))
               ((sum asum nrm2 minimum maximum argmin argmax)
                (reduction-loop-form reducer type count index value))
               (otherwise
                `(progn
                   (dotimes (,index ,count)
                     (setf (aref ,destination (+ ,output-start ,index))
                           ,(if (eq kind :mask) `(if ,value 1 0) value)))
                   ,destination)))))
      `(lambda (,destination ,output-start ,count inputs offsets phases)
         (declare (ignorable ,destination ,output-start)
                  (type fixnum ,count ,output-start)
                  ,@(unless reducer
                      `((type (simple-array ,(if (eq kind :mask) '(unsigned-byte 8) element) (*))
                              ,destination))))
         (destructuring-bind ,arguments inputs
           (declare (ignorable ,@arguments)
                    ,@(loop for argument in arguments for (source nil) in specs
                            collect `(type (simple-array ,(kernel-element-type (if (integer-type-p source) source type)) (*))
                                           ,argument)))
           (destructuring-bind ,offsets offsets
             (declare (type fixnum ,@offsets) (ignorable ,@offsets))
             (destructuring-bind ,phases phases
               (declare (type fixnum ,@phases) (ignorable ,@phases))
               ,loop-form)))))))

(defun declared-staging-order (inputs specs offsets count destination output-start)
  (if (or (null destination) (zerop count))
      :forward
      (flet ((order (pointers)
               (let* ((output (if (vector-view-p destination) destination
                                  (and (car (last pointers))
                                       (make-vector-view (car (last pointers)) (vector-type destination)
                                                         (vector-length destination)))))
                      (bindings (list (list output output-start 1 :out))))
                 (loop for input in inputs for pointer in pointers for offset in offsets
                       for (source repeat) in specs
                       when (and (= repeat 1) (not (integer-type-p source)))
                         do (when (and output pointer
                                       (/= (vector-element-size (vector-type input))
                                           (vector-element-size (vector-type destination))))
                              (let* ((input-bytes (vector-element-size (vector-type input)))
                                     (output-bytes (vector-element-size (vector-type destination)))
                                     (input-low (+ (cffi:pointer-address pointer) (* offset input-bytes)))
                                     (output-low (+ (cffi:pointer-address (vector-view-pointer output))
                                                    (* output-start output-bytes))))
                                (when (and (< input-low (+ output-low (* count output-bytes)))
                                           (< output-low (+ input-low (* count input-bytes))))
                                  (return-from declared-staging-order :whole))))
                            (push (list (if (vector-view-p input) input
                                            (and pointer (make-vector-view pointer (vector-type input)
                                                                          (vector-length input))))
                                        offset 1 :in) bindings))
                 (staging-order bindings count))))
        #+(or sbcl ccl ecl)
        (call-with-row-pointers (append inputs (list destination)) #'order)
        #-(or sbcl ccl ecl)
        (order (mapcar (lambda (input) (when (vector-view-p input) (vector-view-pointer input)))
                        (append inputs (list destination)))))))

(defun declared-input-value (input offset index repeat phase type)
  (let ((position (+ offset (floor (+ phase index) repeat))))
    (coerce (if (vector-view-p input)
                (cffi:mem-aref (vector-view-pointer input)
                               (third (numeric-type (vector-type input))) position)
                (aref input position))
            (kernel-element-type type))))

(defun run-declared-lisp (runner inputs specs offsets phases count destination output-start
                          reducer type tree)
  (if (not (some #'vector-view-p (append inputs (when destination (list destination)))))
      (funcall runner destination output-start count inputs offsets phases)
      (let* ((order (declared-staging-order inputs specs offsets count destination output-start))
             (size (if (eq order :whole) (max 1 count) *view-block-size*))
             (step (if (eq order :reverse) (- size) size))
             (first (if (and (eq order :reverse) (plusp count)) (* size (floor (1- count) size)) 0))
             (combine (case reducer
                        ((sum asum) (sum-combiner type))
                        (nrm2 #'norm-combiner)
                        ((minimum maximum) (extremum-combiner (eq reducer 'maximum)))
                        ((argmin argmax) (index-combiner (eq reducer 'argmax)))
                        (count #'count-combiner) (any #'any-combiner) (all #'all-combiner)))
             (buffers (loop for input in inputs for (nil repeat) in specs
                            collect (when (vector-view-p input)
                                      (staging-buffer input (+ 1 (ceiling (min count size) repeat))))))
             (output-buffer (when (and destination (vector-view-p destination))
                              (staging-buffer destination (min count size))))
             (result nil))
        (loop for base = first then (+ base step)
              while (< -1 base (max 1 count))
              for length = (min size (- count base))
              do (let ((block-inputs nil) (block-offsets nil) (block-phases nil))
                   (loop for input in inputs for buffer in buffers
                         for offset in offsets for phase in phases for (nil repeat) in specs
                         do (multiple-value-bind (shift next-phase) (floor (+ phase base) repeat)
                              (let ((next-offset (+ offset shift)))
                                (when buffer
                                  (stage-in input next-offset 1
                                            (if (zerop length) 0 (1+ (floor (+ next-phase length -1) repeat)))
                                            buffer))
                                (push (or buffer input) block-inputs)
                                (push (if buffer 0 next-offset) block-offsets)
                                (push next-phase block-phases))))
                   (let ((value (funcall runner (or output-buffer destination)
                                         (if output-buffer 0 (+ output-start base)) length
                                         (nreverse block-inputs) (nreverse block-offsets)
                                         (nreverse block-phases))))
                     (when output-buffer
                       (stage-out output-buffer destination (+ output-start base) 1 length))
                     (when (and value (member reducer '(argmin argmax)))
                       (let* ((local value)
                              (elements (loop for input in inputs for offset in offsets
                                              for phase in phases for (nil repeat) in specs
                                              collect (make-array 1 :element-type (kernel-element-type type)
                                                                  :initial-element
                                                                  (declared-input-value input offset (+ base local)
                                                                                        repeat phase type)))))
                         (setf value (cons local (kernel-tree-value tree type elements
                                                                   (zero-offsets (length inputs)) 0 t)))))
                     (setf result (if combine (funcall combine result base value) value)))))
        (if destination destination (if (member reducer '(argmin argmax)) (staged-index result) result)))))

(declaim (notinline call-declared-native))
(defun call-declared-native (program type descriptors input-count output rows count mode wide index
                              &optional (scale 1d0))
  (unwind-protect
       (check-native-kernel-status
        (funcall (ecase type (:f32 #'%native-kernel-inputs-f32) (:f64 #'%native-kernel-inputs-f64))
                 (native-program-code program) (native-program-code-length program)
                 (native-program-constants program) descriptors input-count output rows count
                 (native-program-scratch-count program) mode scale wide index))
    (keep-native-program-alive program)))

(defun call-with-declared-native-storage (inputs spans destination output-start output-count function)
  (let ((vectors (append inputs (when destination (list destination)))))
    (labels ((run (pointers)
               (let ((copies nil) (native-pointers nil))
                 (unwind-protect
                      (progn
                        (loop for vector in vectors for pointer in pointers
                              for index from 0
                              for output-p = (>= index (length inputs))
                              for span = (if output-p (list output-start (+ output-start output-count))
                                             (nth index spans))
                              for foreign = (third (numeric-type (vector-type vector)))
                              do (destructuring-bind (low high) span
                                   (let ((memory
                                           (if (or (vector-view-p vector) (eq *native-array-access* :pointer))
                                               (element-pointer pointer foreign low)
                                               (let ((copy (cffi:foreign-alloc foreign :count (max 1 (- high low)))))
                                                 (push copy copies)
                                                 (when (cffi:null-pointer-p copy)
                                                   (error "Unable to allocate native kernel input storage"))
                                                 (unless output-p (copy-to-foreign vector copy foreign low (- high low)))
                                                 copy))))
                                     (push memory native-pointers))))
                        (setf native-pointers (nreverse native-pointers))
                        (multiple-value-prog1 (funcall function native-pointers)
                          (when (and destination (not (vector-view-p destination))
                                     (eq *native-array-access* :copy))
                            (copy-from-foreign (car (last native-pointers)) destination
                                               (third (numeric-type (vector-type destination)))
                                               output-start output-count))))
                   (mapc #'cffi:foreign-free copies)))))
      #+(or sbcl ccl ecl)
      (call-with-row-pointers vectors #'run)
      #-(or sbcl ccl ecl)
      (run (mapcar (lambda (vector) (when (vector-view-p vector) (vector-view-pointer vector))) vectors)))))

(defun run-declared-native (program type inputs specs offsets phases spans destination output-start
                            rows count kind reducer &optional strides)
  (call-with-declared-native-storage
   inputs spans destination output-start (if strides rows count)
   (lambda (pointers)
     (let ((foreign (third (numeric-type type))))
       (cffi:with-foreign-objects ((descriptors '(:struct kernel-input) (length inputs))
                                  (value foreign) (wide :double) (index :size))
         (loop for input in inputs for pointer in pointers for (nil repeat) in specs
               for offset in offsets for phase in phases for (low high) in spans
               for stride in (or strides (make-list (length inputs) :initial-element 0))
               for j from 0
               for descriptor = (cffi:mem-aptr descriptors '(:struct kernel-input) j)
               do (setf (cffi:foreign-slot-value descriptor '(:struct kernel-input) 'data) pointer
                        (cffi:foreign-slot-value descriptor '(:struct kernel-input) 'length) (- high low)
                        (cffi:foreign-slot-value descriptor '(:struct kernel-input) 'start) (- offset low)
                        (cffi:foreign-slot-value descriptor '(:struct kernel-input) 'type)
                        (position (vector-type input) *numeric-types* :key #'first)
                        (cffi:foreign-slot-value descriptor '(:struct kernel-input) 'repeat) repeat
                        (cffi:foreign-slot-value descriptor '(:struct kernel-input) 'phase) phase
                        (cffi:foreign-slot-value descriptor '(:struct kernel-input) 'stride) stride))
         (let ((output (if destination (car (last pointers)) value)))
           (flet ((call (mode &optional (scale 1d0))
                    (call-declared-native program type descriptors (length inputs) output
                                          rows count mode wide index scale))
                  (read-value () (cffi:mem-ref value foreign)))
             (cond
               (strides (call 1) destination)
               ((null reducer) (call (if (eq kind :mask) 10 0)) destination)
               ((eq reducer 'sum) (call 1) (read-value))
               ((eq reducer 'asum) (call 6) (read-value))
               ((member reducer '(minimum maximum argmin argmax))
                (when (plusp count)
                  (call (+ 2 (position reducer '(minimum maximum argmin argmax))))
                  (if (member reducer '(argmin argmax)) (cffi:mem-ref index :size) (read-value))))
               ((member reducer '(count any all))
                (call (+ 11 (position reducer '(count any all))))
                (if (eq reducer 'count) (cffi:mem-ref index :size)
                    (not (zerop (cffi:mem-ref index :size)))))
               ((eq reducer 'nrm2)
                (call 7)
                (let ((sum (cffi:mem-ref wide :double)))
                  (coerce
                   (if (or (eq type :f32) (<= +nrm2-fast-lower+ sum most-positive-double-float))
                       (sqrt sum)
                       (progn
                         (call 8)
                         (let ((scale (cffi:mem-ref wide :double)))
                           (if (zerop scale) 0d0
                               (progn (call 9 scale)
                                      (* scale (sqrt (cffi:mem-ref wide :double))))))))
                   (kernel-element-type type))))))))))))

(defun execute-declared-kernel (runners programs bytes constants scratch type inputs specs offsets phases
                                spans destination output-start rows count kind reducer tree &optional strides)
  (let ((slot (if (eq type :f32) 0 1)))
    (if (and (eq *backend* :native) *native-kernel-inputs-available-p*
             (or (null strides) (and (plusp rows) (plusp count))))
        (run-declared-native
         (or (aref programs slot)
             (setf (aref programs slot) (make-native-program bytes constants scratch type)))
         type inputs specs offsets phases spans destination output-start rows count kind reducer strides)
        (let ((runner (aref runners slot)))
          (if strides
              (progn
                (dotimes (row rows)
                  (let ((value (run-declared-lisp
                                runner inputs specs
                                (mapcar (lambda (offset stride) (+ offset (* row stride))) offsets strides)
                                phases count nil 0 reducer type tree)))
                    (if (vector-view-p destination)
                        (setf (cffi:mem-aref (vector-view-pointer destination)
                                             (third (numeric-type type)) (+ output-start row)) value)
                        (setf (aref destination (+ output-start row)) value))))
                destination)
              (run-declared-lisp runner inputs specs offsets phases count destination output-start
                                 reducer type tree))))))

(defun run-declared-rows (runners programs bytes constants scratch inputs specs destination
                          destination-start rows row-length starts strides kind reducer tree)
  (let* ((type (declared-kernel-precision inputs specs destination kind))
         (limit (declared-kernel-limit type))
         (output-start destination-start))
    (unless (and (eq reducer 'sum) destination
                 (typep rows `(integer 0 ,limit))
                 (typep row-length `(integer 0 ,limit)))
      (error "Declared-input batching requires float SUM, destination, rows and row-length"))
    (declared-kernel-span destination output-start rows 1 0)
    (let* ((offsets (mapcar (lambda (start) (or start 0)) starts))
           (strides (loop for stride in strides for (nil repeat) in specs
                          collect (if (null stride) (ceiling row-length repeat) stride)))
           (phases (zero-offsets (length inputs)))
           (spans
             (loop for input in inputs for offset in offsets for stride in strides for (nil repeat) in specs
                   collect
                   (progn
                     (unless (and (typep stride '(signed-byte 64))
                                  (typep offset `(integer 0 ,(declared-kernel-limit (vector-type input)))))
                       (error "Invalid kernel row start or stride"))
                     (let* ((shift (if (and (plusp rows) (plusp row-length)) (* (1- rows) stride) 0))
                            (low (+ offset (min shift 0)))
                            (high (+ offset (max shift 0)
                                     (if (plusp rows) (ceiling row-length repeat) 0))))
                       (unless (<= 0 low high (vector-length input)
                                   (declared-kernel-limit (vector-type input)))
                         (error "Kernel row span is out of bounds"))
                       (declared-kernel-span input low (- high low) 1 0)
                       (list low high))))))
      (check-declared-kernel-overlap inputs spans destination output-start rows)
      (execute-declared-kernel runners programs bytes constants scratch type inputs specs offsets phases
                               spans destination output-start rows row-length kind reducer tree strides))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun declared-kernel-expansion (name declarations expression)
    (multiple-value-bind (arguments specs) (normalize-kernel-inputs declarations)
      (let* ((outer (and (consp expression) (first expression)))
             (reducer (and (member outer (list* 'count 'any 'all *kernel-reducers*)) outer))
             (body (if reducer (second expression) expression))
             (kind (mask-kernel-kind body arguments))
             (destination (unless reducer (gensym "DESTINATION")))
             (starts (mapcar (lambda (argument) (intern (format nil "~A-START" argument))) arguments))
             (strides (mapcar (lambda (argument) (intern (format nil "~A-ROW-STRIDE" argument))) arguments))
             (batch-p (eq reducer 'sum))
             (supplied (loop repeat (+ 3 (length arguments)) collect (gensym "SUPPLIED")))
             (programs (gensym "PROGRAMS")) (runners (gensym "RUNNERS"))
             (type (gensym "TYPE")) (count (gensym "COUNT")) (offsets (gensym "OFFSETS"))
             (phases (gensym "PHASES")) (spans (gensym "SPANS")) (output-start (gensym "OUTPUT-START"))
             (tree (mask-kernel-tree body arguments))
             (lowered (multiple-value-list (lower-kernel tree)))
             (bytes (kernel-bytes (first lowered)))
             (constants (second lowered)) (scratch (third lowered)))
        (when (and reducer (/= 2 (length expression))) (error "A reduction needs one expression"))
        (when (and reducer (not (eq kind (if (member reducer '(count any all)) :mask :numeric))))
          (error "Kernel reduction has the wrong expression kind"))
        `(let ((,programs (make-array 2 :initial-element nil))
               (,runners
                 (vector ,@(loop for type in '(:f32 :f64)
                                 for form = (declared-kernel-loop arguments specs expression kind reducer type)
                                 collect #+ecl `(let ((runner ,form))
                                                  (if (compiled-function-p runner) runner
                                                      (compile-native-kernel-runner ',form runner)))
                                         #-ecl form))))
           (defun ,name (,@(when destination (list destination)) ,@arguments
                         &key (start nil start-p) (end nil end-p)
                         ,@(when destination '(destination-start)) ,@starts
                         ,@(when batch-p
                             `((rows nil rows-p) (row-length nil ,(first supplied))
                               (destination nil ,(second supplied))
                               (destination-start 0 ,(third supplied))
                               ,@(loop for stride in strides for flag in (cdddr supplied)
                                       collect `(,stride nil ,flag)))))
             (declare (ignorable start-p end-p))
             ,@(when batch-p
                 `((when rows-p
                     (when (or start-p end-p) (error "Batch calls reject :START and :END"))
                     (unless (and ,(first supplied) ,(second supplied))
                       (error "Batch calls require :ROW-LENGTH and :DESTINATION"))
                     ,@(loop for stride in strides for flag in (cdddr supplied)
                             collect `(when (and ,flag (not (typep ,stride '(signed-byte 64))))
                                        (error "Invalid kernel row stride")))
                     (return-from ,name
                       (run-declared-rows ,runners ,programs ',bytes ',constants ,scratch
                                          (list ,@arguments) ',specs destination destination-start
                                          rows row-length (list ,@starts) (list ,@strides) ',kind ',reducer ',tree)))
                   (when (or ,@supplied) (error "Batch keywords require :ROWS"))))
             ,(argument-index-result reducer (gensym "RESULT")
                `(multiple-value-bind (,type ,count ,offsets ,phases ,spans ,output-start)
                     (resolve-declared-kernel (list ,@arguments) ',specs ,destination ',kind start end
                                              (list ,@starts) ,(when destination 'destination-start))
                   (execute-declared-kernel ,runners ,programs ',bytes ',constants ,scratch
                                            ,type (list ,@arguments) ',specs ,offsets ,phases ,spans
                                            ,destination ,output-start 1 ,count ',kind ',reducer ',tree)))))))))
