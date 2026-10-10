(in-package #:trivial-simd)

(defun sbcl-mask-pack (type)
  (let* ((bits (fourth (numeric-type type)))
         (width (/ 128 bits))
         (prefix (if (integer-type-p type)
                     (format nil "~A~D.~D" (if (fifth (numeric-type type)) "S" "U") bits width)
                     (if (eq type :f32) "F32.4" "F64.2")))
         (mask (format nil "U~D.~D" bits width)))
    (values prefix mask width)))

(defun sbcl-mask-symbol (prefix suffix)
  (let ((symbol (find-symbol (concatenate 'string prefix suffix) "SB-SIMD-SSE2")))
    (unless (and symbol (fboundp symbol))
      (error "Missing packed mask operation ~A~A" prefix suffix))
    symbol))

(defun sbcl-mask-safe-p (tree)
  (case (first tree)
    ((:argument :constant) t)
    (:negate (sbcl-mask-safe-p (second tree)))
    (:select (every #'sbcl-mask-safe-p (rest tree)))
    (:operation (and (member (second tree) '(:add :subtract :multiply :eq :ne :lt :le :gt :ge))
                     (every #'sbcl-mask-safe-p (cddr tree))))))

(defun sbcl-mask-form (tree type kind reduction destination arguments offsets d-offset count scalar
                       &optional references byte-mask value-function)
  (unless (and (sbcl-mask-safe-p tree) (member reduction '(nil count any all))
               (not (and (member type '(:f32 :f64)) (member reduction '(any all)))))
    (return-from sbcl-mask-form scalar))
  (when (and (not (integer-type-p type))
             (labels ((arithmetic-p (node)
                        (case (first node)
                          (:operation (or (member (second node) '(:add :subtract :multiply))
                                          (some #'arithmetic-p (cddr node))))
                          (:select (some #'arithmetic-p (rest node)))
                          (:negate (arithmetic-p (second node)))))
                      (selection-p (node)
                        (and (consp node) (or (eq (first node) :select)
                                             (some #'selection-p (rest node))))))
               (and (arithmetic-p tree) (or (member reduction '(any all)) (selection-p tree)))))
    (return-from sbcl-mask-form scalar))
  (handler-case
      (multiple-value-bind (prefix mask width) (sbcl-mask-pack type)
        (let ((index (gensym "I")) (total (gensym "TOTAL"))
              (lanes (loop repeat width collect (gensym "LANE")))
              (cache (make-hash-table :test #'equal)) (assignments nil))
          (labels ((sym (suffix) (sbcl-mask-symbol prefix suffix))
                   (msym (suffix) (sbcl-mask-symbol mask suffix))
                   (cast (value) `(,(sym "!-FROM-P128") ,value))
                   (mcast (value) `(,(msym "!-FROM-P128") ,value))
                   (walk (node)
                     (or (gethash node cache)
                         (let ((name (gensym "PACK")))
                           (let ((value (expression node)))
                             (push (list name value
                                         (if (and (eq (first node) :operation)
                                                  (member (second node) '(:eq :ne :lt :le :gt :ge)))
                                             mask prefix))
                                   assignments))
                           (setf (gethash node cache) name))))
                   (expression (node)
                     (case (first node)
                       (:argument
                        (let ((position (second node)))
                          (if references
                              `(,(nth position references) ,index)
                              `(,(sym "-AREF") ,(nth position arguments)
                                (+ ,(nth position offsets) ,index)))))
                       (:constant `(,(sym "") ,(kernel-constant-form (second node) type)))
                       (:negate `(,(sym "-") (,(sym "") 0) ,(walk (second node))))
                       (:select
                        (let ((condition (gensym "MASK")))
                          `(let ((,condition ,(if byte-mask
                                                  (if (string= mask "U8.16")
                                                      `(,(msym "/=")
                                                        (,(msym "-AREF") ,(first byte-mask) (+ ,(second byte-mask) ,index))
                                                        (,(msym "") 0))
                                                      `(,(sbcl-mask-symbol (concatenate 'string "MAKE-" mask) "")
                                                        ,@(loop for lane below width
                                                                collect `(if (zerop (aref ,(first byte-mask) (+ ,(second byte-mask) ,index ,lane))) 0 ,(1- (ash 1 (fourth (numeric-type type))))))))
                                                  (walk (second node)))))
                             ,(cast `(,(msym "-OR")
                                      (,(msym "-AND") ,condition ,(mcast (walk (third node))))
                                      (,(msym "-ANDC1") ,condition ,(mcast (walk (fourth node)))))))))
                       (:operation
                        `(,(sym (ecase (second node)
                                  (:add "+") (:subtract "-") (:multiply "*")
                                  (:eq "=") (:ne "/=") (:lt "<") (:le "<=") (:gt ">") (:ge ">=")))
                          ,(walk (third node)) ,(walk (fourth node)))))))
            (let* ((result (walk tree))
                   (ordered (reverse assignments))
                   (packed `(let ,(loop for (name nil pack) in ordered
                                        collect `(,name (,(sbcl-mask-symbol pack "") 0)))
                              (declare ,@(loop for (name nil pack) in ordered
                                               collect `(type ,(sbcl-mask-symbol pack "") ,name)))
                              ,@(loop for (name value) in ordered collect `(setf ,name ,value))
                              ,result)))
              `(let ((,index 0) (,total 0))
                 (declare (type fixnum ,index ,total ,count ,@(remove nil offsets) ,@(when destination (list d-offset)))
                          ,@(unless references `((type (simple-array ,(second (numeric-type type)) (*)) ,@arguments)))
                          ,@(when destination `((type (simple-array ,(if (eq kind :mask) '(unsigned-byte 8) (second (numeric-type type))) (*)) ,destination))))
                 (loop while (<= (+ ,index ,width) ,count) do
                   ,(if (eq kind :numeric)
                        `(setf (,(sym "-AREF") ,destination (+ ,d-offset ,index)) ,packed)
                        `(multiple-value-bind ,lanes (,(msym "-VALUES") ,packed)
                           ,@(loop for lane in lanes for j from 0 collect
                                   (if reduction
                                       `(unless (zerop ,lane) (incf ,total))
                                       `(setf (aref ,destination (+ ,d-offset ,index ,j))
                                              (if (zerop ,lane) 0 1))))))
                   (incf ,index ,width))
                 ,(cond (value-function
                         `(progn
                            (loop while (< ,index ,count) do
                              ,(if reduction
                                   `(when (funcall ,value-function ,index) (incf ,total))
                                   `(setf (aref ,destination (+ ,d-offset ,index))
                                          ,(if (eq kind :mask)
                                               `(if (funcall ,value-function ,index) 1 0)
                                               `(funcall ,value-function ,index))))
                              (incf ,index))
                            ,(if reduction
                                 (ecase reduction
                                   (count total) (any `(plusp ,total)) (all `(= ,total ,count)))
                                 destination)))
                        (reduction
                      `(let ((tail ,(subst `(- ,count ,index) count
                                          (loop for offset in offsets
                                                for shifted = `(+ ,offset ,index)
                                                do (setf scalar (subst shifted offset scalar))
                                                finally (return scalar)))))
                         ,(ecase reduction
                            (count `(+ ,total tail))
                            (any `(or (plusp ,total) tail))
                            (all `(and (= ,total ,index) tail)))))
                        (t (loop for offset in (cons d-offset offsets)
                            do (setf scalar (subst `(+ ,offset ,index) offset scalar))
                            finally (return (subst `(- ,count ,index) count scalar))))))))))
    ;; TODO: missing sb-simd mask operations stay scalar; synthesize packs (#157).
    (error () scalar)))

(defun %sbcl-mask-count (mask count offset)
  (declare (type (simple-array (unsigned-byte 8) (*)) mask)
           (type fixnum count offset))
  (let ((i 0) (total 0))
    (declare (type fixnum i total))
    (loop while (<= (+ i 16) count) do
      (incf total (logcount (sb-simd-sse2:u8.16-movemask
                            (sb-simd-sse2:u8.16/= (sb-simd-sse2:u8.16-aref mask (+ offset i))
                                                 (sb-simd-sse2:u8.16 0)))))
      (incf i 16))
    (loop while (< i count) do
      (unless (zerop (aref mask (+ offset i))) (incf total))
      (incf i))
    total))
