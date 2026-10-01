(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(defmacro with-function-replaced ((name replacement) &body body)
  (let ((original (gensym "ORIGINAL")))
    `(let ((,original (fdefinition ',name)))
       (unwind-protect
            (progn (setf (fdefinition ',name) ,replacement) ,@body)
         (setf (fdefinition ',name) ,original)))))

(test native-program-partial-initialization
  (let ((copy #'simd::foreign-copy)
        (free #'simd::free-native-program-buffers))
    (dolist (failure '(1 2 3 4))
      (let ((copies 0) (frees 0))
        (with-function-replaced (simd::free-native-program-buffers
                                 (lambda (buffers)
                                   (incf frees (count-if #'identity buffers))
                                   (funcall free buffers)))
          (with-function-replaced (simd::foreign-copy
                                   (lambda (&rest arguments)
                                     (when (= failure (incf copies))
                                       (error "Injected allocation failure"))
                                     (apply copy arguments)))
            (with-function-replaced (simd::register-native-program-finalizer
                                     (lambda (owner callback)
                                       (declare (ignore owner callback))
                                       (error "Injected registration failure")))
              (signals error (simd::make-native-program '(1 255 0 0) '(1) 0)))))
        (is (= (1- failure) frees))))
    (let ((frees 0))
      (with-function-replaced (simd::free-native-program-buffers
                               (lambda (buffers)
                                 (incf frees (count-if #'identity buffers))
                                 (funcall free buffers)))
        (signals error (simd::foreign-copy '(1 invalid) :float 'single-float)))
      (is (= 1 frees)))))

(test native-program-finalizer-is-idempotent
  (let* ((buffers (loop repeat 3 collect (cffi:foreign-alloc :uint8)))
         (release (simd::native-program-finalizer buffers)))
    (funcall release)
    (is (every #'null buffers))
    (finishes (funcall release))))

(defun await-collection (predicate)
  (or (loop repeat 100
            do (collect-test-garbage)
               (when (funcall predicate) (return t))
               (sleep 0.01))
      (funcall predicate)))

(defun counted-finalizer (callback counter)
  (lambda () (funcall callback) (incf (car counter))))

(defun track-finalizers (finalize counter &optional weak registered)
  (lambda (owner callback)
    (when registered (incf (car registered)))
    (when weak (setf (car weak) (trivial-garbage:make-weak-pointer owner)))
    (funcall finalize owner (counted-finalizer callback counter))))

(defun make-tracked-native-program (released weak)
  (with-function-replaced (simd::register-native-program-finalizer
                           (track-finalizers #'simd::register-native-program-finalizer released weak))
    (simd::make-native-program '(1 255 0 0) '(1) 0))
  nil)

(defun run-on-test-thread (function)
  (if bordeaux-threads:*supports-threads-p*
      (let* ((result nil) (failure nil)
             #+ecl (cache simd::*native-kernel-runners*)
             #+(and ecl threads) (lock simd::*native-kernel-runner-lock*)
             (thread (bordeaux-threads:make-thread
                      (lambda ()
                        (let (#+ecl (simd::*native-kernel-runners* cache)
                              #+(and ecl threads) (simd::*native-kernel-runner-lock* lock))
                          (handler-case
                              (setf result (multiple-value-list (funcall function)))
                            (error (condition) (setf failure condition))))
                        nil))))
        (bordeaux-threads:join-thread thread)
        ;; CCL joins before its worker has finished releasing its stack.
        (unless (loop repeat 100
                      when (not (bordeaux-threads:thread-alive-p thread)) return t
                      do (sleep 0.01))
          (error "Test worker did not terminate"))
        (when failure (error failure))
        (values-list result))
      (funcall function)))

#+ccl
(defun clear-test-gc-roots ()
  ;; Replace cached compiler metadata keys without removing kernel entries.
  (dolist (table (list ccl::%documentation ccl::*lfun-names*))
    (setf (gethash 'clear-test-gc-roots table) nil)
    (remhash 'clear-test-gc-roots table))
  nil)

(defun collect-test-garbage ()
  #+ccl (clear-test-gc-roots)
  (run-on-test-thread
   (lambda ()
     (trivial-garbage:gc :full t)
     #+ccl (ccl:drain-termination-queue)
     nil)))

(test native-program-is-collected
  (let ((released (list 0))
        (weak (list nil)))
    (run-on-test-thread (lambda () (make-tracked-native-program released weak)))
    (is (await-collection (lambda () (= 1 (car released)))))
    (is (null (trivial-garbage:weak-pointer-value (car weak))))
    (trivial-garbage:gc :full t)
    (is (= 1 (car released)))))

(defun define-native-lifetime-kernel (constant)
  (eval `(simd:define-kernel lifetime-kernel (a) (+ a ,constant)))
  (symbol-function 'lifetime-kernel))

(defun replace-native-lifetime-kernels (output input)
  (with-backend (:native)
    (loop for constant from 2 to 5
          do (funcall (define-native-lifetime-kernel constant) output input)))
  (fmakunbound 'lifetime-kernel)
  nil)

(defun lifetime-finalizer (callback released reclaimed index)
  (lambda ()
    (funcall callback)
    (incf (aref reclaimed index))
    (incf (car released))))

(defun observe-lifetime-programs (finalize released registered owners reclaimed)
  (lambda (owner callback)
    (let ((index (car registered)))
      (incf (car registered))
      (setf (aref owners index)
            #+ccl (ccl::%cons-population (list owner) ccl::$population_weak-list nil)
            #-ccl (trivial-garbage:make-weak-pointer owner))
      (funcall finalize owner (lifetime-finalizer callback released reclaimed index)))))

(defun live-lifetime-programs (owners)
  (loop for weak across owners for index from 1
        when (and weak
                  #+ccl (ccl::population-data weak)
                  #-ccl (trivial-garbage:weak-pointer-value weak)) collect index))

(defun exercise-retained-native-kernel (released owners reclaimed retained)
  (let ((registered (list 0))
        (input (values-for 5 'single-float 1))
        (output (make-array 5 :element-type 'single-float)))
    (with-backend (:native)
      (with-function-replaced (simd::register-native-program-finalizer
                               (observe-lifetime-programs #'simd::register-native-program-finalizer
                                                          released registered owners reclaimed))
        (run-on-test-thread
         (lambda ()
           (with-backend (:native)
             (setf (car retained) (define-native-lifetime-kernel 1))
             (funcall (car retained) output input))
           nil))
        (run-on-test-thread (lambda () (replace-native-lifetime-kernels output input))))
      (unless (= 5 (car registered))
        (error "Expected five native programs, registered ~D" (car registered)))
      (unless (await-collection (lambda () (= 4 (car released))))
        (error "Unreachable redefinitions retained their programs (~D/4 released; live ~S; finalizers ~S)"
               (car released) (live-lifetime-programs owners) reclaimed))
      (unless (equalp #(0 1 1 1 1) reclaimed)
        (error "Incorrect retained-program ownership: live ~S; finalizers ~S"
               (live-lifetime-programs owners) reclaimed))
      (run-on-test-thread
       (lambda () (with-backend (:native) (funcall (car retained) output input)) nil))
      (dotimes (i 5)
        (unless (= (1+ (aref input i)) (aref output i))
          (error "Retained kernel returned the wrong result")))))
  nil)

(defun exercise-native-redefinitions (released)
  (let ((owners (make-array 5 :initial-element nil))
        (reclaimed (make-array 5 :initial-element 0))
        (retained (list nil)))
    (exercise-retained-native-kernel released owners reclaimed retained)
    (run-on-test-thread (lambda () (setf (car retained) nil)))
    (unless (await-collection (lambda () (= 5 (car released))))
      (error "Dropped retained function kept its program: live ~S; finalizers ~S"
             (live-lifetime-programs owners) reclaimed))
    (unless (and (null (live-lifetime-programs owners))
                 (every (lambda (count) (= 1 count)) reclaimed))
      (error "Native programs were not released exactly once: ~S" reclaimed)))
  nil)

(test native-program-redefinition
  (when simd::*native-available-p*
    (let ((released (list 0)))
      (finishes (exercise-native-redefinitions released)))))

(defun concurrent-kernel-worker (gate elementwise reduction mode type)
  (lambda ()
    (handler-case
        (let ((input (values-for 640 type 1))
              (output (make-array 640 :element-type type :initial-element (coerce -1 type))))
          (bordeaux-threads:with-lock-held (gate))
          (with-backend (mode)
            (loop repeat 4
                  do (funcall elementwise output input :end 601 :destination-start 5 :a-start 11)
                     (unless (and (= -1 (aref output 4)) (= -1 (aref output 606))
                                  (= 14 (aref output 5)) (= 614 (aref output 605))
                                  (= (funcall reduction input :end 601 :a-start 11)
                                     (coerce (* 601 (/ (+ 14 614) 2)) type)))
                       (error "Concurrent kernel returned the wrong result"))))
          :ok)
      (error (condition) condition))))

(test concurrent-native-kernel-first-use
  (when (and simd::*native-available-p* bordeaux-threads:*supports-threads-p*)
    (eval '(simd:define-kernel concurrent-elementwise (a) (+ a 2)))
    (eval '(simd:define-kernel concurrent-reduction (a) (simd:sum (+ a 2))))
    (let ((gate (bordeaux-threads:make-lock "kernel start")) (threads nil))
      (unwind-protect
           (progn
             (bordeaux-threads:with-lock-held (gate)
               (dolist (type '(single-float double-float))
                 (dolist (mode '(:native :native-copy))
                   (push (bordeaux-threads:make-thread
                          (concurrent-kernel-worker gate
                                                    (symbol-function 'concurrent-elementwise)
                                                    (symbol-function 'concurrent-reduction) mode type))
                         threads))))
             (dolist (thread threads)
               (is (eq :ok (bordeaux-threads:join-thread thread)))))
        (fmakunbound 'concurrent-elementwise)
        (fmakunbound 'concurrent-reduction)))))

#+ecl
(defmacro with-fresh-native-runners (&body body)
  `(let ((simd::*native-kernel-runners* (make-hash-table :test 'eql))
         #+threads (simd::*native-kernel-runner-lock* (mp:make-lock :name "test runners")))
     ,@body))

#+ecl
(test ecl-native-runner-compilation-and-cache
  (with-fresh-native-runners
    (let ((calls 0) (compile-runner #'simd::compile-native-kernel-runner))
      (eval '(simd:define-kernel lazy-native-runner (a) (+ a 2)))
      (unwind-protect
           (with-function-replaced (simd::compile-native-kernel-runner
                                    (lambda (form fallback)
                                      (incf calls)
                                      (funcall compile-runner form fallback)))
             (let ((input (values-for 5 'single-float 1))
                   (output (make-array 5 :element-type 'single-float)))
               (with-backend (:lisp) (lazy-native-runner output input))
               (is (zerop calls))
               (when simd::*native-available-p*
                 (with-backend (:native)
                   (lazy-native-runner output input)
                   (lazy-native-runner output input))
                 (is (= 1 calls))
                 (is (= 3 (aref output 0))))))
        (fmakunbound 'lazy-native-runner)))))

#+ecl
(test ecl-native-runner-shared-signatures
  (when simd::*native-available-p*
    (with-fresh-native-runners
      (let ((calls 0) (compile-runner #'simd::compile-native-kernel-runner)
            (definitions `((shared-add (a) (+ a 2) 6)
                           (shared-multiply (a) (* a 3) 12)
                           (shared-spill (a) ,(balanced-expression 9) 2048)
                           (shared-binary (a b) (+ a b) 8)
                           (shared-constant () 7 7)
                           (shared-root-sum (a) (simd:sum (sqrt a)) 2)
                           (shared-binary-sum (a b) (simd:sum (+ a b)) 8))))
        (unwind-protect
             (with-function-replaced (simd::compile-native-kernel-runner
                                      (lambda (form fallback)
                                        (incf calls)
                                        (funcall compile-runner form fallback)))
               (dolist (definition definitions)
                 (destructuring-bind (name arguments expression expected) definition
                   (eval `(simd:define-kernel ,name ,arguments ,expression))
                   (dolist (type '(single-float double-float))
                     (dolist (mode '(:native :native-copy))
                       (dolist (count '(0 5 257))
                         (let* ((input (make-array count :element-type type :initial-element (coerce 4 type)))
                                (output (make-array count :element-type type))
                                (inputs (loop for nil in arguments collect input)))
                           (with-backend (mode)
                             (if (and (consp expression) (eq (first expression) 'simd:sum))
                                 (is (= (* count expected) (apply (symbol-function name) inputs)))
                                 (progn
                                   (is (eq output (apply (symbol-function name) output inputs)))
                                   (is (every (lambda (value) (= value expected)) output)))))))))))
               (is (= 5 calls))
               (is (= 5 (hash-table-count simd::*native-kernel-runners*)))
               (is (every #'functionp (loop for runner being the hash-values of simd::*native-kernel-runners*
                                           collect runner))))
          (dolist (definition definitions) (fmakunbound (first definition))))))))

#+ecl
(test ecl-native-runner-failure-fallback
  (with-fresh-native-runners
    (let ((fallback (lambda () :fallback)))
      (is (eq fallback (simd::compile-native-kernel-runner 'invalid-form fallback)))
      (is (eq fallback
              (simd::compile-native-kernel-runner
               '(lambda () nil) fallback
               (lambda (name form)
                 (declare (ignore name form))
                 (values (lambda () :invalid) nil t)))))
      (is (eq :compiled (funcall (simd::compile-native-kernel-runner
                                 '(lambda () :compiled) fallback))))
      (let ((program (simd::%make-native-program)) (other (simd::%make-native-program))
            (other-fallback (lambda () :other)) (attempts 0))
        (with-function-replaced (simd::compile-native-kernel-runner
                                 (lambda (form fallback)
                                   (declare (ignore form))
                                   (incf attempts)
                                   fallback))
          (is (eq fallback (simd::ensure-native-kernel-runner program 2 '(lambda () nil) fallback)))
          (is (eq fallback (simd::ensure-native-kernel-runner program 2 '(lambda () nil) fallback)))
          (is (eq other-fallback (simd::ensure-native-kernel-runner other 2 '(lambda () nil) other-fallback)))
          (is (eq :failed (gethash 2 simd::*native-kernel-runners*)))
          (is (= 1 attempts)))))
    (when simd::*native-available-p*
      (let ((attempts 0))
        (unwind-protect
             (with-function-replaced (simd::compile-native-kernel-runner
                                      (lambda (form fallback)
                                        (declare (ignore form))
                                        (incf attempts)
                                        fallback))
               (eval '(simd:define-kernel fallback-native-runner (a) (simd:sum (sqrt a))))
               (eval '(simd:define-kernel other-fallback-runner (a) (simd:sum (+ a 3))))
               (dolist (type '(single-float double-float))
                 (dolist (mode '(:native :native-copy))
                   (with-backend (mode)
                     (is (= 10 (fallback-native-runner
                                (make-array 5 :element-type type :initial-element (coerce 4 type)))))
                     (is (= 35 (other-fallback-runner
                                (make-array 5 :element-type type :initial-element (coerce 4 type)))))
                     (signals error (fallback-native-runner
                                     (make-array 5 :element-type type :initial-element (coerce -1 type)))))))
               (is (= 1 attempts)))
          (fmakunbound 'fallback-native-runner)
          (fmakunbound 'other-fallback-runner))))))

#+(and ecl threads)
(test ecl-shared-runner-concurrent-first-use
  (when simd::*native-available-p*
    (with-fresh-native-runners
      (let ((cache simd::*native-kernel-runners*) (lock simd::*native-kernel-runner-lock*)
            (gate (bordeaux-threads:make-lock "shared runner start"))
            (compile-runner #'simd::compile-native-kernel-runner)
            (calls 0) (threads nil) (names nil))
        (unwind-protect
             (with-function-replaced (simd::compile-native-kernel-runner
                                      (lambda (form fallback)
                                        (incf calls)
                                        (funcall compile-runner form fallback)))
               (bordeaux-threads:with-lock-held (gate)
                 (dolist (type '(single-float double-float))
                   (dolist (mode '(:native :native-copy))
                     (let ((elementwise (gensym "CONCURRENT")) (reduction (gensym "SUM")))
                       (push elementwise names)
                       (push reduction names)
                       (eval `(simd:define-kernel ,elementwise (a) (+ a 2)))
                       (eval `(simd:define-kernel ,reduction (a) (simd:sum (+ a 2))))
                       (let ((worker (concurrent-kernel-worker gate (symbol-function elementwise)
                                                               (symbol-function reduction) mode type)))
                         (push (bordeaux-threads:make-thread
                                (lambda ()
                                  (let ((simd::*native-kernel-runners* cache)
                                        (simd::*native-kernel-runner-lock* lock))
                                    (funcall worker)))) threads))))))
               (dolist (thread threads)
                 (is (eq :ok (bordeaux-threads:join-thread thread)))))
          (dolist (thread threads)
            (when (bordeaux-threads:thread-alive-p thread) (bordeaux-threads:join-thread thread)))
          (dolist (name names) (fmakunbound name)))
        (is (= 2 calls))
        (is (= 2 (hash-table-count cache)))))))

#+ecl
(test ecl-shared-runner-program-redefinition
  (when simd::*native-available-p*
    (with-fresh-native-runners
      (let ((released (list 0)))
        (finishes (exercise-native-redefinitions released))
        (is (functionp (gethash 8 simd::*native-kernel-runners*)))))))

(defun collecting-native-call (function counter)
  (lambda (&rest arguments)
    (trivial-garbage:gc :full t)
    (unless (zerop (car counter))
      (error "Active native program was finalized"))
    (incf (cdr counter))
    (apply function arguments)))

(test native-program-survives-active-call
  (when simd::*native-available-p*
    (let ((counter (cons 0 0)))
      (with-function-replaced (simd::register-native-program-finalizer
                               (track-finalizers #'simd::register-native-program-finalizer counter))
        (with-function-replaced (simd::%native-kernel-f32
                                 (collecting-native-call #'simd::%native-kernel-f32 counter))
          (cffi:with-foreign-object (output :float)
            (simd::call-native-kernel (simd::make-native-program '(1 255 0 0) '(7) 0)
                                     :float (cffi:null-pointer) output 1)
            (is (= 7.0 (cffi:mem-ref output :float))))))
      (is (= 1 (cdr counter))))))

#+ecl
(test ecl-fallback-program-redefinition
  (when simd::*native-available-p*
    (with-fresh-native-runners
      (let ((released (list 0)))
        (with-function-replaced (simd::compile-native-kernel-runner
                                 (lambda (form fallback) (declare (ignore form)) fallback))
          (finishes (exercise-native-redefinitions released))
          (is (eq :failed (gethash 8 simd::*native-kernel-runners*))))))))

(test native-copy-compact-slices
  (dolist (type '(single-float double-float))
    (let* ((input (values-for 65536 type 1))
           (output (values-for 65536 type -100))
           (foreign (simd::foreign-type input)))
      (dolist (count '(0 1 5 257))
        (let ((before (copy-seq output)))
          (cffi:with-foreign-object (pointer foreign (max 1 count))
            (simd::copy-to-foreign input pointer foreign 60000 count)
            (dotimes (i count)
              (is (= (aref input (+ 60000 i)) (cffi:mem-aref pointer foreign i))))
            (simd::copy-from-foreign pointer output foreign 50000 count))
          (dotimes (i count)
            (setf (aref before (+ 50000 i)) (aref input (+ 60000 i))))
          (is (equalp before output)))))))

(test native-copy-transfer-ranges
  (when simd::*native-available-p*
    (dolist (type '(single-float double-float))
      (let ((input (values-for 65536 type 1))
            (output (values-for 1024 type -1))
            (copy #'simd::copy-to-foreign)
            (ranges nil))
        (with-function-replaced (simd::copy-to-foreign
                                 (lambda (vector pointer foreign start count)
                                   (push (list start count) ranges)
                                   (funcall copy vector pointer foreign start count)))
          (with-backend (:native-copy)
            (simd:add! output input input :end 5 :destination-start 10
                       :left-start 60000 :right-start 61000)
            (is (equal '((61000 5) (60000 5)) ranges)))))
      (let* ((input (values-for 65536 type 1))
             (before (copy-seq input)))
        (with-backend (:native-copy)
          (dolist (count '(0 1 5 257))
            (replace input before)
            (simd:add! input input input :end count :destination-start 60001
                       :left-start 60000 :right-start 60002)
            (let ((expected (copy-seq before)))
              (dotimes (i count)
                (setf (aref expected (+ 60001 i))
                      (+ (aref before (+ 60000 i)) (aref before (+ 60002 i)))))
              (is (equalp expected input)))))))))
