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
            (with-function-replaced (trivial-garbage:finalize
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
  (loop repeat 30
        do (trivial-garbage:gc :full t)
           (when (funcall predicate) (return t))
           (sleep 0.01)))

(defun counted-finalizer (callback counter)
  (lambda () (funcall callback) (incf (car counter))))

(defun track-finalizers (finalize counter &optional weak registered)
  (lambda (owner callback)
    (when registered (incf (car registered)))
    (when weak (setf (car weak) (trivial-garbage:make-weak-pointer owner)))
    (funcall finalize owner (counted-finalizer callback counter))))

(defun make-tracked-native-program (released weak)
  (with-function-replaced (trivial-garbage:finalize
                           (track-finalizers #'trivial-garbage:finalize released weak))
    (simd::make-native-program '(1 255 0 0) '(1) 0))
  nil)

(defun run-on-test-thread (function)
  (if bordeaux-threads:*supports-threads-p*
      (bordeaux-threads:join-thread (bordeaux-threads:make-thread function))
      (funcall function)))

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

(defun exercise-native-redefinitions (released)
  (let ((registered (list 0))
        (old nil)
        (input (values-for 5 'single-float 1))
        (output (make-array 5 :element-type 'single-float)))
    (with-backend (:native)
      (with-function-replaced (trivial-garbage:finalize
                               (track-finalizers #'trivial-garbage:finalize released nil registered))
        (setf old (define-native-lifetime-kernel 1))
        (funcall old output input)
        (run-on-test-thread (lambda () (replace-native-lifetime-kernels output input))))
      (unless (= 5 (car registered))
        (error "Expected five native programs, registered ~D" (car registered)))
      (unless (await-collection (lambda () (= 4 (car released))))
        (error "Unreachable redefinitions retained their programs (~D/4 released)"
               (car released)))
      (funcall old output input)
      (dotimes (i 5)
        (unless (= (1+ (aref input i)) (aref output i))
          (error "Retained kernel returned the wrong result")))))
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
(test ecl-native-runner-compilation-and-cache
  (let ((calls 0) (compile-runner #'simd::compile-native-kernel-runner))
    (eval '(simd:define-kernel lazy-native-runner (a) (+ a 2)))
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
    (fmakunbound 'lazy-native-runner)))

#+ecl
(test ecl-native-runner-failure-fallback
  (let ((fallback (lambda () :fallback)))
    (is (eq fallback (simd::compile-native-kernel-runner 'invalid-form fallback)))
    (is (eq :compiled (funcall (simd::compile-native-kernel-runner
                               '(lambda () :compiled) fallback))))
    (let ((program (simd::%make-native-program)) (attempts 0))
      (with-function-replaced (simd::compile-native-kernel-runner
                               (lambda (form ignored)
                                 (declare (ignore form ignored))
                                 (incf attempts)
                                 fallback))
        (is (eq fallback (simd::ensure-native-kernel-runner program '(lambda () nil) fallback)))
        (is (eq fallback (simd::ensure-native-kernel-runner program '(lambda () nil) fallback)))
        (is (= 1 attempts)))))
  (when simd::*native-available-p*
    (eval '(simd:define-kernel fallback-native-runner (a) (simd:sum (sqrt a))))
    (let ((attempts 0))
      (with-function-replaced (simd::compile-native-kernel-runner
                               (lambda (form fallback)
                                 (declare (ignore form))
                                 (incf attempts)
                                 fallback))
        (with-backend (:native)
          (is (= 10.0 (fallback-native-runner
                      (make-array 5 :element-type 'single-float :initial-element 4.0))))
          (signals error (fallback-native-runner
                          (make-array 5 :element-type 'single-float :initial-element -1.0)))
          (is (= 1 attempts))))
      (fmakunbound 'fallback-native-runner))))

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
      (with-function-replaced (trivial-garbage:finalize
                               (track-finalizers #'trivial-garbage:finalize counter))
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
    (let ((released (list 0)))
      (with-function-replaced (simd::compile-native-kernel-runner
                               (lambda (form fallback) (declare (ignore form)) fallback))
        (finishes (exercise-native-redefinitions released))))))
