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
              (signals error (simd::make-native-program '(0 255 0 0) '(1) 0)))))
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

(defun track-finalizers (finalize counter &optional weak)
  (lambda (owner callback)
    (when weak (setf (car weak) (trivial-garbage:make-weak-pointer owner)))
    (funcall finalize owner (counted-finalizer callback counter))))

(defun make-tracked-native-program (released weak)
  (with-function-replaced (trivial-garbage:finalize
                           (track-finalizers #'trivial-garbage:finalize released weak))
    (simd::make-native-program '(0 255 0 0) '(1) 0))
  nil)

(test native-program-is-collected
  (let ((released (list 0))
        (weak (list nil)))
    (make-tracked-native-program released weak)
    (is (await-collection (lambda () (= 1 (car released)))))
    (is (null (trivial-garbage:weak-pointer-value (car weak))))
    (trivial-garbage:gc :full t)
    (is (= 1 (car released)))))

(defun define-native-lifetime-kernel (constant)
  (eval `(simd:define-kernel lifetime-kernel (a) (+ a ,constant)))
  (symbol-function 'lifetime-kernel))

(defun exercise-native-redefinitions (released)
  (let ((old nil)
        (input (values-for 5 'single-float 1))
        (output (make-array 5 :element-type 'single-float)))
    (with-backend (:native)
      (with-function-replaced (trivial-garbage:finalize
                               (track-finalizers #'trivial-garbage:finalize released))
        (setf old (define-native-lifetime-kernel 1))
        (funcall old output input)
        (loop for constant from 2 to 5
              do (funcall (define-native-lifetime-kernel constant) output input))
        (fmakunbound 'lifetime-kernel))
      (unless (await-collection (lambda () (= 4 (car released))))
        (error "Unreachable redefinitions retained their programs"))
      (funcall old output input)
      (dotimes (i 5)
        (unless (= (1+ (aref input i)) (aref output i))
          (error "Retained kernel returned the wrong result")))))
  nil)

(test native-program-redefinition
  (when simd::*native-available-p*
    (let ((released (list 0)))
      (finishes (exercise-native-redefinitions released))
      (is (await-collection (lambda () (= 5 (car released))))))))
