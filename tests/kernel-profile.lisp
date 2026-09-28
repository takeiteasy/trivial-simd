(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :asdf)
  (load (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname))))

(defmacro load-profile-system ()
  (let ((root (uiop:pathname-directory-pathname
               (or *compile-file-truename* *load-truename*))))
    `(progn
       (asdf:load-asd ,(merge-pathnames "../trivial-simd.asd" root))
       (asdf:load-system "trivial-simd")
       (asdf:load-system "bordeaux-threads")
       (load ,(merge-pathnames "benchmark-timing.lisp" root)))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (load-profile-system))

(macrolet ((define-borrowed-call (suffix)
             `(trivial-simd::define-native (,(format nil "ts_kernel_with_scratch_~A" suffix)
                             ,(intern (string-upcase (format nil "PROFILE-ELEMENTWISE-~A" suffix)))) :int
                (code :pointer) (code-length :size) (constants :pointer)
                (inputs :pointer) (output :pointer) (length :size)
                (slots :size) (scratch :pointer) (capacity :size)))
           (define-borrowed-sum (suffix)
             `(trivial-simd::define-native (,(format nil "ts_kernel_sum_with_scratch_~A" suffix)
                             ,(intern (string-upcase (format nil "PROFILE-SUM-~A" suffix)))) :int
                (code :pointer) (code-length :size) (constants :pointer)
                (inputs :pointer) (output :pointer) (length :size)
                (slots :size) (scratch :pointer) (capacity :size))))
  (define-borrowed-call "f32")
  (define-borrowed-call "f64")
  (define-borrowed-sum "f32")
  (define-borrowed-sum "f64"))

(defstruct profile-pool
  (lock (bordeaux-threads:make-lock "profile scratch"))
  (single (cons nil nil)) (double (cons nil nil)))

(defvar *profile-pool*)
(defvar *profile-original-call* (fdefinition 'trivial-simd::call-native-kernel))
(defvar *profile-baseline-functions* nil)

(defun profile-baseline-call (program type inputs output count &key sum-p)
  (unwind-protect
       (let ((status
               (cffi:foreign-funcall-pointer
                (svref *profile-baseline-functions*
                       (+ (if sum-p 2 0) (if (eq type :float) 0 1)))
                () :pointer (trivial-simd::native-program-code program)
                :size (trivial-simd::native-program-code-length program)
                :pointer (if (eq type :float)
                             (trivial-simd::native-program-f32-constants program)
                             (trivial-simd::native-program-f64-constants program))
                :pointer inputs :pointer output :size count
                :size (trivial-simd::native-program-scratch-count program) :int)))
         (unless (zerop status) (error "Baseline VM failed: ~D" status)))
    (trivial-simd::keep-native-program-alive program)))

(defun profile-cached-call (program type inputs output count &key sum-p)
  (let ((slots (trivial-simd::native-program-scratch-count program)))
    (if (or (zerop count) (zerop slots))
        (funcall *profile-original-call* program type inputs output count :sum-p sum-p)
        (let* ((pool *profile-pool*)
               (entry (if (eq type :float) (profile-pool-single pool) (profile-pool-double pool)))
               (cached nil)
               (scratch nil))
          (unwind-protect
               (progn
                 (bordeaux-threads:with-lock-held ((profile-pool-lock pool))
                   (unless (cdr entry)
                     (unless (car entry)
                       (setf (car entry) (cffi:foreign-alloc type :count (* slots 256))))
                     (setf scratch (car entry) cached t (cdr entry) t)))
                 (unless cached
                   (setf scratch (cffi:foreign-alloc type :count (* slots 256))))
                 (case (funcall (if sum-p
                                   (if (eq type :float) #'profile-sum-f32 #'profile-sum-f64)
                                   (if (eq type :float) #'profile-elementwise-f32 #'profile-elementwise-f64))
                                (trivial-simd::native-program-code program)
                                (trivial-simd::native-program-code-length program)
                                (if (eq type :float)
                                    (trivial-simd::native-program-f32-constants program)
                                    (trivial-simd::native-program-f64-constants program))
                                inputs output count slots scratch slots)
                   (0 nil)
                   (-2 (error "Negative kernel square root operand"))
                   (otherwise (error "Invalid profile scratch storage"))))
            (if cached
                (bordeaux-threads:with-lock-held ((profile-pool-lock pool))
                  (setf (cdr entry) nil))
                (when scratch (cffi:foreign-free scratch)))
            (trivial-simd::keep-native-program-alive program))))))

(defun profile-tree (depth)
  (if (zerop depth) 'a
      `(+ ,(profile-tree (1- depth)) ,(profile-tree (1- depth)))))

(defun profile-overlap ()
  (when bordeaux-threads:*supports-threads-p*
    (eval `(trivial-simd:define-kernel profile-overlap-kernel (a)
             (trivial-simd:sum ,(profile-tree 9))))
    (let ((*profile-pool* (make-profile-pool))
          (trivial-simd::*backend* :native))
      (unwind-protect
           (dolist (cached '(nil t))
             (setf (fdefinition 'trivial-simd::call-native-kernel)
                   (if cached #'profile-cached-call *profile-original-call*))
             (profile-overlap-kernel (make-array 32 :element-type 'single-float :initial-element 0.125))
             (let ((timings nil) (pool *profile-pool*))
               (dotimes (batch 3)
                 (let ((gate (bordeaux-threads:make-lock "profile start"))
                       (threads nil) (start nil))
                   (bordeaux-threads:with-lock-held (gate)
                     (dotimes (worker 4)
                       (let ((input (make-array 32 :element-type 'single-float :initial-element 0.125)))
                         (push (bordeaux-threads:make-thread
                                (lambda ()
                                  (let ((*profile-pool* pool) (trivial-simd::*backend* :native))
                                    (bordeaux-threads:with-lock-held (gate))
                                    (dotimes (i 10000)
                                      (unless (= 2048.0 (profile-overlap-kernel input))
                                        (error "Overlapping profile result mismatch"))))))
                               threads)))
                     (setf start (get-internal-real-time)))
                   (dolist (thread threads) (bordeaux-threads:join-thread thread))
                   (push (/ (* 1.0d6 (- (get-internal-real-time) start))
                            internal-time-units-per-second 40000)
                         timings)))
               (format t "Overlap 4 workers ~A ~,6F us/call~%"
                       (if cached :cache :allocated) (second (sort timings #'<)))))
        (setf (fdefinition 'trivial-simd::call-native-kernel) *profile-original-call*)
        (dolist (entry (list (profile-pool-single *profile-pool*) (profile-pool-double *profile-pool*)))
          (when (car entry) (cffi:foreign-free (car entry))))))))

(defun profile-direct (library program type table output count sum-p reuse &optional baseline-p)
  (cffi:foreign-funcall-pointer
   (cffi:foreign-symbol-pointer
    (format nil "ts_profile~A_~A" (if baseline-p "_baseline" "")
            (if (eq type :float) "f32" "f64")) :library library)
   () :pointer (trivial-simd::native-program-code program)
   :size (trivial-simd::native-program-code-length program)
   :pointer (if (eq type :float)
                (trivial-simd::native-program-f32-constants program)
                (trivial-simd::native-program-f64-constants program))
   :pointer table :pointer output :size count
   :size (trivial-simd::native-program-scratch-count program)
   :int (if sum-p 1 0) :int (if reuse 1 0) :double))

(defun profile-case (name expression type count sum-p direct-library baseline-library)
  (let* ((foreign (if (eq type 'single-float) :float :double))
         (input (make-array count :element-type type :initial-element (coerce 0.125 type)))
         (output (make-array count :element-type type))
         (function nil)
         (*profile-pool* (make-profile-pool))
         (trivial-simd::*backend* :native))
    (eval `(trivial-simd:define-kernel profile-kernel (a) ,(if sum-p `(trivial-simd:sum ,expression) expression)))
    (setf function (if sum-p (lambda () (profile-kernel input))
                       (lambda () (profile-kernel output input))))
    (multiple-value-bind (instructions constants slots)
        (trivial-simd::lower-kernel (trivial-simd::parse-kernel-expression expression '(a)))
      (let ((program (trivial-simd::make-native-program
                      (trivial-simd::kernel-bytes instructions) constants slots)))
        (unwind-protect
             (progn
               (let ((expected (* (coerce 0.125 type) (if (eq name :dot) (coerce 0.125 type) (expt 2 name)))))
                 (flet ((check ()
                          (let ((result (funcall function)))
                            (unless (if sum-p (= result (* count expected))
                                        (every (lambda (x) (= x expected)) result))
                              (error "Profile result mismatch")))))
                   (check)
                   (setf (fdefinition 'trivial-simd::call-native-kernel) #'profile-cached-call)
                   (check)
                   (setf (fdefinition 'trivial-simd::call-native-kernel) *profile-original-call*)))
               (let* ((allocated (benchmark-time function))
                      (cached (progn
                                (setf (fdefinition 'trivial-simd::call-native-kernel) #'profile-cached-call)
                                (benchmark-time function)))
                      (old (when baseline-library
                             (setf (fdefinition 'trivial-simd::call-native-kernel) #'profile-baseline-call)
                             (benchmark-time function))))
                 (setf (fdefinition 'trivial-simd::call-native-kernel) *profile-original-call*)
                 (cffi:with-foreign-object (values foreign (max 1 count))
                   (cffi:with-foreign-object (out foreign (max 1 count))
                     (cffi:with-foreign-object (table :pointer)
                       (dotimes (i count) (setf (cffi:mem-aref values foreign i) (coerce 0.125 type)))
                       (setf (cffi:mem-ref table :pointer) values)
                       (format t "~A ~A ~D ~A ~D ~D ~D ~D ~D ~,6F ~,6F ~,6F ~,6F ~,6F ~,6F~%"
                               name type count (if sum-p :sum :elementwise)
                               (length instructions) (count :spill instructions :key #'first)
                               (count :reload instructions :key #'first) slots
                               (* slots 256 (cffi:foreign-type-size foreign))
                               allocated cached
                               (profile-direct direct-library program foreign table out count sum-p nil)
                               (profile-direct direct-library program foreign table out count sum-p t)
                               (if baseline-library
                                   (profile-direct baseline-library program foreign table out count sum-p nil t)
                                   0) (or old 0)))))))
          (setf (fdefinition 'trivial-simd::call-native-kernel) *profile-original-call*)
          (dolist (entry (list (profile-pool-single *profile-pool*) (profile-pool-double *profile-pool*)))
            (when (car entry) (cffi:foreign-free (car entry)))))))))

(let* ((direct (cffi:load-foreign-library (uiop:getenv "TRIVIAL_SIMD_PROFILE_LIBRARY")))
       (baseline (let ((path (uiop:getenv "TRIVIAL_SIMD_PROFILE_BASELINE")))
                   (when path (cffi:load-foreign-library path))))
       (*profile-baseline-functions*
         (when baseline
           (map 'vector (lambda (name) (cffi:foreign-symbol-pointer name :library baseline))
                '("ts_baseline_kernel_f32" "ts_baseline_kernel_f64"
                  "ts_baseline_kernel_sum_f32" "ts_baseline_kernel_sum_f64")))))
  (format t "~A ~A | ~A | native ~A~%" (lisp-implementation-type)
          (lisp-implementation-version) (machine-type) trivial-simd::*native-array-access*)
  (format t "Case Type Elements Mode Instructions Spills Reloads Slots Bytes Lisp-allocated Lisp-cache C-allocated C-reused C-baseline Lisp-baseline (us/call)~%")
  (dolist (type '(single-float double-float))
    (dolist (count (if (uiop:getenv "TRIVIAL_SIMD_PROFILE_SHORT")
                       '(32 1024) '(32 1024 65536)))
      (dolist (depth '(3 9 10))
        (dolist (sum-p '(nil t))
          (profile-case depth (profile-tree depth) type count sum-p direct baseline)))
      (profile-case :dot '(* a a) type count t direct baseline)))
  (profile-overlap))
