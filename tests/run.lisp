(require :asdf)
(unless (find-package :ql)
  (dolist (candidate '("quicklisp/setup.lisp" ".roswell/lisp/quicklisp/setup.lisp"))
    (let ((path (merge-pathnames candidate (user-homedir-pathname))))
      (when (probe-file path)
        (load path)
        (return)))))
;; Prefer this checkout over a copy that Quicklisp local-projects links to.
(let ((root (merge-pathnames "../" (uiop:pathname-directory-pathname *load-truename*))))
  (asdf:initialize-source-registry
   `(:source-registry (:directory ,root) :inherit-configuration)))
(defun run-phase (name thunk)
  (let ((start (get-internal-real-time))
        (timing (uiop:getenv "TRIVIAL_SIMD_TEST_TIMING")))
    (when timing
      (format t "~&START ~A process=~A~%" name (or (uiop:getenv "TRIVIAL_SIMD_TEST_RUN") "1"))
      (finish-output))
    (unwind-protect
         (handler-case (funcall thunk)
           (error (condition)
             (format *error-output* "~A~%" condition)
             (uiop:quit 1)))
      (when timing
        (format t "~&END ~A ~,3Fs~%" name
                (/ (- (get-internal-real-time) start) internal-time-units-per-second))
        (finish-output)))))
(run-phase "library load/compile" (lambda () (asdf:load-system "trivial-simd")))

(when (string-equal (or (uiop:getenv "RUNNER_ARCH") "") "ARM64")
  (unless (let ((machine (string-upcase (machine-type))))
            (or (search "ARM64" machine) (search "AARCH64" machine)))
    (error "Expected an ARM64 Lisp, got ~A" (machine-type))))
(let ((expected (uiop:getenv "TRIVIAL_SIMD_BACKEND")))
  (when (and expected (not (string= expected ""))
             (not (string-equal expected "auto"))
             (not (string-equal expected
                                (symbol-name (trivial-simd:backend)))))
    (error "Expected backend ~A, got ~A" expected (trivial-simd:backend))))
(when (find-package :ql)
  (run-phase "test load/compile"
             (lambda () (uiop:symbol-call :ql :quickload "trivial-simd/tests"))))
(run-phase "suite" (lambda () (asdf:test-system "trivial-simd")))
