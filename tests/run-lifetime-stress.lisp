(require :asdf)
(handler-case
    (progn
      (load (merge-pathnames "run.lisp" *load-truename*))
      (unless (symbol-value (find-symbol "*NATIVE-AVAILABLE-P*" :trivial-simd))
        (error "Lifetime stress requires the native library"))
      (dotimes (trial 500)
        (handler-case
            (uiop:symbol-call :trivial-simd/tests :exercise-native-redefinitions (list 0))
          (error (condition)
            (error "Lifetime trial ~D/500: ~A" (1+ trial) condition))))
      (format t "~&Native lifetime stress: 500/500 passed.~%"))
  (error (condition)
    (format *error-output* "~A~%" condition)
    (uiop:quit 1)))
