(require :asdf)
(handler-case
    (let ((path (compile-file "tests/fma-bench.lisp"
                              :output-file (compile-file-pathname
                                            (merge-pathnames "fma-profile.lisp" (uiop:temporary-directory))))))
      (unless path (error "FMA benchmark compilation failed"))
      (load path)
      (load path))
  (error (condition)
    (format *error-output* "~A~%" condition)
    (uiop:quit 1)))
