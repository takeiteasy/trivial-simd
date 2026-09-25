(asdf:defsystem "trivial-simd"
  :description "Bulk SIMD operations for Common Lisp float vectors"
  :author "George Watson"
  :license "MIT"
  :version "0.1.0"
  :depends-on ("cffi")
  :serial t
  :components ((:file "package")
               (:file "native")
               (:file "backends")
               (:file "operations")
               (:file "kernel"))
  :in-order-to ((asdf:test-op (asdf:test-op "trivial-simd/tests"))))

(asdf:defsystem "trivial-simd/tests"
  :depends-on ("trivial-simd" "fiveam")
  :serial t
  :components ((:file "tests/package")
               (:file "tests/operations")
               (:file "tests/kernels"))
  :perform (asdf:test-op (operation component)
             (declare (ignore operation component))
             (unless (uiop:symbol-call :fiveam :run! :trivial-simd)
               (error "trivial-simd tests failed"))))
