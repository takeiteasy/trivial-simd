(asdf:defsystem "trivial-simd"
  :description "Bulk SIMD operations for Common Lisp numeric vectors"
  :author "George Watson"
  :license "MIT"
  :version "0.1.0"
  :depends-on ("cffi" "trivial-garbage")
  :serial t
  :components ((:file "package")
               (:file "numeric-types")
               (:file "integer-lisp")
               (:file "native")
               (:file "kernel-math")
               (:file "kernel-fma-arm64")
               (:file "backends")
               (:file "strided")
               (:file "operations")
               (:file "small-arrays")
               (:file "extended-operations")
               (:file "complex-operations")
               (:file "reductions")
               (:file "copy-operations")
               (:file "kernel")
               (:file "extension"))
  :in-order-to ((asdf:test-op (asdf:test-op "trivial-simd/tests"))))

(asdf:defsystem "trivial-simd/tests"
  :depends-on ("trivial-simd" "fiveam" "bordeaux-threads")
  :serial t
  :components ((:file "tests/package")
               (:file "tests/operations")
               (:file "tests/small-arrays")
               (:file "tests/extended")
               (:file "tests/copy")
               (:file "tests/strides")
               (:file "tests/native")
               (:file "tests/kernels")
               (:file "tests/integers")
               (:file "tests/complex")
               (:file "tests/reductions")
               (:file "tests/kernel-reductions")
               (:file "tests/extension"))
  :perform (asdf:test-op (operation component)
             (declare (ignore operation component))
             (unless (uiop:symbol-call :fiveam :run! :trivial-simd)
               (error "trivial-simd tests failed"))))
