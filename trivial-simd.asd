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
               (:file "operations")
               (:file "extended-operations")
               (:file "kernel"))
  :in-order-to ((asdf:test-op (asdf:test-op "trivial-simd/tests"))))

(asdf:defsystem "trivial-simd/tests"
  :depends-on ("trivial-simd/blas/convenience" "fiveam" "bordeaux-threads")
  :serial t
  :components ((:file "tests/package")
               (:file "tests/operations")
               (:file "tests/extended")
               (:file "tests/native")
               (:file "tests/kernels")
               (:file "tests/integers")
               (:file "tests/blas"))
  :perform (asdf:test-op (operation component)
             (declare (ignore operation component))
             (unless (uiop:symbol-call :fiveam :run! :trivial-simd)
               (error "trivial-simd tests failed"))))

(asdf:defsystem "trivial-simd/blas"
  :description "CBLAS-style Common Lisp BLAS routines"
  :depends-on ("trivial-simd")
  :serial t
  :components ((:file "blas/package")
               (:file "blas/level1"))
  :in-order-to ((asdf:test-op (asdf:test-op "trivial-simd/tests"))))

(asdf:defsystem "trivial-simd/blas/convenience"
  :description "Convenience wrappers for trivial-simd/blas"
  :depends-on ("trivial-simd/blas")
  :components ((:file "blas/convenience"))
  :in-order-to ((asdf:test-op (asdf:test-op "trivial-simd/tests"))))
