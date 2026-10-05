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
               (:file "vector-views")
               (:file "native")
               (:file "kernel-math")
               (:file "kernel-fma-arm64")
               (:file "backends")
               (:file "strided")
               (:file "operations")
               (:file "small-arrays")
               (:file "float-encodings")
               (:file "extended-operations")
               (:file "complex-operations")
               (:file "reductions")
               (:file "copy-operations")
               (:file "kernel")
               (:file "kernel-rows")
               (:file "kernel-inputs"))
  :in-order-to ((asdf:test-op (asdf:test-op "trivial-simd/tests"))))

(asdf:defsystem "trivial-simd/tests"
  :depends-on ("trivial-simd/blas/convenience" "fiveam" "bordeaux-threads")
  :serial t
  :components ((:file "tests/package")
               (:file "tests/diagnostics")
               (:file "tests/operations")
               (:file "tests/small-arrays")
               (:file "tests/extended")
               (:file "tests/float-encodings")
               (:file "tests/copy")
               (:file "tests/strides")
               (:file "tests/native")
               (:file "tests/kernels")
               (:file "tests/integers")
               (:file "tests/blas")
               (:file "tests/complex")
               (:file "tests/reductions")
               (:file "tests/kernel-reductions")
               (:file "tests/kernel-rows")
               (:file "tests/blas-level1")
               (:file "tests/blas-level2")
               (:file "tests/blas-level3")
               (:file "tests/blas-random")
               (:file "tests/blas-native")
               (:file "tests/views")
               (:file "tests/kernel-inputs"))
  :perform (asdf:test-op (operation component)
             (declare (ignore operation component))
             (unless (uiop:symbol-call :trivial-simd/tests :run-tests)
               (error "trivial-simd tests failed"))))

(asdf:defsystem "trivial-simd/blas"
  :description "CBLAS-style Common Lisp BLAS routines"
  :depends-on ("trivial-simd")
  :serial t
  :components ((:file "blas/package")
               (:file "blas/storage")
               (:file "blas/level1")
               (:file "blas/level1-extra")
               (:file "blas/matrix-view")
               (:file "blas/kernel")
               (:file "blas/level2")
               (:file "blas/level3"))
  :in-order-to ((asdf:test-op (asdf:test-op "trivial-simd/tests"))))

(asdf:defsystem "trivial-simd/blas/convenience"
  :description "Convenience wrappers for trivial-simd/blas"
  :depends-on ("trivial-simd/blas")
  :components ((:file "blas/convenience"))
  :in-order-to ((asdf:test-op (asdf:test-op "trivial-simd/tests"))))
