(in-package #:trivial-simd/tests)

(in-suite :trivial-simd)

(test extension-package-exports-the-native-helpers
  (do-external-symbols (symbol '#:trivial-simd/extension)
    (is (eq symbol (find-symbol (symbol-name symbol) '#:trivial-simd)))
    (is (or (boundp symbol) (fboundp symbol)))))
