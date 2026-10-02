(defpackage #:trivial-simd/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:simd #:trivial-simd)))

(in-package #:trivial-simd/tests)

(def-suite :trivial-simd)
(in-suite :trivial-simd)

;; The suite exercises the backends; tests/small-arrays.lisp enables the inline path.
(setf simd::*inline-small-arrays* nil)
