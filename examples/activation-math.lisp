(require :asdf)
(asdf:load-system :trivial-simd)

(defpackage #:trivial-simd/activation-examples
  (:use #:cl)
  (:local-nicknames (#:simd #:trivial-simd)))
(in-package #:trivial-simd/activation-examples)

(simd:define-kernel sigmoid (x) (simd:sigmoid x))
(simd:define-kernel silu (x) (* x (simd:sigmoid x)))
(simd:define-kernel gelu-tanh (x)
  (* 0.5 x (+ 1 (tanh (* 0.7978845608028654d0 (+ x (* 0.044715d0 x x x)))))))
(simd:define-kernel log-softmax (x)
  (- (- x (simd:maximum x))
     (log (simd:sum (exp (- x (simd:maximum x)))))))

(dolist (type '(single-float double-float))
  (let ((x (make-array 5 :element-type type :initial-contents
                      (mapcar (lambda (x) (coerce x type)) '(-2 -1 0 1 2))))
        (out (make-array 5 :element-type type)))
    (sigmoid out x)
    (assert (= 0.5 (aref out 2)))
    (silu out x)
    (assert (= 0 (aref out 2)))
    (gelu-tanh out x)
    (assert (= 0 (aref out 2)))
    (log-softmax out x)
    (assert (< (abs (- 1 (reduce #'+ (map 'vector #'exp out)))) 1e-6))))
