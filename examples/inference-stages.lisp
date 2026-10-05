(require :asdf)
(unless (find-package :ql)
  (let ((setup (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname))))
    (when (probe-file setup) (load setup))))
(let ((root (merge-pathnames "../" (uiop:pathname-directory-pathname *load-truename*))))
  (asdf:initialize-source-registry
   `(:source-registry (:directory ,root) :inherit-configuration)))
(asdf:load-asd (merge-pathnames "../trivial-simd.asd"
                               (uiop:pathname-directory-pathname *load-truename*)))
(asdf:load-system "trivial-simd")

(trivial-simd:define-kernel softmax (x)
  (/ (exp (- x (trivial-simd:maximum x)))
     (trivial-simd:sum (exp (- x (trivial-simd:maximum x))))))
(trivial-simd:define-kernel silu (x) (/ x (+ 1 (exp (- x)))))
(trivial-simd:define-kernel rope-sine (angles) (sin angles))
(trivial-simd:define-kernel rope-cosine (angles) (cos angles))

(let ((logits (make-array 6 :element-type 'single-float
                            :initial-contents '(10f0 11f0 12f0 2f0 1f0 0f0)))
      (probabilities (make-array 6 :element-type 'single-float)))
  (softmax probabilities logits :rows 2 :row-length 3)
  (format t "~&Softmax rows: ~S~%" probabilities)
  (silu probabilities logits)
  (format t "SiLU: ~S~%" probabilities))

(let ((angles (make-array 3 :element-type 'double-float :initial-contents '(0d0 0.5d0 1d0)))
      (sines (make-array 3 :element-type 'double-float))
      (cosines (make-array 3 :element-type 'double-float)))
  (rope-sine sines angles)
  (rope-cosine cosines angles)
  (format t "RoPE tables: ~S ~S~%" sines cosines))
