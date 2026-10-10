(in-package #:trivial-simd)

(defvar *arm64-fma-auto-types* '(single-float double-float))

(defun arm64-fma-enabled-p (type)
  (and *arm64-fma-available-p*
       (or (eq *fma-mode* :in-process)
           (and (eq *fma-mode* :auto) (member type *arm64-fma-auto-types*)))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *arm64-fma-compiler-p*
    (or
     #+(and sbcl arm64)
     (and (member (lisp-implementation-version) '("2.6.8" "2.6.9") :test #'equal)
          (find-symbol "FMADD" :sb-arm64-asm))
     #+(and ccl arm64-target)
     (and (search "1.13" (lisp-implementation-version))
          (macro-function 'ccl::defarm64lapfunction))
     #+(and ecl aarch64)
     (equal (lisp-implementation-version) "26.5.5")
     nil)))

(declaim (inline arm64-fma-f32 arm64-fma-f64))
#+ecl
(declaim (inline %arm64-fma-f32 %arm64-fma-f64))

(defun %arm64-fma-f32 (a b c)
  (declare (type single-float a b c))
  #+(and ecl aarch64)
  (ffi:c-inline (a b c) (:float :float :float) :float
                "__builtin_fmaf(#0,#1,#2)" :one-liner t :side-effects nil)
  #-(and ecl aarch64)
  (portable-fma a b c))

(defun %arm64-fma-f64 (a b c)
  (declare (type double-float a b c))
  #+(and ecl aarch64)
  (ffi:c-inline (a b c) (:double :double :double) :double
                "__builtin_fma(#0,#1,#2)" :one-liner t :side-effects nil)
  #-(and ecl aarch64)
  (portable-fma a b c))

#+(and sbcl arm64)
(eval-when (:compile-toplevel :load-toplevel :execute)
  (when *arm64-fma-compiler-p*
    (sb-c:defknown %arm64-fma-f32 (single-float single-float single-float) single-float
        (sb-c:movable sb-c:foldable sb-c:flushable) :overwrite-fndb-silently t)
    (sb-c:defknown %arm64-fma-f64 (double-float double-float double-float) double-float
        (sb-c:movable sb-c:foldable sb-c:flushable) :overwrite-fndb-silently t)
    (sb-c:define-vop (%arm64-fma-f32)
      (:translate %arm64-fma-f32)
      (:policy :fast-safe)
      (:args (a :scs (sb-vm::single-reg)) (b :scs (sb-vm::single-reg))
             (c :scs (sb-vm::single-reg)))
      (:arg-types single-float single-float single-float)
      (:results (r :scs (sb-vm::single-reg)))
      (:result-types single-float)
      (:generator 2 (sb-assem:inst sb-arm64-asm::fmadd r a b c)))
    (sb-c:define-vop (%arm64-fma-f64)
      (:translate %arm64-fma-f64)
      (:policy :fast-safe)
      (:args (a :scs (sb-vm::double-reg)) (b :scs (sb-vm::double-reg))
             (c :scs (sb-vm::double-reg)))
      (:arg-types double-float double-float double-float)
      (:results (r :scs (sb-vm::double-reg)))
      (:result-types double-float)
      (:generator 2 (sb-assem:inst sb-arm64-asm::fmadd r a b c)))))

#+(and ccl arm64-target)
(eval-when (:compile-toplevel :load-toplevel :execute)
  (when *arm64-fma-compiler-p*
    (ccl::defarm64lapfunction %arm64-fma-f32 ((a ccl::arg_x) (b ccl::arg_y) (c ccl::arg_z))
      (ccl::get-single-float ccl::s0 a)
      (ccl::get-single-float ccl::s1 b)
      (ccl::get-single-float ccl::s2 c)
      (ccl::fmadd ccl::s0 ccl::s0 ccl::s1 ccl::s2)
      (ccl::put-single-float ccl::s0 ccl::arg_z)
      (ccl::ret))
    (ccl::defarm64lapfunction %arm64-fma-f64! ((a 0) (b ccl::arg_x) (c ccl::arg_y)
                                           (out ccl::arg_z))
      (ccl::ldr ccl::temp0 (:@ ccl::vsp (:$ 0)))
      (ccl::get-double-float ccl::d0 ccl::temp0)
      (ccl::get-double-float ccl::d1 b)
      (ccl::get-double-float ccl::d2 c)
      (ccl::fmadd ccl::d0 ccl::d0 ccl::d1 ccl::d2)
      (ccl::put-double-float ccl::d0 out)
      (ccl::add ccl::vsp ccl::vsp (:$ 8))
      (ccl::ret))))

(defun arm64-fma-f32 (a b c)
  (declare (type single-float a b c) (optimize (speed 3)))
  (%arm64-fma-f32 a b c))

(defun arm64-fma-f64 (a b c)
  (declare (type double-float a b c) (optimize (speed 3)))
  #+(and ccl arm64-target)
  (if *arm64-fma-compiler-p*
      ;; TODO: #35 unboxed kernel operands/results remove per-element allocation.
      (%arm64-fma-f64! a b c (ccl::%alloc-misc 2 arm64::subtag-double-float))
      (portable-fma a b c))
  #-(and ccl arm64-target)
  (%arm64-fma-f64 a b c))

(setf *arm64-fma-available-p*
      (and *arm64-fma-compiler-p*
           (ignore-errors
             (and (= 2.625 (arm64-fma-f32 1.25 2.5 -0.5))
                  (= 2.625d0 (arm64-fma-f64 1.25d0 2.5d0 -0.5d0))))))
