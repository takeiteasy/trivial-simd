(defpackage #:trivial-simd/extension
  (:use)
  (:import-from #:trivial-simd
                #:vector-type #:absolute-argmax #:*native-array-access*
                #:with-pinned-pointers #:element-pointer #:define-native)
  (:export #:vector-type #:absolute-argmax #:*native-array-access*
           #:with-pinned-pointers #:element-pointer #:define-native))
