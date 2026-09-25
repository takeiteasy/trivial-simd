# trivial-simd

Bulk SIMD arithmetic for Common Lisp `single-float` and `double-float` vectors.
It uses SBCL SIMD where available, a small C library, or a pure Lisp fallback.

```lisp
(ql:quickload :trivial-simd)
(let ((a (make-array 4 :element-type 'single-float
                     :initial-contents '(1.0 2.0 3.0 4.0)))
      (b (make-array 4 :element-type 'single-float
                     :initial-contents '(2.0 2.0 2.0 2.0)))
      (out (make-array 4 :element-type 'single-float)))
  (trivial-simd:add! out a b)
  (trivial-simd:dot a b)) ; => 20.0
```

Build the native library with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
and `cmake --build build --config Release`. The Lisp fallback loads without it.

- [API](docs/api.md)
- [Backends and build](docs/backends.md)
- [Testing and benchmarks](docs/testing.md)
- [License](LICENSE)
