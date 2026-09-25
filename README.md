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

## Backends

- `:sbcl` — SBCL on x86-64 when `sb-simd` is available.
- `:native` — the C library on macOS, Linux, or Windows: SSE2 on x86-64,
  NEON on ARM64, and scalar C elsewhere.
- `:lisp` — pure Common Lisp fallback.

Build the native library with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
and `cmake --build build --config Release`. The Lisp fallback loads without it.

- [API](docs/api.md)
- [Kernels](docs/kernels.md)
- [Backends and build](docs/backends.md)
- [Testing and benchmarks](docs/testing.md)
- [CI](docs/ci.md)

## Limitations

The planned backends below are **not supported yet**. A backend is supported
when its implementation and platform combination passes the test suite in CI.
See [backend coverage](docs/backends.md#limitations) for details.

| Area | Planned targets |
|---|---|
| Architecture | [AVX2](https://todo.sr.ht/~takeiteasy/trivial-simd/4), [AVX-512](https://todo.sr.ht/~takeiteasy/trivial-simd/5), [SVE/SVE2](https://todo.sr.ht/~takeiteasy/trivial-simd/6), [x86-32 SSE2](https://todo.sr.ht/~takeiteasy/trivial-simd/7), [ARMv7 NEON](https://todo.sr.ht/~takeiteasy/trivial-simd/8), [RISC-V Vector](https://todo.sr.ht/~takeiteasy/trivial-simd/9), [PowerPC VSX](https://todo.sr.ht/~takeiteasy/trivial-simd/10), [WebAssembly SIMD128](https://todo.sr.ht/~takeiteasy/trivial-simd/11), [SBCL ARM64 SIMD](https://todo.sr.ht/~takeiteasy/trivial-simd/31) |
| Platform | [FreeBSD](https://todo.sr.ht/~takeiteasy/trivial-simd/14), [OpenBSD](https://todo.sr.ht/~takeiteasy/trivial-simd/15), [NetBSD](https://todo.sr.ht/~takeiteasy/trivial-simd/16), [Android](https://todo.sr.ht/~takeiteasy/trivial-simd/17), [iOS](https://todo.sr.ht/~takeiteasy/trivial-simd/18) |
| Lisp implementation | [ABCL](https://todo.sr.ht/~takeiteasy/trivial-simd/20), [CLISP](https://todo.sr.ht/~takeiteasy/trivial-simd/21), [Clasp](https://todo.sr.ht/~takeiteasy/trivial-simd/22), [CMUCL](https://todo.sr.ht/~takeiteasy/trivial-simd/23), [MKCL](https://todo.sr.ht/~takeiteasy/trivial-simd/24), [LispWorks](https://todo.sr.ht/~takeiteasy/trivial-simd/25), [Allegro CL](https://todo.sr.ht/~takeiteasy/trivial-simd/26), [JSCL](https://todo.sr.ht/~takeiteasy/trivial-simd/27), [CCL Windows](https://todo.sr.ht/~takeiteasy/trivial-simd/28), [ECL Windows](https://todo.sr.ht/~takeiteasy/trivial-simd/29), [GCL](https://todo.sr.ht/~takeiteasy/trivial-simd/30) |

## License

```text
MIT License

Copyright (c) 2026 George Watson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
