# trivial-simd

Bulk SIMD arithmetic and BLAS for Common Lisp float and fixed-width integer
vectors. It uses SBCL SIMD where available, a small C library, or a pure Lisp
fallback.

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
- `:native` — the C library on macOS, Linux, or Windows: SSE2 on x86-64 (plus
  AVX+FMA for real BLAS on CPUs that have it), NEON on ARM64, and scalar C
  elsewhere.
- `:lisp` — portable Lisp arithmetic with optional scalar FMA acceleration.

Build the native library with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
and `cmake --build build --config Release`. The Lisp fallback loads without it.

- [API](docs/api.md)
- [Elementwise operations](docs/elementwise.md)
- [Small arrays](docs/small-arrays.md)
- [Copy, fill and swap](docs/copy.md)
- [Type conversion](docs/conversion.md)
- [Masks and selection](docs/masks.md)
- [Integer vectors](docs/integers.md)
- [Reductions](docs/reductions.md)
- [Vector views of foreign memory](docs/vector-views.md)
- [BLAS subsystem](docs/blas.md)
- [Matrix storage design](docs/matrix-views.md)
- [BLAS Level 2](docs/blas-level2.md)
- [BLAS Level 3](docs/blas-level3.md)
- [BLAS benchmarks](docs/blas-benchmarks.md)
- [Kernels](docs/kernels.md)
- [Backends and build](docs/backends.md)
- [Testing and benchmarks](docs/testing.md)
- [CI](docs/ci.md)

## Benchmarks

Apple M1, SBCL 2.6.8, native NEON with pointer access, measured 2026-10-02:
speedups over typed scalar Lisp loops for 1,024-element arrays.

| Operation | Single-float | Double-float |
|---|---:|---:|
| Add | 4.50x | 3.05x |
| Multiply-add | 2.78x | 1.80x |
| Dot | 3.12x | 1.78x |

Native `sum(a*b)` kernels take 10.72 µs for 65,536 single-float elements,
versus 23.75 µs for separate multiply and sum calls. Short float vectors of up to
64 elements [run inline](docs/small-arrays.md) at scalar-loop speed; reusable
spill scratch does not meet the 10% improvement gate.
See [full results and measurement details](docs/benchmarks.md).

## Limitations

`X` is supported: the combination passes the test suite in
[CI](docs/backends.md#tested-combinations). `L` passes the suite in
[local runs](docs/backends.md#local-runs) but not in CI. `H` has historical CI coverage but no active CI job. `P` is planned and not
supported yet. Blank is not planned.

### Platforms and Lisp implementations

| Platform / architecture | SBCL | CCL | ECL |
|---|:---:|:---:|:---:|
| Linux x86-64 | X | H | X |
| Linux ARM64 | X | H | X |
| macOS ARM64[^macos-local] | L | L | L |
| macOS x86-64[^macos-local] | L | L | L |
| Windows x86-64 | X | [P](https://todo.sr.ht/~takeiteasy/trivial-simd/28) | [P](https://todo.sr.ht/~takeiteasy/trivial-simd/29) |
| Windows ARM64 | X | | |
| [FreeBSD](https://todo.sr.ht/~takeiteasy/trivial-simd/14) | P | P | P |
| [OpenBSD](https://todo.sr.ht/~takeiteasy/trivial-simd/15) | P | | P |
| [NetBSD](https://todo.sr.ht/~takeiteasy/trivial-simd/16) | P | | P |
| [Android](https://todo.sr.ht/~takeiteasy/trivial-simd/17) | | | P |
| [iOS](https://todo.sr.ht/~takeiteasy/trivial-simd/18) | | | P |

Planned implementations:
[ABCL](https://todo.sr.ht/~takeiteasy/trivial-simd/20), [CLISP](https://todo.sr.ht/~takeiteasy/trivial-simd/21), [Clasp](https://todo.sr.ht/~takeiteasy/trivial-simd/22), [CMUCL](https://todo.sr.ht/~takeiteasy/trivial-simd/23), [MKCL](https://todo.sr.ht/~takeiteasy/trivial-simd/24), [LispWorks](https://todo.sr.ht/~takeiteasy/trivial-simd/25), [Allegro CL](https://todo.sr.ht/~takeiteasy/trivial-simd/26), [JSCL](https://todo.sr.ht/~takeiteasy/trivial-simd/27) (Node and browsers), [GCL](https://todo.sr.ht/~takeiteasy/trivial-simd/30).

[^macos-local]: macOS testing is local only. ARM64 scripts use installed SBCL 2.6.8,
    ECL 26.5.5, and CCL `v1.13-459-g690ff7ea`; x86-64 scripts use cached
    SBCL 2.6.8, ECL 26.5.5, and CCL 1.13 under Rosetta. See
    [local platform runs](docs/testing.md#local-platform-runs).


### Instruction sets and backends

| Instruction set | `:sbcl` | `:native` | `:lisp` |
|---|:---:|:---:|:---:|
| x86-64 SSE2 | X | X | X |
| x86-64 AVX+FMA (real BLAS) | | X | |
| ARM64 NEON | | X | X |
| [x86-64 AVX2](https://todo.sr.ht/~takeiteasy/trivial-simd/4) | | P | |
| [x86-64 AVX-512](https://todo.sr.ht/~takeiteasy/trivial-simd/5) | | P | |
| [ARM64 SVE/SVE2](https://todo.sr.ht/~takeiteasy/trivial-simd/6) | | P | |
| [ARM64 SBCL SIMD](https://todo.sr.ht/~takeiteasy/trivial-simd/31) | P | | |
| [x86-32 SSE2](https://todo.sr.ht/~takeiteasy/trivial-simd/7) | | P | |
| [ARMv7 NEON](https://todo.sr.ht/~takeiteasy/trivial-simd/8) | | P | |
| [RISC-V Vector](https://todo.sr.ht/~takeiteasy/trivial-simd/9) | | P | |
| [PowerPC VSX](https://todo.sr.ht/~takeiteasy/trivial-simd/10) | | P | |
| [WebAssembly SIMD128](https://todo.sr.ht/~takeiteasy/trivial-simd/11) | | P | |

A scalar fallback compiling on an unlisted target does not establish tested
SIMD support. See [backend coverage](docs/backends.md#limitations).

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

