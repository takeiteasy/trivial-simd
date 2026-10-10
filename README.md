# trivial-simd

> **Work in progress.** This project is under development; expect missing features and breaking changes.

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

## Installation

From the takeiteasy Quicklisp dist, which is served over HTTPS so Quicklisp needs [ql-https](https://github.com/takeiteasy/ql-dist#install):

```lisp
(ql-dist:install-dist "https://takeiteasy.github.io/ql-dist/dist/takeiteasy.txt")
(ql:quickload :trivial-simd)
```

Or clone into Quicklisp's local-projects:

```sh
git clone https://github.com/takeiteasy/trivial-simd ~/quicklisp/local-projects/trivial-simd
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
- [N-D strided operations](docs/nd.md)
- [Small arrays](docs/small-arrays.md)
- [Copy, fill and swap](docs/copy.md)
- [Type conversion](docs/conversion.md)
- [Masks and selection](docs/masks.md)
- [Integer vectors](docs/integers.md)
- [Complex vectors](docs/complex.md)
- [Reductions](docs/reductions.md)
- [Vector views of foreign memory](docs/vector-views.md)
- [BLAS subsystem](docs/blas.md)
- [Matrix storage design](docs/matrix-views.md)
- [BLAS Level 2](docs/blas-level2.md)
- [BLAS Level 3](docs/blas-level3.md)
- [Batched GEMM](docs/blas-batches.md)
- [BLAS benchmarks](docs/blas-benchmarks.md)
- [Kernels](docs/kernels.md), including nested reductions and transcendental math
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
| Windows x86-64 | X | [P](https://github.com/communal-software/trivial-simd/issues/27) | [P](https://github.com/communal-software/trivial-simd/issues/28) |
| Windows ARM64 | X | | |
| [FreeBSD](https://github.com/communal-software/trivial-simd/issues/14) | P | P | P |
| [OpenBSD](https://github.com/communal-software/trivial-simd/issues/15) | P | | P |
| [NetBSD](https://github.com/communal-software/trivial-simd/issues/16) | P | | P |
| [Android](https://github.com/communal-software/trivial-simd/issues/17) | | | P |
| [iOS](https://github.com/communal-software/trivial-simd/issues/18) | | | P |

Planned implementations:
[ABCL](https://github.com/communal-software/trivial-simd/issues/19), [CLISP](https://github.com/communal-software/trivial-simd/issues/20), [Clasp](https://github.com/communal-software/trivial-simd/issues/21), [CMUCL](https://github.com/communal-software/trivial-simd/issues/22), [MKCL](https://github.com/communal-software/trivial-simd/issues/23), [LispWorks](https://github.com/communal-software/trivial-simd/issues/24), [Allegro CL](https://github.com/communal-software/trivial-simd/issues/25), [JSCL](https://github.com/communal-software/trivial-simd/issues/26) (Node and browsers), [GCL](https://github.com/communal-software/trivial-simd/issues/29).

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
| [x86-64 AVX2](https://github.com/communal-software/trivial-simd/issues/6) | | P | |
| [x86-64 AVX-512](https://github.com/communal-software/trivial-simd/issues/7) | | P | |
| [ARM64 SVE/SVE2](https://github.com/communal-software/trivial-simd/issues/8) | | P | |
| [ARM64 SBCL SIMD](https://github.com/communal-software/trivial-simd/issues/30) | P | | |
| [x86-32 SSE2](https://github.com/communal-software/trivial-simd/issues/9) | | P | |
| [ARMv7 NEON](https://github.com/communal-software/trivial-simd/issues/10) | | P | |
| [RISC-V Vector](https://github.com/communal-software/trivial-simd/issues/11) | | P | |
| [PowerPC VSX](https://github.com/communal-software/trivial-simd/issues/12) | | P | |
| [WebAssembly SIMD128](https://github.com/communal-software/trivial-simd/issues/13) | | P | |

A scalar fallback compiling on an unlisted target does not establish tested
SIMD support. See [backend coverage](docs/backends.md#limitations).

## License

```text
trivial-simd

Copyright (C) 2026 George Watson

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
```
