# Backends

The library selects the first available backend in this order:

| Backend | Availability | Array access |
|---|---|---|
| `:sbcl` | SBCL on x86-64 with `sb-simd` | Direct specialized-array loads and stores |
| `:native` | Built C library on macOS, Linux, or Windows | Direct pointers on SBCL, ECL, and CCL[^pointer] |
| `:lisp` | Any supported Common Lisp implementation | Direct Lisp array access |

The C backend uses SSE2 on x86-64 and NEON on ARM64. Other architectures use
scalar C loops. SBCL on ARM64 uses the C backend when built because its
`sb-simd` module does not provide an ARM64 instruction set in the tested SBCL.

## Tested combinations

| Platform | Lisp implementation | Required backend in CI |
|---|---|---|
| Linux | SBCL | `:sbcl`, `:lisp` |
| Linux | CCL, ECL | `:native` |
| macOS | SBCL | `:native`, `:lisp` |
| macOS | ECL, CCL on Intel | `:native` |
| Windows | SBCL | `:native`, `:lisp` |

The [CI matrix](../.github/workflows/ci.yml) defines the runner versions and
exercises the full test suite for each row.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The shared library stays in `build/`. Start a fresh Lisp process after
building it, then load `trivial-simd`. If no shared library is present, the
system loads with its Lisp backend. Set `TRIVIAL_SIMD_BACKEND` to `auto`
(default), `lisp`, `native`, or `sbcl` before starting Lisp to select a backend;
requesting an unavailable backend signals an error.

## Limitations

Only the implementation and platform combinations exercised by
[CI](../.github/workflows/ci.yml) are supported. The planned targets below are
**not supported yet**; a scalar fallback compiling elsewhere does not establish
tested SIMD support.

### Planned architectures and instruction sets

| Target | Planned work |
|---|---|
| x86-64 AVX2 | [Add AVX2 backend](https://todo.sr.ht/~takeiteasy/trivial-simd/4) |
| x86-64 AVX-512 | [Add AVX-512 backend](https://todo.sr.ht/~takeiteasy/trivial-simd/5) |
| ARM64 SVE/SVE2 | [Add scalable-vector backend](https://todo.sr.ht/~takeiteasy/trivial-simd/6) |
| x86-32 SSE2 | [Validate 32-bit x86 backend](https://todo.sr.ht/~takeiteasy/trivial-simd/7) |
| ARMv7 NEON | [Add 32-bit ARM backend](https://todo.sr.ht/~takeiteasy/trivial-simd/8) |
| RISC-V Vector | [Add RVV backend](https://todo.sr.ht/~takeiteasy/trivial-simd/9) |
| PowerPC VSX | [Add VSX backend](https://todo.sr.ht/~takeiteasy/trivial-simd/10) |
| WebAssembly SIMD128 | [Add wasm SIMD backend](https://todo.sr.ht/~takeiteasy/trivial-simd/11) |
| SBCL ARM64 SIMD | [Use an in-process SBCL backend](https://todo.sr.ht/~takeiteasy/trivial-simd/31) |

The current native C backend uses SSE2 on x86-64 and NEON on ARM64. The
additional instruction sets above are not selected by it.

### Planned platforms

| Target | Planned work |
|---|---|
| ARM64 Linux | [Validate native backend and CI](https://todo.sr.ht/~takeiteasy/trivial-simd/12) |
| ARM64 Windows | [Validate native backend and CI](https://todo.sr.ht/~takeiteasy/trivial-simd/13) |
| FreeBSD | [Build and test](https://todo.sr.ht/~takeiteasy/trivial-simd/14) |
| OpenBSD | [Build and test](https://todo.sr.ht/~takeiteasy/trivial-simd/15) |
| NetBSD | [Build and test](https://todo.sr.ht/~takeiteasy/trivial-simd/16) |
| Android | [Build and test](https://todo.sr.ht/~takeiteasy/trivial-simd/17) |
| iOS | [Build and test](https://todo.sr.ht/~takeiteasy/trivial-simd/18) |
| CCL on ARM64 macOS | [Add CI coverage](https://todo.sr.ht/~takeiteasy/trivial-simd/19) |

### Planned Lisp implementations

| Target | Planned work |
|---|---|
| ABCL | [Add fallback and tests](https://todo.sr.ht/~takeiteasy/trivial-simd/20) |
| CLISP | [Add fallback and tests](https://todo.sr.ht/~takeiteasy/trivial-simd/21) |
| Clasp | [Add fallback and tests](https://todo.sr.ht/~takeiteasy/trivial-simd/22) |
| CMUCL | [Add fallback and tests](https://todo.sr.ht/~takeiteasy/trivial-simd/23) |
| MKCL | [Add fallback and tests](https://todo.sr.ht/~takeiteasy/trivial-simd/24) |
| LispWorks | [Add fallback and tests](https://todo.sr.ht/~takeiteasy/trivial-simd/25) |
| Allegro CL | [Add fallback and tests](https://todo.sr.ht/~takeiteasy/trivial-simd/26) |
| JSCL in Node and browsers | [Add a JavaScript-compatible backend](https://todo.sr.ht/~takeiteasy/trivial-simd/27) |
| CCL on Windows | [Add CI coverage](https://todo.sr.ht/~takeiteasy/trivial-simd/28) |
| ECL on Windows | [Add CI coverage](https://todo.sr.ht/~takeiteasy/trivial-simd/29) |
| GCL | [Add fallback and tests](https://todo.sr.ht/~takeiteasy/trivial-simd/30) |

### Array access

Direct float-vector access relies on the tested SBCL, ECL, and CCL CFFI
implementations; CFFI documents shareable byte vectors rather than a general
float-vector guarantee.[^pointer] The native backend copies arrays on other
Lisp implementations. The
[array-access ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/2) tracks
tested direct access on more implementations. Small arrays may still run faster
in Lisp; use the [benchmark](testing.md) for a workload.

[^pointer]: CFFI's pointer macro maps to implementation-specific pinned or
    foreign views of specialized arrays on these three implementations.
