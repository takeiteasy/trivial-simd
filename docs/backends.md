# Backends

The library selects the first available backend in this order:

| Backend | Availability | Array access |
|---|---|---|
| `:sbcl` | SBCL on x86-64 with `sb-simd` | Direct specialized-array loads and stores |
| `:native` | Built C library on macOS, Linux, or Windows | Direct pointers on SBCL, ECL, and CCL[^pointer]; copies elsewhere |
| `:lisp` | Any supported Common Lisp implementation | Direct Lisp array access |

The C backend uses SSE2 on x86-64 and NEON on ARM64. Other architectures use
scalar C loops. This library enables its SBCL backend only on x86-64.
SBCL on ARM64 selects the C backend when built, or the Lisp fallback otherwise.

## Tested combinations

| Platform | Lisp implementation | Required backend in CI |
|---|---|---|
| Linux x86-64 | SBCL | `:sbcl`, `:lisp` |
| Linux x86-64 | CCL, ECL | `:native` |
| Linux ARM64 | SBCL, ECL, CCL[^ccl-arm] | `:native` |
| macOS ARM64 | SBCL, ECL, CCL[^ccl-arm] | `:native`; SBCL also uses `:lisp` |
| macOS Intel | CCL | `:native` |
| Windows x86-64 | SBCL | `:native`, `:lisp` |
| Windows ARM64 | SBCL | `:native` |

The [CI matrix](../.github/workflows/ci.yml) defines the runner versions and
exercises the full test suite for each row.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

MSVC builds use `/fp:strict` to preserve signed zeros and rounding.

The shared library stays in `build/`. Start a fresh Lisp process after
building it, then load `trivial-simd`. If no shared library is present, the
system loads with its Lisp backend. Set `TRIVIAL_SIMD_BACKEND` to `auto`
(default), `lisp`, `native`, or `sbcl` before starting Lisp to select a backend;
requesting an unavailable backend signals an error.

Scalar FMA uses guarded SBCL x86 hardware instructions and
[ARM64 compiler adapters](arm64-fma.md). These paths work without the native
library. Other configurations use optional correctly rounded native helpers,
then exact portable arithmetic when the helpers are absent.

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
| FreeBSD | [Build and test](https://todo.sr.ht/~takeiteasy/trivial-simd/14) |
| OpenBSD | [Build and test](https://todo.sr.ht/~takeiteasy/trivial-simd/15) |
| NetBSD | [Build and test](https://todo.sr.ht/~takeiteasy/trivial-simd/16) |
| Android | [Build and test](https://todo.sr.ht/~takeiteasy/trivial-simd/17) |
| iOS | [Build and test](https://todo.sr.ht/~takeiteasy/trivial-simd/18) |

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
Lisp implementations, using typed per-element loops.[^copy] Support
for more implementations is tracked by their own tickets above. Small arrays may
still run faster in Lisp; use the [benchmark](testing.md) for a workload.
Copy mode allocates buffers for the requested slice and copies only its elements.
Each input has separate storage, including repeated and aliased inputs.

[^pointer]: CFFI's pointer macro maps to implementation-specific pinned or
    foreign views of specialized arrays on these three implementations.
[^copy]: The suite also runs the native backend in copy mode on every tested
    implementation. Copy mode dispatches once by float type and uses literal CFFI types
    in specialized loops. Foreign index zero corresponds to each vector’s requested
    start; output copy-back touches only the destination slice (`*native-array-access*` is `:copy`).
[^ccl-arm]: ARM64 CI uses Clozure CL `v1.13-arm64-pre2` on Linux and macOS.
