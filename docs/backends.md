# Backends

The library selects the first available backend in this order:

| Backend | Availability | Array access |
|---|---|---|
| `:sbcl` | SBCL on x86-64 with `sb-simd` | Direct specialized-array loads and stores |
| `:native` | Built C library on macOS, Linux, or Windows | Copies through foreign buffers |
| `:lisp` | Any supported Common Lisp implementation | Direct Lisp array access |

The C backend uses SSE2 on x86-64 and NEON on ARM64. Other architectures use
scalar C loops. SBCL on ARM64 uses the C backend when built because its
`sb-simd` module does not provide an ARM64 instruction set in the tested SBCL.

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

The native backend copies Lisp arrays into foreign buffers on each call, so
small arrays may run faster in Lisp. The
[zero-copy ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/2) tracks safe
per-implementation array access. Use the [benchmark](testing.md) to check
which backend suits a workload.
