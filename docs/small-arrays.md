# Small arrays

Calls on short float vectors run an inline loop and skip the backend, so a
call costs about as much as a hand-written loop.

| Condition | Inline path |
|---|---|
| Element type | `single-float` or `double-float` simple vectors |
| Length | At most 64 elements, equal across vectors |
| Arguments | Positional only: no `:start`, `:end` or `:stride` keywords |
| Call site | A compiled call to the function by name |
| `*inline-small-arrays*` | True (the default) |

Anything else takes the normal path with its validation and errors, including
integer and complex vectors, mismatched lengths or types, and calls through
`funcall`, `apply` or `#'add!`.

```lisp
(add! out a b)           ; inline when out, a, b are 4 single-floats
(add! out a b :start 0)  ; backend call
(funcall #'add! out a b) ; backend call
```

## Operations

`add!`, `subtract!`, `multiply!`, `divide!`, `scale!`, `axpy!`, `sum`, `dot`,
`negate!`, `abs!`, `reciprocal!`, `min!`, `max!` and `clamp!`. The
binary forms take two vectors. `scale!`, `axpy!` and `clamp!` take scalars of
the element type.

## Timing

Microseconds per call at 4 and 32 elements, single-float:[^timing]

| Elements | Operation | Scalar loop | Inline | `:lisp` | `:native` |
|---:|---|---:|---:|---:|---:|
| 4 | add | 0.011 | 0.010 | 0.056 | 0.083 |
| 4 | dot | 0.009 | 0.009 | 0.035 | 0.063 |
| 4 | min | - | 0.010 | 0.130 | 0.193 |
| 32 | add | 0.029 | 0.025 | 0.119 | 0.088 |
| 32 | dot | 0.040 | 0.019 | 0.073 | 0.070 |
| 32 | min | - | 0.025 | 0.176 | 0.192 |

## Behaviour

- Inline calls ignore the active backend and accumulate in element order, like
  the `:lisp` backend. Results can differ from `:native` in the last bits; see
  [accumulation order](reductions.md#accumulation-order).
- Float traps and NaNs follow Lisp's rules, as on the `:lisp` backend.
- Bind `*inline-small-arrays*` to false to send every call to the backend, for
  example to compare backends:

```lisp
(let ((trivial-simd:*inline-small-arrays* nil))
  (add! out a b))
```

- Interpreted code and compilers that do not apply compiler macros use the
  backend.[^ecl]

## Limitations

- Other operations, vector-scalar forms and per-operation length limits are
  not covered; see [#50](https://github.com/communal-software/trivial-simd/issues/50).

[^timing]: From `tests/overhead-bench.lisp` on SBCL 2.6.8 and macOS ARM64,
    2026-10-02. See [call overhead](benchmarks.md#call-overhead).
[^ecl]: ECL does not expand them for calls written directly inside FiveAM test
    bodies; compiled application code is expanded.
