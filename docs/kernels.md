# Kernels

`define-kernel` compiles an elementwise expression into one loop per backend.
It is experimental.

```lisp
(trivial-simd:define-kernel fused-multiply-add (a b c) (+ (* a b) c))

(fused-multiply-add out x y z)                        ; out[i] = x[i]*y[i] + z[i]
(fused-multiply-add out x y z :start 0 :end 64)       ; same slice keywords as bulk operations
(fused-multiply-add out x y z :a-start 8 :end 64)     ; per-argument starts: <argument>-start
```

The kernel takes `destination`, one vector per argument, and the
[slice keywords](api.md#slices), plus one `<argument>-start` keyword per argument.
It returns `destination`.

## Expressions

| Form | Meaning |
|---|---|
| argument name | Element of that vector |
| real number | Constant[^constants] |
| `(+ x y ...)`, `(* x y ...)` | N-ary, folded left |
| `(- x y ...)`, `(/ x y ...)` | N-ary, folded left |
| `(- x)`, `(/ x)` | Negation, reciprocal |

Anything else signals an error when the kernel is defined. Vectors share one
`single-float` or `double-float` element type, chosen per call.

## Backends

| Backend | Implementation |
|---|---|
| `:lisp` | Typed scalar loop |
| `:sbcl` | `sb-simd` pack loop with a scalar tail |
| `:native` | Register bytecode run by a C interpreter over 256-element blocks[^vm] |

The native backend makes one foreign call per kernel call and needs no
intermediate arrays.

## Performance

Microseconds per call for `(+ (* a b) c)` on `single-float` vectors (Apple M1;
`two ops` is `multiply!` then `add!`).

| Elements | Lisp kernel | Lisp two ops | Native kernel (SBCL) | Native two ops (SBCL) | Native kernel (CCL) | Native kernel (ECL) |
|---|---|---|---|---|---|---|
| 32 | 0.12 | 1.3 | 0.13 | 0.19 | 0.44 | 4.9 |
| 1,024 | 2.6 | 38 | 0.35 | 0.40 | 0.68 | 7.8 |
| 65,536 | 165 | 2,470 | 16 | 16 | 17 | 23 |

- The native kernel matches two native bulk calls for a single fused
  expression; the gain is avoiding intermediate arrays for longer expressions.
- The Lisp kernel beats separate Lisp bulk calls by 10x or more.

Slice keywords add roughly 35 ns per call.

## Limitations

- Experimental; the operator set is `+ - * /`.[^ops] See the
  [operators ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/36).
- An expression may need at most 8 registers, so it may nest about 8 levels of
  balanced operations. See the
  [register spilling ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/35).
- No reductions inside a kernel. See the
  [reductions ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/37).
- ECL native kernel calls cost more than two bulk calls at small sizes. See the
  [ECL overhead ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/38).
- Kernel program memory is not freed. See the
  [memory ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/34).
- Element types are `single-float` and `double-float`. See the
  [integer ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/39).
- SBCL kernels are generated only on x86-64 with `sb-simd`.

[^constants]: Constants must be representable as `single-float`.
[^vm]: Instructions are four bytes: opcode, destination, and two operands. An
    operand names a register (0-7) or an input vector; the last instruction
    writes to `destination` directly.
[^ops]: `+` and `*` with one operand return it unchanged.
