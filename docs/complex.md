# Complex vectors

The core API accepts simple vectors of `(complex single-float)` and
`(complex double-float)` for arithmetic, `scale!`, `axpy!`, `sum`, and `dot`.
`dot` is unconjugated; `dotc` conjugates its first vector. `negate!`,
`reciprocal!`, and `sqrt!` also accept complex vectors. Scalar operands match
the vector's complex precision.

| Operation | Complex result |
|---|---|
| `abs!` | Magnitudes in a real vector of matching precision |
| `compare!` | Equality or inequality mask |
| `select!` | Complex destination |
| `convert!` | Real or integer to complex, complex precision changes, and zero-imaginary complex to real or integer |

Converting a complex value to real or integer requires a zero imaginary part;
otherwise it signals an error. Complex magnitudes use Euclidean absolute
value. BLAS absolute sums use a different definition, described in
[Level 1 BLAS](blas-level1.md).

```lisp
(let ((x (make-array 2 :element-type '(complex single-float)
                       :initial-contents '(#C(1.0f0 2.0f0) #C(3.0f0 4.0f0))))
      (magnitudes (make-array 2 :element-type 'single-float)))
  (trivial-simd:abs! magnitudes x)
  magnitudes) ; => #(2.236068 5.0)
```

The native backend copies complex vectors to interleaved real/imaginary
buffers for SIMD add, subtract, and multiply on supported CPUs. Other complex
operations use Lisp arithmetic. The core slice rules apply to all complex
operations.

## Kernels

`define-kernel` accepts complex arithmetic, square roots, equality masks,
selection, and reductions. Inputs share one complex precision; numeric outputs
match it, and comparisons write byte masks.

| Expression | Result |
|---|---|
| `(= a b)`, `(/= a b)` | Both components determine equality; output mask bytes are `0` or `1` |
| `(trivial-simd:select (= a b) a 0)` | Complex vector; real constants have zero imaginary part |
| `(trivial-simd:count (= a b))` | Integer count |
| `(trivial-simd:any (= a b))`, `(trivial-simd:all (= a b))` | Boolean |
| `sum` over a numeric expression | Complex scalar of input precision |
| `asum`, `nrm2` over a numeric expression | Real scalar of input precision |

Selections nest inside arithmetic, comparisons, and numeric reductions.
[Dependent numeric reductions](kernel-passes.md) broadcast complex sums and
promote real magnitudes/norms to complex values with zero imaginary part.
Mask reductions stay top-level. Empty masks return zero for `count`, false for
`any`, and true for `all`; empty numeric reductions return typed zero.

On `:native`, NEON/SSE2 execute equality, selection, addition, subtraction,
and multiplication in packed lanes. Division and square root use scalar lanes.
Native numeric reductions evaluate 256-element blocks and accumulate their
results; norms scale components to avoid overflow and underflow.
The Lisp fallback applies when the library lacks the complex kernel symbols.
On `:lisp` and `:sbcl`, complex expressions use scalar helpers compiled on first
use on SBCL/CCL and interpreted on ECL.[^helpers]

See the [runnable complex kernel example](../examples/complex-kernels.lisp).

## Limitations

Complex numbers have no ordering for `min!`, `max!`, `clamp!`, or ordered
comparisons. Complex kernels reject `min`, `max`, `abs`, FMA, and transcendental
expressions; `asum` computes complex magnitudes. Declared inputs and row batches
retain their real-float restrictions.
`fma!` retains its real floating-point contract.
The native complex path copies vectors into temporary buffers; reducing that
overhead is tracked by [#74](https://todo.sr.ht/~takeiteasy/trivial-simd/74).
Native complex reductions materialize a block before reducing it; packed fusion
is tracked by [#96](https://todo.sr.ht/~takeiteasy/trivial-simd/96).
CCL complex helpers call generic arithmetic to avoid an ARM64 compiler bug with
typed complex-double constants in CCL 1.13 (`v1.13-459-g690ff7ea`,
`DarwinARM6464`). A compiled sum drops its argument; interpreted evaluation
and a `NOTINLINE` declaration produce the expected result.[^ccl]
The cached kernel helpers retain the workaround. The compiler defect remains
tracked by [#138](https://todo.sr.ht/~takeiteasy/trivial-simd/138); its scope on
x86-64 and in other typed complex arithmetic paths is unverified.
Cold concurrent complex Lisp kernels have an intermittent ARM64 CCL SIGBUS
under investigation in [#142](https://todo.sr.ht/~takeiteasy/trivial-simd/142).
Exceptional floating-point behavior retains the
[kernel numerical limitations](kernels.md#limitations).

[^helpers]: Helpers and native programs are cached separately for each complex
    precision and kernel definition. The expression tree is stored once rather
    than expanding additional complex loops into the real/integer branches.
    Calls own their mutable constants, transfers, and scratch, including calls
    retained across redefinition.

[^ccl]: Minimal reproducer:

    ```lisp
    (let ((f (compile nil
                      '(lambda (a)
                         (declare (type (complex double-float) a))
                         (+ a #c(1d0 0d0))))))
      (funcall f #c(2d0 2d0)))
    ```

    Expected: `#C(3d0 2d0)`. Observed: `#C(1d0 0d0)`.
    Adding `(declare (notinline +))` inside the lambda restores the expected
    result. Cached complex-kernel helpers declare `+`, `-`, `*`, `/`, `sqrt`,
    `abs`, `=`, and `/=` not inline, including comparisons and selection.
