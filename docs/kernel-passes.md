# Nested kernel reductions

A nested numeric reduction runs before the expression that consumes it. Its
scalar result broadcasts over the complete logical slice. Kernels share identical
reductions and execute their dependencies first.

```lisp
(trivial-simd:define-kernel normalize (x)
  (/ x (trivial-simd:nrm2 x)))

(normalize output input :start 4 :end 20)
```

The norm covers `input[4..20)`, and the output uses that one norm for all 16
values. Per-input starts change the loaded elements without changing the logical
slice coordinates. [Declared inputs](kernel-inputs.md) retain their conversion
and repetition rules in every pass.

## Results and types

All eight [numeric reducers](kernels.md#reduction-kernels) support nesting:
`sum`, `prod`, `asum`, `nrm2`, `minimum`, `maximum`, `argmin`, and `argmax`. Their existing
type restrictions, precision, wrapping arithmetic, ties, and norm scaling apply.

| Result | Consumption by another expression |
|---|---|
| Numeric value | Broadcast in the computation type |
| `argmin` / `argmax` index | Logical `:start` plus slice-relative index, converted to the computation type |
| Complex `asum` / `nrm2` | Real value promoted to complex with zero imaginary part |

Integer index conversion wraps to the integer computation width. Nested
reductions of scalar expressions still reduce one broadcast value per element.
A top-level reducer returns its ordinary scalar result; other numeric expressions
write a destination vector.

Empty slices skip inner evaluation. An elementwise call writes nothing, and a
top-level reducer retains its ordinary empty result, including `NIL` for extrema
and index reducers. This avoids evaluating arithmetic on empty inner extrema.

## Stable softmax

```lisp
(trivial-simd:define-kernel softmax (x)
  (/ (exp (- x (trivial-simd:maximum x)))
     (trivial-simd:sum (exp (- x (trivial-simd:maximum x))))))
```

| Pass | Work |
|---|---|
| 1 | Maximum of the input slice |
| 2 | Sum of exponentials after subtracting that maximum |
| 3 | Write shifted exponentials divided by the sum |

The two identical maximum expressions share one reduction. Exponentials are
recomputed in the last pass; no full-length exponential vector is required.
See [transcendental math](kernel-transcendentals.md), the
[runnable examples](../examples/inference-stages.lisp), and
[complete-stage measurements](kernel-stage-performance.md).

## Execution and errors

Each pass uses the selected backend and its existing fallbacks. Native bytecode
is reusable; scalar constant buffers belong to each call. Inner reductions
complete before destination writes. Views reduce the whole slice rather than
individual staging blocks. Native copy access retains its existing input-copy
costs.[^copy]

Types, bounds, and prohibited overlap are checked before writes. Slice calls
follow the [kernel overlap rule](kernels.md#overlap). An arithmetic error in an output pass may
leave partial output. Redefined kernels and retained older functions have
separate pass closures and program lifetimes.
Complex equality and selection compose with dependent `sum`, `prod`, `asum`, and `nrm2`
passes. Compatible complex passes use native execution when its symbols are
available; otherwise they use the [complex scalar helpers](complex.md#kernels).

[Row batches](kernel-rows.md) evaluate the graph independently for each row.
Zero rows write nothing; zero-length rows skip inner evaluation.

## Limitations

- Nested mask reducers (`count`, `any`, `all`) are unsupported.
- Structural deduplication scans earlier reductions. Native row execution sets
  up each row/pass separately, and vector expressions are recomputed. Further
  planning and execution work is tracked in
  [#58](https://github.com/communal-software/trivial-simd/issues/58).
- Reduction order and exceptional floating-point behavior retain the
  [kernel numerical limitations](kernels.md#limitations).

[^copy]: Native copy access may allocate a full accessed input span for a pass.
    Bounded VM scratch does not imply that all access modes use bounded total
    temporary storage.
