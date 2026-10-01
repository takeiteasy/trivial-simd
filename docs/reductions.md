# Reductions

Scalar reductions over a slice of one vector: `sum`, `dot`, `dotc`, `asum`,
`nrm2`, `minimum`, `maximum`, `argmin` and `argmax`. All accept the
[bulk slice keywords](api.md#slices); the single-vector functions take
`:input-start`.

| Function | Result | Types |
|---|---|---|
| `(sum x &key accumulate)` | Sum | Float, complex, integer |
| `(dot x y &key accumulate)` | Dot product | Float, complex, integer |
| `(asum x &key accumulate)` | Sum of `abs`; complex uses the modulus | Float, complex, integer |
| `(nrm2 x &key accumulate)` | Euclidean norm | Float, complex |
| `(minimum x)`, `(maximum x)` | Smallest or largest element | Float, integer |
| `(argmin x)`, `(argmax x)` | Index of that element in `x` | Float, integer |

```lisp
(trivial-simd:argmax v :start 2 :end 10)   ; => 7, an index into V
(trivial-simd:nrm2 v)                      ; no overflow for 1d200 elements
(trivial-simd:sum singles :accumulate :f64) ; => double-float
```

## Empty slices

`minimum`, `maximum`, `argmin` and `argmax` return `NIL`. `sum`, `dot`,
`asum` and `nrm2` return a zero.

## Ties and signed zeros

The first extreme element wins: `argmin` of the single-floats `0.0, -0.0` is `0`,
and `minimum` of `-0.0, 0.0` is `-0.0`. Every backend returns the same index.
`asum` of integers [wraps](integers.md) at the vector width, including
`abs` of the most negative value. Complex vectors have no ordering, and
`nrm2` has no integer form; both signal an error. NaN inputs follow the
[IEEE consistency limitation](kernels.md#limitations).

## Precision

`:accumulate :f64` widens partial sums of `single-float` and
`(complex single-float)` vectors to double precision and returns doubles.
It has no effect on double-float inputs and signals an error for integers
or any value other than `:f64`. `sum`, `dot`, `dotc` and `asum` use the
widened path directly.

`nrm2` of a `single-float` vector always accumulates squares in double
precision, then rounds to single unless `:accumulate :f64` is given. For
`double-float` it takes one pass of squares, and falls back to a scaled pass
when the squares overflow or may have underflowed. Complex vectors use the
same rules on their parts.[^nrm2]

## Accumulation order

| Backend | Order[^order] |
|---|---|
| `:lisp` | Element order |
| `:sbcl` | Element order for new reductions;[^placeholder] `sum` and `dot` add SIMD lanes, then the tail |
| `:native` | Lane partial sums, lanes added in order, then the tail |

Widened `single-float` sums on `:native` use four double-precision lanes on
every CPU.

## Limitations

| Area | Ticket |
|---|---|
| Reducers inside `define-kernel` | [#90](https://todo.sr.ht/~takeiteasy/trivial-simd/90) |
| SIMD `argmin`, `argmax` and `asum` on SBCL and for integers | [#92](https://todo.sr.ht/~takeiteasy/trivial-simd/92) |
| The Lisp backend's `sum` and `dot` use untyped loops | [#93](https://todo.sr.ht/~takeiteasy/trivial-simd/93) |

[^nrm2]: The `double-float` fast path accepts sums from `1d-280` to the largest
    double; the scaled pass divides by the largest absolute part. A zero sum
    also takes the scaled pass, which returns zero for an all-zero vector.
[^order]: Floating-point results can differ between backends in the last bits.
    `argmin`, `argmax`, `minimum` and `maximum` do not.
[^placeholder]: SBCL and all integer vectors use typed scalar loops for
    `asum`, `argmin` and `argmax`. Native float `argmin` and `argmax` scan
    32-element blocks and skip blocks that cannot improve the result.
