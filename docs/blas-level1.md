# BLAS Level 1

`trivial-simd/blas` provides the standard Level 1 real and complex routines.
Names use the BLAS precision prefixes: `s` and `d` for real single and double;
`c` and `z` for complex single and double. All vectors are simple specialized
arrays. Counts are nonnegative, increments are nonzero, and offsets default to
zero. A negative increment visits a span from its far end to its near end.
Invalid types or spans signal an error before the routine starts.

| Family | Variants | Result |
|---|---|---|
| Swap, copy, AXPY | `s`, `d`, `c`, `z` | Modified vector; swap returns both vectors |
| Scale | `s`, `d`, `c`, `z`; `csscal`, `zdscal` for real factors | Modified vector |
| Dot | `sdot`, `ddot`, `cdotu`, `zdotu`, `cdotc`, `zdotc`, `sdsdot`, `dsdot` | Scalar |
| Euclidean norm | `snrm2`, `dnrm2`, `scnrm2`, `dznrm2` | Real scalar |
| Absolute sum | `sasum`, `dasum`, `scasum`, `dzasum` | Real scalar |
| Maximum absolute index | `isamax`, `idamax`, `icamax`, `izamax` | Zero-based logical index |
| Rotation | `srotg`, `drotg`, `crotg`, `zrotg`, `srotmg`, `drotmg`, `srot`, `drot`, `csrot`, `zdrot`, `crot`, `zrot`, `srotm`, `drotm` | Parameters or modified vectors |

Unit-stride real `asum`, `nrm2` and `i?amax` call the core [reductions](reductions.md)
and their native paths. Complex absolute sums and index searches use `abs(realpart) + abs(imagpart)`;
norms use Euclidean magnitude. Index ties select the first visited element.
The empty index result is zero. `dotu` does not conjugate; `dotc` conjugates
its first vector. `sdsdot` accumulates in double precision, adds a
single-float scalar, then returns a single float. `dsdot` returns a double.

```lisp
(trivial-simd/blas:zdotc n x 1 y 1 :x-offset 2 :y-offset 5)
(trivial-simd/blas:csscal n 2.0f0 x -1)
```

Real `rotg` takes two same-precision scalars and returns `r`, `z`, `c`, `s` as
multiple values. `rotmg` takes `d1`, `d2`, `b1`, `b2` and returns updated
`d1`, `d2`, `b1`, and a five-element parameter vector. `rot` and `rotm`
modify both vectors and return them as two values. `rotm` takes the parameter
vector produced by `rotmg`.[^rotm]
Complex `rotg` returns `r`, the unchanged second input, real `c`, and complex
`s`. `csrot` and `zdrot` use real `c` and `s`; `crot` and `zrot` use real `c`
and complex `s`.

The optional convenience package supplies `axpy!`, `scal!`, `copy!`, `swap!`,
`dot`, and `norm2`. These use unit increments and derive the count from an
optional shared `:start`/`:end` slice. `dot` accepts `:conjugate t` for complex
vectors. Without `:end`, paired vectors have equal lengths.

## Limitations

Nonunit increments and complex reductions use scalar loops. Core strided
operations are tracked by [#43](https://todo.sr.ht/~takeiteasy/trivial-simd/43)
and native complex reductions by [#91](https://todo.sr.ht/~takeiteasy/trivial-simd/91). Shifted overlap
between input and output spans follows the [overlap limitation](api.md#limitations).

[^rotm]: The flag in the first parameter element selects which matrix entries
    are active. The other four elements follow the reference BLAS order
    `h11`, `h21`, `h12`, `h22`.
