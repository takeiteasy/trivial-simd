# Activation measurements

Typed Lisp loops and native kernels each suit different call sizes. Scalar system
math does not guarantee that native execution is faster. These measurements cover
complete unary and composed activation calls, including kernel setup.

Run the benchmark:

```sh
sbcl --non-interactive --load tests/activation-bench.lisp
```

The benchmark checks results against typed loops, warms each path and reports
microseconds per call for log, tanh, sigmoid, SiLU and tanh-approximate GELU.
Inputs range from 0.001 to 4.001; both float precisions are measured.[^method]

## Representative results

ARM64 macOS 15.7.5, SBCL 2.6.8. Times are microseconds per call; lower is faster.

| Dtype | Elements | Operation | Typed Lisp | Lisp kernel | Native kernel |
|---|---:|---|---:|---:|---:|
| :f32 | 32 | log | 0.184 | 0.248 | 0.245 |
| :f32 | 32 | tanh | 0.171 | 0.457 | 0.232 |
| :f32 | 32 | sigmoid | 0.146 | 0.273 | 0.227 |
| :f32 | 32 | silu | 0.156 | 0.284 | 0.476 |
| :f32 | 32 | gelu | 0.220 | 0.371 | 0.355 |
| :f32 | 65,536 | log | 365.667 | 298.367 | 151.233 |
| :f32 | 65,536 | tanh | 341.500 | 404.200 | 148.533 |
| :f32 | 65,536 | sigmoid | 298.800 | 373.000 | 122.833 |
| :f32 | 65,536 | silu | 317.400 | 395.667 | 129.933 |
| :f32 | 65,536 | gelu | 469.067 | 567.333 | 312.500 |
| :f64 | 32 | log | 0.281 | 0.237 | 0.260 |
| :f64 | 32 | tanh | 0.159 | 0.284 | 0.301 |
| :f64 | 32 | sigmoid | 0.140 | 0.269 | 0.281 |
| :f64 | 32 | silu | 0.149 | 0.279 | 0.273 |
| :f64 | 32 | gelu | 0.209 | 0.348 | 0.420 |
| :f64 | 65,536 | log | 575.567 | 289.900 | 185.967 |
| :f64 | 65,536 | tanh | 328.333 | 386.600 | 273.600 |
| :f64 | 65,536 | sigmoid | 283.667 | 357.133 | 180.700 |
| :f64 | 65,536 | silu | 307.933 | 378.867 | 195.933 |
| :f64 | 65,536 | gelu | 445.767 | 547.100 | 522.533 |

The command also reports 1,024-element calls. Short calls include fixed native
setup cost. Larger calls amortize that setup, but the relative cost of each system
math routine still matters. These are local observations, not end-to-end model
performance claims.

[^method]: One warmed process, `max(20, floor(2000000/n))` calls per case, ordinary wall-clock timing. Validation jobs may run concurrently, so the values illustrate scale rather than establish a controlled performance ranking. Lisp paths include normal Lisp allocation/GC; native allocation is not reported. See the [numerical contract](kernel-transcendentals.md).
