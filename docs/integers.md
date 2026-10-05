# Integer vectors

`trivial-simd` accepts simple specialized vectors of signed or unsigned
8-, 16-, 32-, and 64-bit integers. Inputs and destination must have the same
element type.

| Operation | Integer behavior |
|---|---|
| `add!`, `subtract!`, `multiply!`, `scale!`, `axpy!` | Wrap to the vector's width |
| `divide!` | Quotient truncated toward zero; zero divisor signals `division-by-zero` |
| `sum`, `dot` | Same-width wrapped integer result; empty slice returns `0` |
| `negate!`, `abs!` | Wrap to the vector width |
| `min!`, `max!`, `clamp!` | Compare values without wrapping |
| `convert!` | Clamp to the destination integer range |
| `define-kernel` | Wrap each arithmetic step and the result of a sum kernel |

Signed results use the two's-complement range. For example, signed 8-bit
`127 + 1` is `-128`, and unsigned 8-bit `250 + 10` is `4`. Signed minimum
divided by `-1` wraps to signed minimum. Negation and `abs` wrap too, so
`abs(-128)` is `-128` for signed 8-bit vectors.

The four binary operations accept one in-range integer scalar in place of a
vector. `scale!` multiplies a vector in place; `axpy!` computes `y := a*x+y`
with wrapping multiplication and addition. `sqrt!`, `reciprocal!`, and `fma!`
remain float-only. See [elementwise operations](elementwise.md),
[conversion](conversion.md), and [masks](masks.md) for their bulk APIs.

```lisp
(let* ((a (make-array 3 :element-type '(unsigned-byte 8)
                        :initial-contents '(250 7 2)))
       (b (make-array 3 :element-type '(unsigned-byte 8)
                        :initial-contents '(10 3 4)))
       (out (make-array 3 :element-type '(unsigned-byte 8))))
  (trivial-simd:add! out a b)  ; => #(4 10 6)
  (trivial-simd:scale! out 2)  ; => #(8 20 12)
  (trivial-simd:dot a b))     ; => 225
```

Integer kernels accept `+`, `-`, `*`, `/`, `abs`, `min`, and `max`, plus a
top-level `trivial-simd:sum`. Integer constants must fit the vector type;
fractional, floating-point, and out-of-range constants signal a type error.
`sqrt` and `trivial-simd:fma` require floats.[^dispatch]

```lisp
(trivial-simd:define-kernel clipped-products (a b)
  (trivial-simd:sum (max 0 (* a b))))
```

The [runnable examples](../examples/integers.lisp) show wrapping, division, and a
sum kernel.

An arithmetic error may leave part of an output vector updated. Slice and
in-place rules match the [bulk API](api.md#slices).

## Backend coverage

| Backend | Integer execution |
|---|---|
| `:lisp` | Typed scalar arithmetic |
| `:sbcl` on x86-64 | SSE2 add/subtract and packed kernels where the instruction is available; typed scalar paths for other operations |
| `:native` on x86-64 | SSE2 add/subtract and 16-bit multiply; typed scalar paths for other operations |
| `:native` on ARM64 | NEON add/subtract and 8/16/32-bit multiply; typed scalar paths for other operations |

The native backend uses scalar integer operations on other architectures.
Copy mode supports all eight types. Integer division runs lane by lane on all
backends. These fallback paths preserve the same wrapping results.[^reduction]
Native 64-bit kernels evaluate `(+ (* a b) constant)` in one pass when the
destination is disjoint from the inputs or uses the same range.

## Performance

Apple M1, SBCL 2.6.8, warmed calls on 2026-09-29. Times are microseconds
per call, measured with three calibrated batches; values are the median of three batches. The kernel computes `(+ (* a b) 3)`.[^benchmark]

| Type | Elements | Lisp add | Native add | Lisp dot | Native dot | Lisp kernel | Native kernel |
|---|---:|---:|---:|---:|---:|---:|---:|
| signed 8-bit | 32 | 0.565 | 0.110 | 0.570 | 0.077 | 0.252 | 0.253 |
| signed 8-bit | 1,024 | 15.236 | 0.134 | 18.271 | 0.134 | 2.261 | 0.342 |
| signed 8-bit | 65,536 | 960.313 | 1.787 | 1039.859 | 4.063 | 132.750 | 6.535 |
| unsigned 64-bit | 32 | 0.571 | 0.149 | 0.615 | 0.128 | 0.346 | 0.364 |
| unsigned 64-bit | 1,024 | 15.545 | 0.376 | 13.364 | 0.447 | 2.168 | 0.704 |
| unsigned 64-bit | 65,536 | 960.594 | 17.195 | 771.172 | 21.414 | 121.150 | 22.747 |

## Limitations

The SBCL ARM64 backend is tracked separately; ARM64 currently uses native C
or Lisp. See [the ARM64 SBCL ticket](https://todo.sr.ht/~takeiteasy/trivial-simd/31).

Integer kernels read every input before writing; see
[kernel overlap](kernels.md#overlap). Shifted overlap in bulk integer operations
follows the [bulk overlap limitation](api.md#limitations).

[^dispatch]: A kernel can serve floats and integers. Type-dependent validation
    happens at call time, including for empty vectors. An integer type rejects
    float-only operators before execution.
[^reduction]: Reduction addition wraps to the input width. The order of
    additions is therefore immaterial for integer results.

[^benchmark]: Arrays contain 7 and 3. The benchmark calibrates each timed
    batch to at least 50 ms, repeats it three times, and retains the last result.
    Two other post-change runs showed the same performance ordering. The
    [raw run](benchmark-runs/2026-09-29-integers-sbcl-66.txt),
    [pre-optimization run](benchmark-runs/2026-09-29-integers-sbcl.txt), and
    [benchmark script](../tests/integer-bench.lisp) are available for reproduction.
