# Type conversion

`(convert! destination input &key start end destination-start input-start
rounding destination-encoding input-encoding)` converts among the
[numeric vector types](api.md) and the [bf16 and f16](#bf16-and-f16) storage
encodings. It writes and
returns `destination`. The vectors may have
different element types; the usual [slice rules](api.md#slices) apply.

| Conversion | Behavior |
|---|---|
| Integer to integer | Clamp to the destination range |
| Integer to float | Round to the destination float precision |
| Float to float | Round to the destination float precision |
| Float to integer | Apply `:rounding`, then clamp to the destination range |
| Real or integer to complex | Convert the real component; use zero imaginary component |
| Complex to complex | Round both components to the destination precision |
| Complex to real or integer | Require zero imaginary component, then convert the real component |

Float-to-integer `:rounding` accepts `:nearest-even` (default), `:truncate`,
`:floor`, and `:ceiling`. It has no effect on other conversions. Positive and
negative infinity clamp to the corresponding integer bound. Converting NaN
to an integer signals an error. An arithmetic error may leave some destination
elements updated.

```lisp
(trivial-simd:convert! integer-output float-input :rounding :truncate)
```

The native backend uses packed SIMD for single-float/double-float conversion
where the CPU provides it. The SBCL backend uses that path when the native
library is present. Other pairs use typed scalar conversion.

## bf16 and f16

Lisp has no 16-bit float type, so bf16 (bfloat16) and f16 (IEEE binary16)
values are stored as their bit patterns in `(unsigned-byte 16)` vectors.
`:input-encoding` or `:destination-encoding` names the encoding of that
vector, `:bf16` or `:f16`, and the other vector must be `single-float`:

```lisp
(let ((weights (make-array n :element-type '(unsigned-byte 16)))
      (floats (make-array n :element-type 'single-float)))
  ;; ... read raw bf16 weights into WEIGHTS ...
  (trivial-simd:convert! floats weights :input-encoding :bf16)
  (trivial-simd:convert! weights floats :destination-encoding :bf16))
```

Without an encoding keyword an `(unsigned-byte 16)` vector holds integers, as
for any other conversion. Only one side may be encoded; to convert between bf16
and f16, or to or from double-float, convert through a single-float vector.

| Direction | Behavior |
|---|---|
| bf16 or f16 to single-float | Exact, including subnormals, infinities and signed zeros |
| single-float to bf16 or f16 | Round to nearest, ties to even; overflow becomes infinity of the same sign |
| NaN in either direction | Quiet NaN with the same sign and the leading payload bits that fit |

Narrowing ignores `:rounding`. f16 rounds values from 65520 upward to infinity
and values at or below 2^-25 to a signed zero; values between round to f16
subnormals. bf16 shares the single-float exponent range, so single-float
subnormals round to bf16 subnormals, and only magnitudes that round above the
largest bf16 value become infinity. A NaN never rounds to infinity.

The conversions work on bit patterns, so their results do not depend on the
floating-point rounding mode, flush-to-zero setting or trap mask. Every backend
produces the same bits. The native and SBCL backends use the native library:
SSE2 integer code on x86-64 for both encodings, and NEON on ARM64, with the
`FCVTL`/`FCVTN` conversion instructions for f16. On x86-64, a block of eight
elements with one that rounds to an f16 subnormal takes the scalar path, as do
vector tails. Windows ARM64 converts f16 with scalar code. The Lisp backend
uses typed integer loops.

## Limitations

Most type pairs use scalar loops; expanding packed SIMD coverage is tracked in
[#72](https://todo.sr.ht/~takeiteasy/trivial-simd/72).
Shifted overlap follows the [bulk overlap limitation](api.md#limitations).
Floating-point exceptional behavior outside the float-to-integer rules is
covered by the [IEEE consistency limitation](kernels.md#limitations).
