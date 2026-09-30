# Type conversion

`(convert! destination input &key start end destination-start input-start
rounding)` converts between every pair of the [ten numeric vector
types](api.md). It writes and returns `destination`. The vectors may have
different element types; the usual [slice rules](api.md#slices) apply.

| Conversion | Behavior |
|---|---|
| Integer to integer | Clamp to the destination range |
| Integer to float | Round to the destination float precision |
| Float to float | Round to the destination float precision |
| Float to integer | Apply `:rounding`, then clamp to the destination range |

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

## Limitations

Most type pairs use scalar loops; expanding packed SIMD coverage is tracked in
[#72](https://todo.sr.ht/~takeiteasy/trivial-simd/72).
Shifted overlap follows the [bulk overlap limitation](api.md#limitations).
Floating-point exceptional behavior outside the float-to-integer rules is
covered by the [IEEE consistency limitation](kernels.md#limitations).
