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
`:floor`, and `:ceiling`. These modes also apply to
[bf16/f16 narrowing](#bf16-and-f16); other conversions ignore `:rounding`.
Positive and negative infinity clamp to the corresponding integer bound. Converting NaN
to an integer signals an error. An arithmetic error may leave some destination
elements updated.

```lisp
(trivial-simd:convert! integer-output float-input :rounding :truncate)
```

Packed conversion coverage and the size-dependent dispatch rules are listed
in [execution paths](#execution-paths).

## bf16 and f16

Lisp has no 16-bit float type, so bf16 (bfloat16) and f16 (IEEE binary16)
values are stored as their bit patterns in `(unsigned-byte 16)` vectors.
`:input-encoding` and `:destination-encoding` name the encoding of each
unsigned-byte-16 vector, `:bf16` or `:f16`. The other operand may be
`single-float`, `double-float`, or another encoded vector:

```lisp
(let ((weights (make-array n :element-type '(unsigned-byte 16)))
      (floats (make-array n :element-type 'single-float)))
  ;; ... read raw bf16 weights into WEIGHTS ...
  (trivial-simd:convert! floats weights :input-encoding :bf16)
  (trivial-simd:convert! weights floats :destination-encoding :bf16))
```

Without an encoding keyword an `(unsigned-byte 16)` vector holds integers, as
for any other conversion. Both encoding keywords convert between encoded
formats. Identical encodings copy raw bits, including NaN payloads, and accept
in-place use. Shifted overlap follows the [bulk overlap limitation](api.md#limitations).

```lisp
(trivial-simd:convert! halves doubles :destination-encoding :f16)
(trivial-simd:convert! doubles halves :input-encoding :f16)
(trivial-simd:convert! bfloat halves :input-encoding :f16 :destination-encoding :bf16)
```

Double-float narrowing rounds once from the original value to the destination
encoding. Cross-format conversions need no full-vector intermediate.[^extended]

| Direction | Behavior |
|---|---|
| bf16 or f16 to single-float or double-float | Exact, including subnormals, infinities and signed zeros |
| single-float or double-float to bf16 or f16 | Apply the requested rounding mode, including at subnormal and overflow boundaries |
| NaN when formats differ | Quiet NaN with the same sign and the leading payload bits that fit |

Narrowing accepts all four rounding modes. Exact values and signed zeros are
unchanged. Widening is exact, so it accepts any valid `:rounding`. Infinities
and NaN are unaffected by the rounding mode.

| `:rounding` | Inexact finite values | Positive finite overflow | Negative finite overflow |
|---|---|---|---|
| `:nearest-even` (default) | Nearest value, ties to even | +infinity | -infinity |
| `:truncate` | Toward zero | Largest finite value | Negative largest finite value |
| `:floor` | Toward -infinity | Largest finite value | -infinity |
| `:ceiling` | Toward +infinity | +infinity | Negative largest finite value |

```lisp
(let ((floats (make-array 2 :element-type 'single-float
                           :initial-contents '(1.0001 -1.0001)))
      (halves (make-array 2 :element-type '(unsigned-byte 16))))
  (trivial-simd:convert! halves floats :destination-encoding :f16 :rounding :floor)
  (equalp halves #(#x3c00 #xbc01))) ; => T
```

A positive nonzero input below the smallest subnormal rounds to that subnormal
with `:ceiling`, or to +0 with `:floor`. A negative input of the same magnitude
rounds to the negative smallest subnormal with `:floor`, or to -0 with
`:ceiling`.

With nearest-even, f16 rounds values from 65520 upward to infinity
and values at or below 2^-25 to a signed zero; values between round to f16
subnormals. bf16 shares the single-float exponent range, so single-float
subnormals round to bf16 subnormals, and only magnitudes that round above the
largest bf16 value become infinity. A NaN never rounds to infinity.

The conversions do not depend on the caller's floating-point rounding mode,
flush-to-zero setting or trap mask, and raise no floating-point exceptions.
Every backend produces the same bits.[^fpcr] The native and SBCL backends use
the native library when it supports the requested mode.[^capability][^paths]

## Execution paths

| Conversion | Packed path |
|---|---|
| f32 ↔ f64 | Native SSE2/NEON |
| s8/u8/s16/u16/s32/u32 → f32/f64 | Shared native SSE2/NEON loaders |
| s64/u64 → f64 | Native NEON |
| s16/u16 → s8/u8; s32/u32 → s16/u16 | Native saturating SSE2/NEON |
| s32 → f32 on the SBCL backend | In-process SSE2; large pointer calls use native when available |
| f64 ↔ bf16/f16 | Native packed integer conversion with scalar exceptional lanes |

Additional native numeric conversions use pointer access for Lisp-vector slices
of at least 32,768 elements. Smaller slices and copy-mode Lisp vectors use typed
Lisp loops. Direct foreign views use native conversion at every size when the
pair is supported. Missing native conversion symbols select Lisp. The shared
loaders also prepare contiguous declared integer inputs; repeated-input runs
retain broadcast preparation.

See [conversion and mask measurements](conversion-mask-performance.md).

## Limitations

Float-to-integer and integer pairs outside the packed coverage table use scalar
loops. Direct s64/u64-to-f32 conversion stays scalar to preserve single rounding.
Further native coverage and platform tuning are tracked in
[#156](https://todo.sr.ht/~takeiteasy/trivial-simd/156); additional in-process SBCL
conversion pairs are tracked in [#157](https://todo.sr.ht/~takeiteasy/trivial-simd/157).
Shifted overlap follows the [bulk snapshot rules](api.md#overlap).
Floating-point exceptional behavior outside the float-to-integer rules is
covered by the [IEEE consistency limitation](kernels.md#limitations).

[^fpcr]: The bf16 paths and x86-64 narrowing use integer instructions. SSE2
    f16 widening uses an exact subtraction for subnormals and clears its sign
    before restoring the source sign, preserving signed zeros in every caller
    rounding mode. The ARM64 f16 loops use `FCVTL`/`FCVTN`, so they save FPCR and FPSR, run with FPCR set
    to the requested rounding mode, no flush-to-zero and no traps, and restore
    both on return. Widening uses round to nearest even.

[^capability]: The native library reports supported modes through
    `ts_extended_float_encoding_rounding_modes()`: bits 0–3 correspond to
    nearest-even, truncate, floor and ceiling. A library without this symbol
    supports only nearest-even and truncate; directed narrowing uses Lisp.
    Rebuild with CMake to enable native directed rounding. If encoding symbols
    are missing, all encoded conversions use Lisp.

[^paths]: Native f32 encoding conversions use SSE2 integer code on x86-64 for both encodings,
    and NEON on ARM64, with `FCVTL`/`FCVTN` for f16. On x86-64, a block of eight
    elements containing an f16 subnormal result uses scalar conversion, as do
    vector tails. Directed rounding also sends nonzero inputs below the f16
    normal range through that scalar path. Windows ARM64 uses scalar f16
    conversion. The Lisp backend uses typed integer loops.

[^extended]: The native encoded cross-format path reuses a 256-element f32 stack
    buffer; both source formats widen exactly to f32. f64 narrowing uses packed
    integer shifts and rounding for normal lanes, and scalar bit conversion for
    exceptional lanes and tails. Encoded widening to f64 uses packed integer
    construction for normal lanes; zeros, encoded subnormals, infinities and NaNs
    use scalar bit conversion. The Lisp paths convert one element at a
    time. Stride staging and native copy mode retain their existing allocation
    rules. The extended native entry point reports rounding support separately;
    missing symbols or modes select Lisp without disabling the existing f32 paths.
