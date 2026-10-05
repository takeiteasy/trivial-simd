# Typed and repeated kernel inputs

Kernel input declarations convert integers to float values on load and repeat
stored values over blocks. Arithmetic uses single- or double-float precision.

```lisp
(trivial-simd:define-kernel quantized-dot
    (x (q :type :s8) (scale :repeat 32))
  (trivial-simd:sum (* x q scale)))

(let ((x (make-array 33 :element-type 'single-float :initial-element 1f0))
      (q (make-array 33 :element-type '(signed-byte 8) :initial-element 2))
      (scale (make-array 2 :element-type 'single-float
                           :initial-contents '(0.25f0 0.5f0))))
  (quantized-dot x q scale)) ; => 17f0
```

The first 32 products use `scale[0]`; the last uses `scale[1]`. No expanded
weight or scale vector is required.

## Declarations

| Argument form | Values loaded |
|---|---|
| `x` | Ordinary float input |
| `(q :type :s8)` | Signed bytes converted to the kernel precision |
| `(scale :repeat 32)` | One float value per 32 logical elements |
| `(q :type :u8 :repeat 3)` | One unsigned byte per three elements, converted on load |

`:type` accepts the eight [integer types](integers.md) and `:f32`/`:f64`.
Float inputs match the computation precision; they do not convert between
precisions. Integer conversion rounds to that precision without saturation.
`:repeat` is a positive integer constant, defaulting to 1. Unknown or duplicate
options, duplicate names and invalid declarations signal errors at definition.

A numeric destination determines the precision of an elementwise kernel.
Reductions and mask-output kernels use the first non-repeated float input.
At least one non-repeated input establishes the logical length. Other ordinary
float inputs match the precision. Arrays and [foreign views](vector-views.md)
may be mixed.

All existing float expressions, numeric reducers, comparisons, selection,
mask outputs and mask reducers accept declarations. Float `sum` kernels also
support [row batching](kernel-rows.md).

## Slices and block alignment

Ordinary inputs retain the [kernel slice interface](kernels.md). Without
`:end`, non-repeated inputs and an elementwise destination have equal logical
lengths. Repeated inputs only need enough stored values for the accessed range.

For repetition factor `B`, input start `S`, and slice-relative index `i`, a
repeated input reads `input[S + floor((start+i)/B)]`. `S` defaults to zero and
counts stored entries. Block position follows `:start`, independently of other
inputs' individual starts.

```lisp
(quantized-dot x q scale :start 5 :end 34)
; Uses scale[0] for logical indices 5..31 and scale[1] for 32..33.

(quantized-dot x q scale :start 5 :end 34 :scale-start 2)
; Reads the same block positions from scale[2] and scale[3].
```

Types, bounds and prohibited overlap are checked before output writes.
Destination overlap with converted or repeated inputs is rejected, including
distinct views of overlapping memory. Ordinary elementwise inputs retain
in-place aliasing. Row batches reject output overlap with every input span.
Empty slices read no input values and retain the existing reducer results.

## Row batching

Each input start and row stride counts stored elements of that input.
The default repeated-input stride is `ceiling(row-length/B)`; ordinary inputs
use `row-length`. Each row starts a new repetition block.

```lisp
(quantized-dot matrix q scales :rows 2 :row-length 33
               :destination results)
; Row 0 uses scales[0..2); row 1 uses scales[2..4).

(quantized-dot matrix q scales :rows 2 :row-length 33
               :q-row-stride 0 :scale-row-stride 0
               :destination results)
; Both rows reuse the same quantized weights and scales.
```

Padding, zero-stride reuse and negative row strides follow the ordinary
row-batch rules. Zero rows write nothing; zero-length rows write typed zeros.

## Execution

| Backend | Declared-input execution |
|---|---|
| `:native` | Source-type descriptors; packed ARM64 loaders; bounded buffers; float VM |
| `:lisp` | Typed scalar loads and arithmetic |
| `:sbcl` | Typed Lisp path |
| Native library missing descriptor symbols | Typed Lisp fallback |

The native path prepares converted and repeated inputs once per 256-element
block and uses one foreign call for an ordinary kernel or sum batch.[^norm] Preparation and
spill storage are private to the invocation and reused across blocks and rows.
Source-type dispatch occurs once per input and block. ARM64 uses NEON integer
conversion and repeated-value broadcasts. Repeated inputs load and convert each
stored value once per run. Other native targets use typed scalar conversion
and SIMD broadcasts where available. The 64-bit-to-f32 path uses direct scalar
conversion to preserve rounding.[^rounding]

Copy access transfers stored source spans in their original types, rather than
full-length float expansions. Lisp view staging preserves repetition phase
across buffer boundaries.[^staging]

## Limitations

- Computation is f32/f64 only. Integer and complex kernels use bare arguments.
- Packed quantization formats require decoding before the kernel call.
- Float inputs do not convert between precisions.
- Declared-input kernels use scalar Lisp loops under `:sbcl`; compiler
  integration is tracked in [#129](https://todo.sr.ht/~takeiteasy/trivial-simd/129).
- Native x86-64 integer preparation uses typed source loops. Additional packed
  conversion primitives are tracked in
  [#72](https://todo.sr.ht/~takeiteasy/trivial-simd/72).
- On implementations without pinned array access, overlap checks compare array
  identity and foreign addresses; they cannot detect a foreign view into array
  storage.

[^staging]: View buffers hold source elements, including the stored scale entries
    needed for the current block. Overlapping ordinary view inputs use the
    existing forward/reverse staging rules; conflicting directions require a
    whole-slice buffer. Native copy access may copy a full accessed source span.

[^norm]: Native `nrm2` can evaluate the expression in additional passes to rescale
    extreme double-float magnitudes, as with ordinary kernel inputs.

[^rounding]: ARM64 widens s8/u8/s16/u16/s32/u32 directly to f32 or f64, and
    s64/u64 directly to f64. Converting s64/u64 through f64 before f32 can round
    twice; volatile scalar reads prevent Clang from introducing that intermediate.
