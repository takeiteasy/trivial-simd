# Copy, fill and swap

`copy!`, `fill!` and `swap!` move or set whole slices of any
[numeric vector](api.md), including complex and integer vectors. All accept the
[bulk slice keywords](api.md#slices).

| Function | Effect | Returns |
|---|---|---|
| `(copy! destination source &key ...)` | `destination[i] = source[i]` | `destination` |
| `(fill! destination value &key ...)` | `destination[i] = value` | `destination` |
| `(swap! x y &key ...)` | Exchange `x[i]` and `y[i]` | `x` and `y` as two values |

```lisp
(trivial-simd:copy! out a :start 4 :end 12)                   ; out[4..12) = a[4..12)
(trivial-simd:copy! out a :end 8 :destination-start 16)       ; out[16..24) = a[0..8)
(trivial-simd:fill! out 0.0f0 :start 2 :end 10)
(trivial-simd:swap! x y :end 8 :y-start 8)                    ; x[0..8) <-> y[8..16)
(trivial-simd:copy! out a :end 4 :source-stride 2)            ; out[0..4) = a[0], a[2], a[4], a[6]
(trivial-simd:fill! out 0.0f0 :end 4 :stride 3)               ; out[0], out[3], out[6], out[9]
```

The `-stride` keywords step through a slice; see [strides](api.md#strides).

| Keyword | Applies to |
|---|---|
| `:start`, `:end` | Every function |
| `:destination-start`, `:source-start` | `copy!` |
| `:x-start`, `:y-start` | `swap!` |
| `:stride` | Every function |
| `:destination-stride`, `:source-stride` | `copy!` |
| `:x-stride`, `:y-stride` | `swap!` |

## Rules

- `copy!` and `swap!` need the same element type in both vectors. `fill!`
  needs a `value` of exactly the element type; integer values fit the vector's
  signed or unsigned range.
- Overlapping slices of one vector copy as if the source were read first, in
  either direction, with or without strides.[^overlap] `swap!` rejects overlapping
  output slices before writes unless their element mappings are identical, in
  which case it leaves them unchanged. Disjoint interleaved slices are valid.
- Slice errors, empty slices and start keywords behave as in the
  [other bulk functions](api.md#slices).

## Backends

| Operation | `:native` | `:sbcl`, `:lisp` |
|---|---|---|
| `copy!` | `replace` | `replace` |
| `fill!` | C fill from 4,096 bytes upward;[^threshold] otherwise `fill` | `fill` |
| `swap!` | C block swap from 64 bytes upward;[^threshold] otherwise a typed loop | Typed loop |

Complex vectors, and native calls in `:copy` array-access mode, use the Lisp
forms. Strided calls gather, run the contiguous operation, and scatter.

[^overlap]: Shifted overlaps are snapshotted before gathering or block staging,
    including views sharing foreign memory. The contiguous Lisp path uses
    `replace`; see the [bulk overlap contract](api.md#overlap).
[^threshold]: Native calls cost about 0.2 µs to set up, so they only win
    above these sizes on SBCL with an Apple M1. See
    [benchmarks](benchmarks.md#copy-fill-and-swap).
