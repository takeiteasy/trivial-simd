# Elementwise operations

`negate!`, `abs!`, `sqrt!`, `reciprocal!`, `min!`, `max!`, and `clamp!` write
into a caller-provided destination and return it. All accept the [bulk slice
keywords](api.md#slices). Inputs and destination have the same element type,
except complex `abs!` writes a real vector of matching precision.

| Function | Operation | Types |
|---|---|---|
| `(negate! destination input &key ...)` | `-input` | Real, complex, integer |
| `(abs! destination input &key ...)` | Absolute value | Real, complex, integer |
| `(sqrt! destination input &key ...)` | Square root | Real, complex |
| `(reciprocal! destination input &key ...)` | `1/input` | Real, complex |
| `(min! destination left right &key ...)` | Elementwise minimum | Float and integer |
| `(max! destination left right &key ...)` | Elementwise maximum | Float and integer |
| `(clamp! destination input lower upper &key ...)` | Limit to `[lower, upper]` | Float and integer |

`min!` and `max!` accept one scalar operand. `clamp!` accepts scalar or vector
bounds. Scalars have exactly the destination element type; integer scalars fit
the vector width. A lower bound greater than its upper bound signals an error.
Equal values in `min!` and `max!` retain the left operand, including its zero
sign. Integer negation and absolute value [wrap](integers.md) at the vector
width.

```lisp
(trivial-simd:negate! out input :start 2 :end 6 :input-start 8)
(trivial-simd:clamp! out input 0.0 1.0)
```

The unary functions use `:destination-start` and `:input-start` for separate
offsets. `min!` and `max!` use `:destination-start`, `:left-start`, and
`:right-start`. `clamp!` uses `:destination-start`, `:input-start`,
`:lower-start`, and `:upper-start`. A start keyword for a scalar signals an
error. Destinations may also be an input at the same slice position.

Use [N-D entry points](nd.md) for shapes with several strided axes or zero-stride broadcasting.

## Limitations

Shifted overlap between destination and input slices has
[backend-dependent results](https://todo.sr.ht/~takeiteasy/trivial-simd/67).
Floating-point NaNs, infinities, non-default rounding modes, and traps follow
the [IEEE consistency limitation](kernels.md#limitations).
