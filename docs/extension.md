# Extension API

`trivial-simd/extension` exports the low-level helpers that libraries built on
trivial-simd, such as [CLBLAS](https://github.com/takeiteasy/CLBLAS), use to
reach vectors and the native library.

| Symbol | Use |
|---|---|
| `vector-type` | Keyword for a simple vector's element type (`:f32`, `:c64`, …) |
| `*native-array-access*` | `:pointer` when native calls see pinned vector data, `:copy` otherwise |
| `with-pinned-pointers` | Bind pointers to the data of simple vectors for the body |
| `element-pointer` | Pointer advanced by an element offset for a foreign type |
| `define-native` | Define a Lisp function as a foreign call to a C symbol |
| `absolute-argmax` | Index of the first largest-magnitude element of a real float slice |

```lisp
(trivial-simd/extension:with-pinned-pointers ((p vector))
  (trivial-simd/extension:element-pointer p :float 2))
```

Pinned access exists on SBCL, CCL and ECL.

These are internals made visible: signatures follow the library and are not
covered by the [API](api.md) stability of the bulk operations.
