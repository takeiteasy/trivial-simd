# Row-batched kernels

A float `sum` kernel reduces several rows into a destination vector. Arithmetic
expressions execute in one native call; each row uses the same VM and addition
order as a separate invocation on that backend.

```lisp
(trivial-simd:define-kernel kernel-dot (a b)
  (trivial-simd:sum (* a b)))

(kernel-dot matrix weights
            :rows 1000 :row-length 32
            :a-row-stride 32 :b-row-stride 0
            :destination results)
```

`matrix` holds 1,000 contiguous rows of 32 elements. The zero stride reuses
`weights` for every row. `results` receives 1,000 dot products. All operands are
simple single-float or double-float vectors, or [foreign-memory views](vector-views.md)
of the same precision. [Input declarations](kernel-inputs.md) also accept
integer sources and block-repeated scales. The call returns `results`.

## Arguments

| Keyword | Meaning | Default |
|---|---|---|
| `:rows` | Number of output rows; activates batching when supplied | Required |
| `:row-length` | Elements reduced within each row | Required |
| `:destination` | Vector receiving the sums | Required |
| `:destination-start` | First output element | `0` |
| `<argument>-start` | First element of that input's first row | `0` |
| `<argument>-row-stride` | Signed distance between row starts, in elements | Row length |

Each row is contiguous. Padding, overlapping input rows, and zero-stride
broadcasting are valid. A negative row stride reads rows in reverse order:

```lisp
(kernel-dot matrix weights :rows 3 :row-length 4
            :a-start 8 :a-row-stride -4 :b-row-stride 0
            :destination results)
```

The input row starts are 8, 4, and 0. Output stays in forward order.

For [repeated inputs](kernel-inputs.md#row-batching), starts and strides count
stored entries, the default stride is `ceiling(row-length/repeat)`, and every
row restarts its block position.

Dimensions and starts are nonnegative integers. Strides are signed 64-bit
integers; the complete accessed spans must fit the vectors and native address
limits. Batch calls reject `:start` and `:end`; use `:row-length` and individual
input starts instead. Batch-only keywords require `:rows`.

Calls without batch keywords retain the ordinary [scalar reduction interface](kernels.md#reduction-kernels),
including its integer and complex support.

## Output and errors

The destination needs space for `:rows` elements at `:destination-start`.
Elements outside that region stay unchanged. Zero rows write nothing;
zero-length rows write zeros of the input precision without native execution.

The destination must not overlap any input's bounding span, including padding
between rows. Bounds, types, and overlap are checked before any output write.
Inputs may overlap each other. Distinct views of overlapping memory and views
into pinned Lisp vectors are checked by address on SBCL, CCL, and ECL.

An arithmetic error may leave partial output. Output contents after that error
are unspecified; temporary buffers are released.

## Execution

| Path | Batch execution |
|---|---|
| Native arithmetic sum | One foreign call for all nonempty rows |
| Native copy access | Copy input spans once, one foreign call, copy results back[^copy] |
| Lisp or SBCL SIMD | Existing reduction kernel invoked for each row |
| Sum containing comparisons or `select` | Existing scalar path invoked for each row |
| Native library without batch symbols | Existing reduction kernel invoked for each row |

Native spill scratch is private to the invocation and reused across its rows.
Programs are initialized lazily and reclaimed with their defining function.
Overlapping calls use separate mutable buffers. ECL uses the shared batch helper
compiled when the system loads, including for kernels defined with `eval`.

See [row-batch measurements](kernel-performance.md#row-batching) and
[the benchmark command](testing.md#row-batch-benchmark).

## Limitations

- Batching supports only single- and double-float `sum` kernels. Other reducers,
  integer sums, complex sums, and elementwise kernels reject batch keywords.
- Destination overlap is rejected conservatively: output inside unused row
  padding is also rejected.
- A single row can cost more than a scalar invocation. Short native scalar
  setup remains tracked in [#59](https://todo.sr.ht/~takeiteasy/trivial-simd/59).
- On implementations without pinned Lisp-vector access, overlap checks compare
  Lisp vector identity and foreign-view addresses; views into Lisp array storage
  require implementation-specific pinning support.

[^copy]: Each Lisp input transfers its bounding span, including padding, once.
    A broadcast input transfers only one row. Foreign views access their memory
    directly. Results need only `:rows` elements of temporary storage.
