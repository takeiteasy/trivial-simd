# Kernel expansion size

Complex support keeps representative real/integer kernel expansions within a
10% growth budget. The optional diagnostic measures expanded forms, cold kernel
compilation, and compiled-file size; it does not time kernel execution.

## Expansion measurements

Apple M1, measured 2026-10-06 against reference checkout `0cb596a`.
SBCL and CCL report the same expansion sizes.[^nodes]

| Kernel | Reference nodes | Current nodes | Growth |
|---|---:|---:|---:|
| Comparison | 3,203 | 3,335 | 4.1% |
| Selection | 3,467 | 3,615 | 4.3% |
| Mask count | 2,961 | 3,093 | 4.5% |
| Sum of selection | 1,611 | 1,759 | 9.2% |
| Arithmetic | 2,457 | 2,415 | -1.7% |
| Seven-level nested selection | 48,231 | 51,907 | 7.6% |

Complex expression trees belong to shared state. Helpers and native programs
are prepared only for complex precisions that a kernel uses.

## Compilation measurements

Each compiled file contains the six kernels above. Values are medians of five
fresh processes; compilation excludes system and dependency loading.[^timing]

| Lisp | Reference seconds | Current seconds | Reference bytes | Current bytes |
|---|---:|---:|---:|---:|
| SBCL 2.6.8 ARM64 | 0.743 | 0.739 | 261,553 | 264,521 |
| CCL `v1.13-459-g690ff7ea` ARM64 | 0.474 | 0.481 | 368,515 | 375,804 |

Compiled size grows by 1.1% on SBCL and 2.0% on CCL. See the retained
[SBCL measurements](benchmark-runs/2026-10-06-kernel-size-sbcl.txt) and
[CCL measurements](benchmark-runs/2026-10-06-kernel-size-ccl.txt).

## Run the diagnostic

`TRIVIAL_SIMD_SIZE_ROOT` selects a reference checkout. Without it, the script
loads this checkout. `TRIVIAL_SIMD_SIZE_BASELINE` checks each expansion against
the `SIZE` lines in a saved reference log and fails above 10% growth.

```sh
TRIVIAL_SIMD_SIZE_ROOT=/path/to/reference/ \
  sbcl --dynamic-space-size 4096 --script tests/kernel-size.lisp > /tmp/reference-size.log

TRIVIAL_SIMD_SIZE_BASELINE=/tmp/reference-size.log \
  sbcl --dynamic-space-size 4096 --script tests/kernel-size.lisp

ccl --no-init --batch --load tests/kernel-size.lisp --eval '(quit)'
```

The script leaves uniquely named source and compiled files in the temporary
directory for inspection. Use `trash` to discard them.

[^nodes]: A node is one cons or atom in the fully expanded outer
    `define-kernel` form. The count includes quoted metadata and all generated
    real/integer branches; it is independent of printed gensym names.

[^timing]: Compiler timings vary with machine load and compiler state. The
    expansion budget is the deterministic acceptance check. Compiled-file size
    also includes source-location and compiler metadata.
