# Test diagnostics

`TRIVIAL_SIMD_TEST_TIMING=1` prints flushed phase and per-test timings.
`TRIVIAL_SIMD_TESTS` selects test names separated by spaces or commas;
without it the complete suite runs. `TRIVIAL_SIMD_TEST_RUN` labels fresh
processes. `TRIVIAL_SIMD_TEST_PHASES=1` also prints each BLAS phase before
execution. Native BLAS comparisons report setup, Lisp execution, native
execution and comparison totals separately.[^diagnostics]

```sh
TRIVIAL_SIMD_TEST_TIMING=1 \
TRIVIAL_SIMD_TESTS=blas-native-gemm-matches-lisp \
python3 tests/bounded-run.py --log /tmp/ccl-gemm.log -- \
  ccl --no-init --batch --load tests/run.lisp --eval '(quit)'
```

The wrapper defaults to 300 seconds, retains stdout/stderr in the named log,
and exits 124 on timeout. It terminates the command's process group; use
`--seconds` to choose another bound. Use it around either local architecture
script or a direct Lisp command. The scripts retain full coverage; selecting
tests is for diagnosis, not a full validation run.

## Limitations

Rosetta CCL has costly cold compilation, test-data setup and lifetime trials.
The Linux x86-64 stall is unconfirmed; local macOS timings do not establish
Linux behavior. Remaining investigation is tracked in
[#125](https://todo.sr.ht/~takeiteasy/trivial-simd/125).

[^diagnostics]: Per-test timing wraps FiveAM's test execution method. Setup and
    comparison timings wrap the native BLAS test helpers only during a timed
    run; original functions are restored afterward. Timings include assertions
    and instrumentation overhead and are diagnostic rather than benchmarks.
