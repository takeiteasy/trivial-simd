# Rosetta concurrency diagnostics

CCL x86-64 concurrency checks run under Rosetta on an ARM64 Mac. Host aborts,
Lisp assertion failures, numeric mismatches, and diagnostic timeouts are separate
outcomes. A timeout does not establish a deadlock.

## Run the investigation

Use an isolated checkout with an x86-64 native library. The runner warms separate
ASDF caches for each process lane and retains every log and exit status. Cache
verification runs the ownership check; measured trials also run concurrency.

```sh
mkdir -p /tmp/rosetta-checkout
rsync -a --exclude .git --exclude build ./ /tmp/rosetta-checkout/
cmake -S /tmp/rosetta-checkout -B /tmp/rosetta-checkout/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=x86_64
cmake --build /tmp/rosetta-checkout/build
python3 tests/rosetta-concurrency.py \
  --checkout /tmp/rosetta-checkout \
  --ccl "$HOME/.cache/trivial-simd/x86-ccl/ccl-1.13/dx86cl64" \
  --revision "$(git rev-parse HEAD)" \
  --log-dir /tmp/rosetta-concurrency-results
```

Choose a new log directory for each investigation. Keep other Rosetta Lisp
processes stopped during serial and paired trials. Results are diagnostic
wall-clock measurements, not controlled performance benchmarks.

The defaults below use 20 serial trials and 20 pairs. Set `--serial-trials 5
--paired-trials 5` for a smaller bounded investigation. Each pair contains two
fresh processes; the three background-suite trials are independent of these
counts.

Use `--background-checkout PATH --background-revision REVISION` to validate a
different checkout in the background suites. Both checkouts need an x86-64
native library. Results record each process's checkout and revision; the
background checkout's initial cache verification has a 900-second bound.

| Phase | Fresh processes | Bound per process |
|---|---:|---:|
| Cold cache warm-up | 1 | 900 seconds |
| Separate cache verification | 4 | 300 seconds |
| Serial targeted checks | 20 | 300 seconds |
| Simultaneous targeted pairs | 20 pairs | 300 seconds |
| Targeted checks alongside a full suite | 3 pairs | Targeted: 300 seconds; full suite: 900 seconds |

Targeted processes run `complex-mask-spilling-and-concurrent-calls` and
`complex-native-constant-ownership`. The first test exercises spilling and four
concurrent workers for both complex precisions. The second checks foreign
constant ownership. `metadata.json` records the checkout revision, command and
host versions; `results.json` records process labels, wrapper PIDs, exit statuses
and durations. Individual logs retain phase timings and assertion output.

## Isolate a reproduced abort

`tests/rosetta-isolation.lisp` runs a small dependent complex kernel with four work
items executed sequentially or by four CCL workers, for both precisions. Choose
the Lisp or native backend; `plain` exercises threaded complex arithmetic without
loading the library.
The smaller kernel omits the spilling expression from the original test.

```sh
TRIVIAL_SIMD_DIAGNOSTIC_ROOT=/tmp/rosetta-checkout/ \
TRIVIAL_SIMD_ISOLATION_BACKEND=native \
TRIVIAL_SIMD_ISOLATION_WORKERS=4 \
python3 tests/bounded-run.py --seconds 300 --log /tmp/rosetta-native-4.log -- \
  arch -x86_64 "$HOME/.cache/trivial-simd/x86-ccl/ccl-1.13/dx86cl64" \
  --no-init --batch --load tests/rosetta-isolation.lisp
```

Run at most ten fresh processes per backend/worker case and matching ARM64 controls.
Stop an isolation case earlier when it already establishes a smaller reproducer;
retain the completed count and every failure.
Use a writable ASDF output cache, as in the main runner. A warmed cache from an
earlier investigation may seed a new run with `--cache-seed`; source freshness
still controls recompilation.

## Observed outcomes

The [retained results](benchmark-runs/2026-10-06-rosetta-concurrency.json) record
trial counts, exit statuses, durations and local raw-log directories.

| Targeted execution condition | Completed measured trials | Result |
|---|---:|---|
| Serial processes | 5 | 2 passes; 3 timeouts |
| Simultaneous pairs | 0 | Stopped before execution on the user's wrap-up request |
| Alongside a full suite | 0 | Stopped before execution on the user's wrap-up request |

Cache verification also retains a host assertion from a single process. Warm-up
and verification results are separate from measured serial trials.

A four-worker plain complex arithmetic control reproduces the host register-state
assertion without loading trivial-simd or its native library. A single targeted
Rosetta process also reproduces it; simultaneous Lisp processes are not required.
Five serial targeted trials yield
two passes and three diagnostic timeouts. A smaller four-worker native kernel
also waits during double-float work outside the agent sandbox. Sequential native
and four-worker Lisp controls pass ten trials each.

| Smaller isolation case | Rosetta ordinary CLI | ARM64 ordinary CLI | ARM64 sandbox |
|---|---|---|---|
| Lisp, sequential | 10 passes | 10 passes | 10 passes |
| Lisp, four workers | 10 passes | 10 passes | 9 passes; 1 SIGBUS |
| Native, sequential | 10 passes | 10 passes | 10 passes |
| Native, four workers | 7 passes; 3 timeouts | 10 passes | 10 passes |
| Plain arithmetic, four workers | 2 host assertions; 1 timeout | 10 passes | 10 passes |

The plain Rosetta case stops after three completed trials because both failure
forms reproduce without the library. Operator-cancelled subsequent startups
are excluded from measured counts.

Feature validation passes full suites on ARM64 SBCL, CCL and ECL, plus the
final 10,056 overlap checks on each. X86-64 full suites pass 613,477 checks on
SBCL and 469,099 on ECL; native C tests pass on both architectures. The x86-64
CCL overlap attempt is operator-cancelled while progressing through its first
exhaustive test, with no completed suite result. Full x86-64 CCL validation is
incomplete.[^validation]

Samples of the original, smaller native and plain waiting processes reach
`gc_like_from_xp` and `suspend_other_threads`. This identifies an observed runtime
wait path, without establishing its cause. The smaller kernel omits the spilling
expression, so the evidence does not require that expression to produce a wait.

ARM64 controls pass all 50 ordinary CLI trials. A separate set of 50 sandboxed
ARM64 trials retains one four-worker Lisp SIGBUS and 49 passes. These sets keep
execution context explicit; passing controls do not prove the intermittent
failure is absent.[^environment]

## Limitations

The root cause of an intermittent host register-state assertion is unconfirmed.
Passing retries do not establish a Rosetta fix; investigation remains tracked
by [#60](https://github.com/communal-software/trivial-simd/issues/60).[^assertion]
Cold compilation, suite setup and lifetime overhead remain tracked separately
by [#56](https://github.com/communal-software/trivial-simd/issues/56). These macOS checks do
not establish the cause of a Linux CCL stall. The cancelled overlap validation
also shows slow Rosetta execution with substantial exception handling in its
sample; its cause is unconfirmed.
Cold concurrent Lisp complex kernels also have an intermittent ARM64 CCL SIGBUS,
tracked separately by the CCL ARM64 SIGBUS investigation.

[^assertion]: The host abort reports
    `assertion failed: state update should occur from waiters' queue`, in
    `ThreadContextRegisterState.cpp:1076 host_fpr_state_from_guest_state`.
    It is distinct from a Lisp assertion failure or a native VM result mismatch.

[^environment]: The diagnostic host uses macOS 15.7.5 (24G624), Apple M1 and
    Rosetta package version 1.0.0.0.1.1773146330. Rosetta trials use CCL 1.13
    (v1.13), DarwinX8664. The original targeted checkout is
    `b334cd0419ac78e7b4e7e46d4de86cbb7d145dfc`; the native library is built
    for x86-64. Diagnostic bounds retain waiting processes as timeouts rather
    than treating successful retries as replacements for failed trials.

[^validation]: Feature code is `84087c9`. ARM64 full-suite results precede the
    final expansion of stride mappings in the overlap tests; the expanded tests
    pass separately on all three Lisps. ARM64 full-suite counts are 463,090
    (SBCL), 463,080 (CCL) and 465,865 (ECL). The x86-64 SBCL and ECL full suites
    include the expanded tests. CCL's cancelled x86-64 attempt has a retained
    log and one-second sample under `/private/tmp/tickets141-x86-ccl-overlap*`.
