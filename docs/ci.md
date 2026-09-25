# CI

CI runs a small set of jobs automatically and the full platform matrix on
request.

## When jobs run

| Trigger | Jobs |
|---|---|
| Push to `trunk` or pull request | Tier 1, chosen by changed paths |
| Docs-only change | None |
| `[ci full]` in the latest commit message | Full matrix |
| `ci-full` label on a pull request | Full matrix on every push while labelled |
| Tag `v*` | Full matrix |
| Manual dispatch | Full matrix, narrowed by inputs |

New pushes cancel the previous run for the same branch or pull request.

## Tiers

| Tier | Jobs | Runs |
|---|---|---|
| 1 | SBCL / Linux, Lisp fallback / Linux | Any code change |
| 1 | ECL / Linux | Native code changes[^focus] |
| 2 | Every other platform and Lisp | Full matrix only |

The job list lives in [`.github/ci-matrix.json`](../.github/ci-matrix.json).
[`.github/scripts/plan.sh`](../.github/scripts/plan.sh) selects from it.

## Running a slice

```sh
gh workflow run ci.yml -f os=macos              # every macOS job
gh workflow run ci.yml -f os=linux -f lisp=ccl  # CCL on Linux only
gh workflow run ci.yml -f lisp=ecl              # ECL on every OS
```

`os` is `all`, `linux`, `macos`, or `windows`. `lisp` is matched against the
Lisp name in the matrix (`sbcl`, `ccl`, `ecl`).

## Before dispatching

Run the suite locally first; see [testing](testing.md). A local run on one
platform catches most failures; dispatch a slice for the platform you cannot
run.

## sr.ht builds

[`.build.yml`](../.build.yml) runs the suite on Linux with SBCL on every push
to the sr.ht mirror, at no GitHub cost. It covers the same ground as the tier 1
SBCL job.

## Cost

Private repositories bill macOS minutes at about ten times and Windows at about
twice the Linux rate; public repositories use standard runners for free.[^cost]

[^focus]: Paths matching `native/`, `native.lisp`, `backends.lisp`,
    `kernel.lisp`, or `CMakeLists.txt`. The path lists are in the `paths`
    section of the matrix file.
[^cost]: GitHub's current rates are in its Actions billing documentation.
