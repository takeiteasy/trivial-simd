#!/bin/sh
set -eu

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] \
    || { echo "This script needs an ARM64 Mac" >&2; exit 1; }
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
export RUNNER_ARCH=ARM64
export TRIVIAL_SIMD_BACKEND=${TRIVIAL_SIMD_BACKEND:-native}
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build build
ctest --test-dir build --output-on-failure

exec arch -arm64 sbcl --dynamic-space-size 4096 --noinform --no-sysinit --no-userinit \
    --non-interactive --load tests/run.lisp "$@"
