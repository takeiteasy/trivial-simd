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

TRIVIAL_SIMD_TEST_RUN=1 arch -arm64 ccl --no-init --batch --load tests/run.lisp "$@" --eval '(quit)'
TRIVIAL_SIMD_TEST_RUN=2 arch -arm64 ccl --no-init --batch --load tests/run-lifetime-stress.lisp --eval '(quit)'
for trial in 1 2 3; do
    TRIVIAL_SIMD_TEST_RUN=$((trial + 2)) arch -arm64 ccl --no-init --batch --load tests/run.lisp --eval '(quit)'
done
