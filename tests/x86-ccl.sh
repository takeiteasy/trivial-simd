#!/bin/sh
set -eu

version=${CCL_X86_VERSION:-1.13}
cache=${TRIVIAL_SIMD_X86_CCL_CACHE:-$HOME/.cache/trivial-simd/x86-ccl}
root=$(cd "$(dirname "$0")/.." && pwd)
ccl_dir=$cache/ccl-$version

[ "$(uname -s)" = Darwin ] || { echo "x86-ccl.sh needs macOS with Rosetta" >&2; exit 1; }
arch -x86_64 /usr/bin/true 2>/dev/null || { echo "Rosetta is not installed" >&2; exit 1; }

if [ ! -x "$ccl_dir/dx86cl64" ]; then
    mkdir -p "$ccl_dir"
    curl -fsSL -o "$cache/ccl-$version.tar.gz" \
        "https://github.com/Clozure/ccl/releases/download/v$version/ccl-$version-darwinx86.tar.gz"
    tar xzf "$cache/ccl-$version.tar.gz" -C "$ccl_dir" --strip-components=1
fi

# The Lisp loader expects build/, so the x86-64 build lives in a copy of the tree.
work=$cache/work
mkdir -p "$work/build"
rsync -a --delete --exclude build --exclude .git "$root/" "$work/"
cd "$work"
export RUNNER_ARCH=X64
export TRIVIAL_SIMD_BACKEND=${TRIVIAL_SIMD_BACKEND:-native}
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=x86_64
cmake --build build
ctest --test-dir build --output-on-failure

TRIVIAL_SIMD_TEST_RUN=1 arch -x86_64 "$ccl_dir/dx86cl64" --no-init --batch --load tests/run.lisp "$@" --eval '(quit)'
TRIVIAL_SIMD_TEST_RUN=2 arch -x86_64 "$ccl_dir/dx86cl64" --no-init --batch --load tests/run-lifetime-stress.lisp --eval '(quit)'
for trial in 1 2 3; do
    TRIVIAL_SIMD_TEST_RUN=$((trial + 2)) arch -x86_64 "$ccl_dir/dx86cl64" --no-init --batch --load tests/run.lisp --eval '(quit)'
done
