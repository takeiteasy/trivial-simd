#!/bin/sh
# Run the C and Lisp test suites on x86-64 SBCL under Rosetta from an ARM64 Mac.
# Extra arguments are passed to SBCL after the suite is loaded.
set -eu

version=${SBCL_X86_VERSION:-2.6.8}
cache=${TRIVIAL_SIMD_X86_CACHE:-$HOME/.cache/trivial-simd/x86-sbcl}
root=$(cd "$(dirname "$0")/.." && pwd)
name=sbcl-$version-x86-64-darwin
sbcl_dir=$cache/$name

[ "$(uname -s)" = Darwin ] || { echo "x86-sbcl.sh needs macOS with Rosetta" >&2; exit 1; }
arch -x86_64 /usr/bin/true 2>/dev/null || { echo "Rosetta is not installed" >&2; exit 1; }

if [ ! -x "$sbcl_dir/src/runtime/sbcl" ]; then
    mkdir -p "$cache"
    curl -fsSL -o "$cache/$name.tar.bz2" \
        "https://github.com/roswell/sbcl_bin/releases/download/$version/$name-binary.tar.bz2"
    tar xjf "$cache/$name.tar.bz2" -C "$cache"
fi

# The Lisp loader expects build/, so the x86-64 build lives in a copy of the tree.
work=$cache/work
mkdir -p "$work/build"
rsync -a --delete --exclude build --exclude .git "$root/" "$work/"
(cd "$work/build" \
    && cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 .. > /dev/null \
    && cmake --build . > /dev/null \
    && ctest --output-on-failure)

cd "$work"
# Contrib fasls (asdf, sb-simd) sit under obj/sbcl-home in this tarball.
SBCL_HOME=$sbcl_dir/obj/sbcl-home exec arch -x86_64 "$sbcl_dir/src/runtime/sbcl" \
    --dynamic-space-size 4096 --core "$sbcl_dir/output/sbcl.core" --noinform --no-sysinit --no-userinit \
    --non-interactive --load tests/run.lisp "$@"
