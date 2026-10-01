#!/bin/sh
# Run the C and Lisp test suites on x86-64 ECL under Rosetta from an ARM64 Mac.
# Extra arguments are passed to ECL after the suite is loaded.
set -eu

version=${ECL_X86_VERSION:-26.5.5}
cache=${TRIVIAL_SIMD_X86_ECL_CACHE:-$HOME/.cache/trivial-simd/x86-ecl}
root=$(cd "$(dirname "$0")/.." && pwd)
prefix=$cache/ecl-$version-install

[ "$(uname -s)" = Darwin ] || { echo "x86-ecl.sh needs macOS with Rosetta" >&2; exit 1; }
arch -x86_64 /usr/bin/true 2>/dev/null || { echo "Rosetta is not installed" >&2; exit 1; }

# ECL builds its own gmp, libgc, and libffi when the system has no x86-64 copies.
if [ ! -x "$prefix/bin/ecl" ]; then
    mkdir -p "$cache"
    curl -fsSL -o "$cache/ecl-$version.tgz" \
        "https://ecl.common-lisp.dev/static/files/release/ecl-$version.tgz"
    tar xzf "$cache/ecl-$version.tgz" -C "$cache"
    (cd "$cache/ecl-$version" \
        && arch -x86_64 ./configure --prefix="$prefix" > "$cache/configure.log" 2>&1 \
        && arch -x86_64 /usr/bin/make -j"$(sysctl -n hw.ncpu)" > "$cache/make.log" 2>&1 \
        && arch -x86_64 /usr/bin/make install > "$cache/install.log" 2>&1) \
        || { echo "ECL build failed; see logs in $cache" >&2; exit 1; }
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
exec arch -x86_64 "$prefix/bin/ecl" --load tests/run.lisp --eval '(quit)' "$@"
