#!/usr/bin/env bash
# WHAT: Compile Frigate's vendored MLX Metal shaders and install them where MLX looks.
# IN:   [debug|release] (default debug); --package DIR to target a consumer's .build.
# OUT:  mlx.metallib next to every binary in that .build, test bundles included.
# PIN:  `swift build` has NO Metal step — SwiftPM does not compile Sources/Cmlx's .metal
#       files, so a clean checkout has no metallib and MLX dies on the first GPU op with
#       "Failed to load the default metallib". Verified against a worktree at c06a125:
#       this is long-standing, not something the 2026-09 vendored update introduced.
#
#   ./scripts/build-metallib.sh                       # Frigate's own .build/debug
#   ./scripts/build-metallib.sh release
#   ./scripts/build-metallib.sh release --package ../Thread
#
# WHERE MLX ACTUALLY LOOKS. Sources/Cmlx/mlx/mlx/backend/metal/device.cpp,
# load_default_library(), probes five paths in order, all but the last relative to the
# directory of the RUNNING BINARY:
#
#   1. mlx.metallib                                  <- what this script installs
#   2. Resources/mlx.metallib
#   3. mlx-swift_Cmlx.bundle/…/default.metallib      (SWIFTPM_BUNDLE, Package.swift:214)
#   4. Resources/default.metallib                    <- also installed, belt and braces
#   5. default.metallib                              (relative to CWD, not the binary)
#
# Rung 1 is the one every consumer already relies on — Pelican's `ModelStore.metallibPresent`
# checks for exactly `mlx.metallib` beside the executable. Rung 4 is written too because
# METAL_PATH is defined as "default.metallib" and that name is what the failure message
# quotes, so having both removes the guesswork when someone is debugging a load failure.
#
# "Next to the binary" is not one directory. `swift test` runs out of a bundle
# (.build/<arch>/<config>/<Name>.xctest/Contents/MacOS/), which is why MLX-touching tests
# are gated behind FRIGATE_MLX_TESTS rather than run by default — so every executable and
# every .xctest bundle in the build directory gets its own copy.
set -euo pipefail

CONFIG="debug"
PACKAGE_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        debug|release) CONFIG="$1"; shift ;;
        --package) PACKAGE_DIR="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "build-metallib: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

FRIGATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_DIR="${PACKAGE_DIR:-$FRIGATE_ROOT}"

# A NON-DARWIN HOST IS A SKIP, NOT A FAILURE — same contract as check-linux-manifest.sh.
# Frigate builds on Linux (CUDA and CPU-only) where there is no Metal and no metallib to
# make; callers bake this into build pipelines that run on both.
if [ "$(uname -s)" != "Darwin" ]; then
    echo "build-metallib: not Darwin — no Metal backend, nothing to build."
    exit 0
fi

if ! xcrun -sdk macosx -f metal > /dev/null 2>&1; then
    echo "build-metallib: no Metal toolchain (xcrun -sdk macosx metal)." >&2
    echo "  Install the Xcode command line tools, or the Metal toolchain component." >&2
    exit 1
fi

METAL_SRC="$FRIGATE_ROOT/Sources/Cmlx/mlx-generated/metal"
if [ ! -d "$METAL_SRC" ]; then
    echo "build-metallib: no MLX metal shaders at $METAL_SRC" >&2
    exit 1
fi

if [ ! -d "$PACKAGE_DIR" ]; then
    echo "build-metallib: no package at $PACKAGE_DIR" >&2
    exit 1
fi
PACKAGE_DIR="$(cd "$PACKAGE_DIR" && pwd)"

# .build/<config> is a symlink to .build/<arch>-apple-macosx/<config>; resolve it so the
# copies land on the real directory rather than duplicating through the link.
BUILD_DIR="$PACKAGE_DIR/.build/$CONFIG"
if [ ! -d "$BUILD_DIR" ]; then
    echo "build-metallib: $BUILD_DIR does not exist — run 'swift build${CONFIG:+ -c $CONFIG}' first." >&2
    exit 1
fi
BUILD_DIR="$(cd "$BUILD_DIR" && pwd -P)"

MIN_MACOS="${MACOSX_DEPLOYMENT_TARGET:-15.0}"   # matches Package.swift's .macOS("15.0")
STAGE="$FRIGATE_ROOT/.build/metallib-$CONFIG"
mkdir -p "$STAGE"
METALLIB="$STAGE/mlx.metallib"

# Rebuild only when a shader is newer than the library we already have. Consumers wire
# this into every `swift build`, so the common case must be close to free.
if [ -f "$METALLIB" ] \
   && [ -z "$(find "$METAL_SRC" -name '*.metal' -newer "$METALLIB" -print -quit)" ]; then
    echo "build-metallib: $(basename "$METALLIB") is current"
else
    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMP_DIR"' EXIT

    AIR_FILES=()
    while IFS= read -r -d '' metal_file; do
        # Shaders live in subdirectories too (fft/, indexing/, reduction/, steel/), and
        # basenames repeat across them — mirror the relative path so nothing is clobbered.
        rel="${metal_file#"$METAL_SRC"/}"
        air_file="$TMP_DIR/${rel%.metal}.air"
        mkdir -p "$(dirname "$air_file")"
        xcrun -sdk macosx metal \
            -x metal \
            -fno-fast-math \
            -Wno-c++17-extensions \
            -Wno-c++20-extensions \
            -mmacosx-version-min="$MIN_MACOS" \
            -I "$METAL_SRC" \
            -c "$metal_file" \
            -o "$air_file"
        AIR_FILES+=("$air_file")
    done < <(find "$METAL_SRC" -name '*.metal' -print0)

    if [ ${#AIR_FILES[@]} -eq 0 ]; then
        echo "build-metallib: found no .metal files — refusing to write an empty library" >&2
        exit 1
    fi

    echo "build-metallib: linking ${#AIR_FILES[@]} shaders"
    xcrun -sdk macosx metallib "${AIR_FILES[@]}" -o "$METALLIB"
fi

# Install beside every binary that could load it: the build directory itself, and each
# .xctest bundle's Contents/MacOS. Both names, per the rung list in the header.
install_beside() {
    local dir="$1"
    mkdir -p "$dir/Resources"
    cp -f "$METALLIB" "$dir/mlx.metallib"
    cp -f "$METALLIB" "$dir/Resources/default.metallib"
    echo "  $dir"
}

echo "build-metallib: installing into $PACKAGE_DIR/.build/$CONFIG"
install_beside "$BUILD_DIR"

while IFS= read -r -d '' bundle; do
    macos_dir="$bundle/Contents/MacOS"
    [ -d "$macos_dir" ] && install_beside "$macos_dir"
done < <(find "$BUILD_DIR" -maxdepth 1 -name '*.xctest' -print0 2>/dev/null)

echo "build-metallib: done"
