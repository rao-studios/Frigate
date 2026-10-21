#!/bin/bash
# WHAT: Build and run the tests with the MLX-gated suites included (FRIGATE_MLX_TESTS=1).
# IN:   Any `swift test` arguments, e.g. `--filter SinatraHarnessTests`.
# PIN:  A test bundle is code-signed at build time, and a metallib copied into it before the
#       build breaks the seal. So every bundle's copy is removed first, the tests are built,
#       build-metallib.sh installs a fresh copy beside each binary and inside each bundle, and
#       the tests run from the already-built bundles. Came over from the SinatraHarness
#       repository with the harness (2026-10).
#
#   ./scripts/test-mlx.sh
#   ./scripts/test-mlx.sh --filter SinatraHarnessTests
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for bundle in .build/debug/*.xctest; do
  [ -d "$bundle" ] || continue
  rm -f "$bundle/Contents/MacOS/mlx.metallib"
  rm -rf "$bundle/Contents/MacOS/Resources"
done
swift build --build-tests
"$ROOT/scripts/build-metallib.sh" debug > /dev/null
FRIGATE_MLX_TESTS=1 swift test --skip-build "$@"
