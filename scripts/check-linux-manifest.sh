#!/usr/bin/env bash
# WHAT: Evaluate Package.swift the way Linux will, without a Linux box.
# OUT:  the manifest's own JSON, checked for anything macOS-only
# PIN:  `#if os(Linux)` is decided where the manifest COMPILES, so a macOS test can only
#       read the guard as text (ManifestPlatformTests does that). This runs the real
#       thing under Linux in a container and asserts the vision graph is absent — the
#       check Totem's CUDA build would otherwise be the first to make.
#
#   ./scripts/check-linux-manifest.sh
#
set -euo pipefail

# A MISSING DOCKER IS A SKIP, NOT A FAILURE. This check is the belt to
# ManifestPlatformTests' braces: that one reads the guard as text on any machine, this
# one evaluates it for real. An installed CLI with no daemon behind it is the same
# situation as no CLI at all, and the first version of this script died on it.
if ! command -v docker > /dev/null || ! docker info > /dev/null 2>&1; then
    echo "no running docker daemon — skipping the Linux manifest check."
    echo "ManifestPlatformTests still reads the guard as text on this machine."
    exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${SWIFT_IMAGE:-swift:6.1}"

echo "==> swift package dump-package under $IMAGE"
if ! JSON=$(docker run --rm -v "$ROOT":/pkg -w /pkg "$IMAGE" \
        swift package dump-package 2>&1); then
    echo "the container could not evaluate the manifest:"
    echo "$JSON" | tail -5
    exit 1
fi

fail=0
for token in VisionAX FrigateVision opencv onnxruntime; do
    if grep -qi "$token" <<< "$JSON"; then
        echo "  ✗ the Linux manifest names $token"
        fail=1
    else
        echo "  ✓ no $token on Linux"
    fi
done

if [[ $fail -eq 1 ]]; then
    echo ""
    echo "Something macOS-only escaped the #if !os(Linux) guard in Package.swift."
    exit 1
fi
echo ""
echo "The Linux manifest carries no macOS-only dependency."
