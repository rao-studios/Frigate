#!/usr/bin/env bash
# Setup and build Frigate on Ubuntu 24.04 (Noble) with MLX CUDA backend.
#
# Run this script on a fresh machine before anything else:
#   ./setup-frigate-ubuntu.sh          # CUDA GPU build (default)
#   ./setup-frigate-ubuntu.sh --cpu    # CPU-only build (no GPU required)
#   ./setup-frigate-ubuntu.sh --skip-build  # Install deps only, don't build
#   ./setup-frigate-ubuntu.sh --debug  # Debug build

set -euo pipefail

CUDA_ENABLED=1
BUILD_CONFIG="release"
SKIP_BUILD=0

for arg in "$@"; do
    case $arg in
        --cpu)        CUDA_ENABLED=0 ;;
        --debug)      BUILD_CONFIG="debug" ;;
        --skip-build) SKIP_BUILD=1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "==> $*"; }

# ── 1. Swift via swiftly ───────────────────────────────────────────────────────

log "Checking Swift..."
NEED_SWIFT=1
if command -v swift &>/dev/null; then
    SWIFT_VER=$(swift --version 2>&1 | grep -oP 'Swift version \K[0-9]+\.[0-9]+' || true)
    MAJOR=$(echo "$SWIFT_VER" | cut -d. -f1)
    MINOR=$(echo "$SWIFT_VER" | cut -d. -f2)
    if [[ $MAJOR -gt 6 || ($MAJOR -eq 6 && $MINOR -ge 3) ]]; then
        log "Swift $SWIFT_VER already installed."
        NEED_SWIFT=0
    fi
fi

if [[ $NEED_SWIFT -eq 1 ]]; then
    log "Installing Swift 6.3.2 via swiftly..."
    curl -L https://swiftlang.github.io/swiftly/swiftly-install.sh | bash -s -- --assume-yes
    export SWIFTLY_HOME="${SWIFTLY_HOME:-$HOME/.local/share/swiftly}"
    export PATH="$SWIFTLY_HOME/bin:$PATH"
    source "$SWIFTLY_HOME/env.sh" 2>/dev/null || true
    swiftly install 6.3.2
    swiftly use 6.3.2
fi

# Ensure swiftly paths are sourced for the rest of the script
if [[ -f "${SWIFTLY_HOME:-$HOME/.local/share/swiftly}/env.sh" ]]; then
    source "${SWIFTLY_HOME:-$HOME/.local/share/swiftly}/env.sh"
fi

# ── 2. System packages ────────────────────────────────────────────────────────

log "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    liblapacke-dev \
    libopenblas-dev \
    gfortran \
    python3-pip \
    libncurses-dev \
    libssl-dev \
    pkg-config

# ── 3. CUDA 12.9 (GPU only) ───────────────────────────────────────────────────

if [[ $CUDA_ENABLED -eq 1 ]]; then
    if [[ -d /usr/local/cuda ]]; then
        log "CUDA already installed at /usr/local/cuda."
    else
        log "Installing CUDA 12.9..."
        KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/$KEYRING_DEB"
        sudo dpkg -i "$KEYRING_DEB"
        rm -f "$KEYRING_DEB"
        sudo apt-get update -qq
        sudo apt-get install -y cuda-toolkit-12-9
    fi

    export PATH="/usr/local/cuda/bin:${PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    export CUDA_ARCH="${CUDA_ARCH:-sm_86}"
fi

# ── 4. cudnn-frontend v1.16.0 (GPU only) ─────────────────────────────────────

if [[ $CUDA_ENABLED -eq 1 ]]; then
    if [[ -d /usr/local/cudnn-frontend ]]; then
        log "cudnn-frontend already installed."
    else
        log "Installing cudnn-frontend v1.16.0..."
        TMPDIR=$(mktemp -d)
        git clone --depth 1 --branch v1.16.0 \
            https://github.com/NVIDIA/cudnn-frontend.git "$TMPDIR/cudnn-frontend"
        sudo cmake \
            -DCUDNN_FRONTEND_BUILD_SAMPLES=OFF \
            -DCUDNN_FRONTEND_BUILD_UNIT_TESTS=OFF \
            -DCMAKE_INSTALL_PREFIX=/usr/local/cudnn-frontend \
            -S "$TMPDIR/cudnn-frontend" -B "$TMPDIR/build"
        sudo cmake --build "$TMPDIR/build" --target install
        rm -rf "$TMPDIR"
    fi
fi

# ── 5. huggingface_hub (model downloads) ─────────────────────────────────────

log "Installing huggingface_hub..."
pip3 install --quiet --upgrade huggingface_hub

# ── 6. Shell environment ──────────────────────────────────────────────────────

log "Updating ~/.bashrc with required env vars..."

add_to_bashrc() {
    local line="$1"
    grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
}

SWIFTLY_HOME_DIR="${SWIFTLY_HOME:-$HOME/.local/share/swiftly}"
add_to_bashrc "export SWIFTLY_HOME=\"$SWIFTLY_HOME_DIR\""
add_to_bashrc "export PATH=\"\$SWIFTLY_HOME/bin:\$PATH\""
add_to_bashrc "[ -f \"\$SWIFTLY_HOME/env.sh\" ] && source \"\$SWIFTLY_HOME/env.sh\""

if [[ $CUDA_ENABLED -eq 1 ]]; then
    add_to_bashrc 'export PATH="/usr/local/cuda/bin:$PATH"'
    add_to_bashrc 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"'
    add_to_bashrc 'export SPM_CUDA=1'
fi

# ── 7. Build ──────────────────────────────────────────────────────────────────

if [[ $SKIP_BUILD -eq 1 ]]; then
    log "Skipping build (--skip-build). Run manually when ready."
else
    log "Building Frigate ($BUILD_CONFIG)..."
    cd "$SCRIPT_DIR"

    if [[ $CUDA_ENABLED -eq 1 ]]; then
        export SPM_CUDA=1
        export CUDA_ARCH="${CUDA_ARCH:-sm_86}"
    else
        export SPM_CUDA=0
    fi

    swift build -c "$BUILD_CONFIG" --jobs 2

    log ""
    log "Build complete."
fi

# ── 8. Next steps ─────────────────────────────────────────────────────────────

cat <<'EOF'

┌─────────────────────────────────────────────────────────────────┐
│  Frigate is ready.                                              │
│                                                                 │
│  Download a model (example):                                    │
│    huggingface-cli download mlx-community/snowflake-arctic-embed-m-v1.5
│                                                                 │
│  Use in Swift:                                                  │
│    import Frigate                                               │
│                                                                 │
│    let embedder = FrigateEmbedder()                             │
│    let vecs = try await embedder.embed(["hello world"])         │
│                                                                 │
│    let llm = FrigateLLM()                                       │
│    for await token in try await llm.generate(prompt: "Hi") {   │
│        print(token, terminator: "")                             │
│    }                                                            │
└─────────────────────────────────────────────────────────────────┘

EOF
