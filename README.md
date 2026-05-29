# Frigate

A fully self-contained Swift package for on-device text embeddings and LLM inference using [MLX](https://github.com/ml-explore/mlx).

All fork sources are vendored directly — no external URLs for patched libraries appear in `Package.swift`. The only external dependencies are three standard Apple packages (`swift-numerics`, `swift-collections`, `swift-crypto`).

---

## What's inside

| API | Default model | Notes |
|---|---|---|
| `FrigateEmbedder` | `mlx-community/snowflake-arctic-embed-m-v1.5` | Returns `[[Float]]` |
| `FrigateLLM` | `mlx-community/Qwen3-0.6B-4bit` | Returns `AsyncStream<String>` |

Models are downloaded from HuggingFace Hub on first use and cached at `~/.cache/huggingface/`.

---

## Requirements

| | Version |
|---|---|
| Swift | 6.3+ |
| Ubuntu | 24.04 Noble (Linux) |
| CUDA | 12.x — GPU sm_86+ recommended (e.g. RTX 3090) |
| macOS | 14+ — Metal, no CUDA needed |

---

## Setup — Ubuntu 24.04 (fresh machine)

### Step 1 — Clone the repo

```bash
git clone <this-repo> Frigate
cd Frigate
```

### Step 2 — Run the setup script

```bash
./setup-frigate-ubuntu.sh
```

The script installs everything in order and then builds Frigate:

1. **Swift 6.3.2** via [swiftly](https://github.com/swiftlang/swiftly) — placed in `~/.local/share/swiftly/`
2. **CUDA 12.9** toolkit — adds the NVIDIA apt repo and installs `cuda-toolkit-12-9`
3. **BLAS / LAPACK / gfortran** — `libopenblas-dev`, `liblapacke-dev`
4. **cudnn-frontend v1.16.0** — clones and cmake-installs headers to `/usr/local/cudnn-frontend/`
5. **huggingface_hub** — `pip3 install huggingface_hub` for model downloads
6. **~/.bashrc** — exports `SWIFTLY_HOME`, CUDA paths, and `SPM_CUDA=1`
7. **`swift build -c release --jobs 2`** — compiles all targets (~20 min first time)

The script is idempotent — safe to re-run if any step failed.

**CPU-only** (no GPU required):
```bash
./setup-frigate-ubuntu.sh --cpu
```

**Install deps only, build later:**
```bash
./setup-frigate-ubuntu.sh --skip-build
```

### Step 3 — Download a model

```bash
# Embedding model (~450 MB)
huggingface-cli download mlx-community/snowflake-arctic-embed-m-v1.5

# LLM (~400 MB)
huggingface-cli download mlx-community/Qwen3-0.6B-4bit
```

HuggingFace Hub caches models at `~/.cache/huggingface/hub/`. Downloads happen automatically at first use if you skip this step.

### Step 4 — Use from Swift

Add Frigate as a local package dependency in your `Package.swift`:

```swift
.package(path: "/path/to/Frigate"),
```

Then import and use:

```swift
import Frigate

// Embeddings
let embedder = FrigateEmbedder()
let vectors: [[Float]] = try await embedder.embed([
    "hello world",
    "machine learning on GPU",
])

// LLM
let llm = FrigateLLM()
for await token in try await llm.generate(prompt: "Explain MLX in one sentence.") {
    print(token, terminator: "")
}
```

### Manual build (after setup script, without --skip-build)

```bash
# CUDA bin must be in PATH — the system /usr/bin/nvcc stub does not include CUDA headers
source ~/.bashrc   # loads SWIFTLY_HOME, CUDA PATH, SPM_CUDA=1
swift build -c release --jobs 2
```

Or inline:
```bash
PATH="/usr/local/cuda/bin:$PATH" SPM_CUDA=1 CUDA_ARCH=sm_86 swift build -c release --jobs 2
```

---

## Setup — macOS

```bash
git clone <this-repo> Frigate
cd Frigate
swift build -c release
```

No additional setup needed. MLX uses Metal automatically. Swift 6.3+ required (`brew install swiftly && swiftly install latest`).

---

## API reference

### FrigateEmbedder

```swift
public actor FrigateEmbedder {
    public init(modelId: String = "mlx-community/snowflake-arctic-embed-m-v1.5")
    public func embed(_ texts: [String]) async throws -> [[Float]]
    public func warmup() async throws
}
```

### FrigateLLM

```swift
public actor FrigateLLM {
    public init(modelId: String = "mlx-community/Qwen3-0.6B-4bit")
    public func generate(prompt: String, maxTokens: Int = 512) async throws -> AsyncStream<String>
    public func warmup() async throws
}
```

Both actors deduplicate concurrent model loads — calling `embed` or `generate` from multiple tasks concurrently is safe.

---

## Vendored sources

All fork sources are copied directly into `Sources/`. No git submodules, no external URLs for patched code.

| Directory | From |
|---|---|
| `Sources/Cmlx/` | `riteshpakala/mlx` @ `gab/cuda1` — C++ MLX with CUDA sm_86 patches |
| `Sources/MLX/` … `Sources/MLXLinalg/` | `riteshpakala/mlx-swift` @ `gab/cuda1` |
| `Sources/Jinja/` | `huggingface/swift-jinja` |
| `Sources/Hub/` … `Sources/Models/` | `riteshpakala/swift-transformers` |
| `Sources/MLXLMCommon/` … `Sources/MLXEmbedders/` | `riteshpakala/mlx-swift-lm` |
| `Sources/mlx_embeddings/` | `riteshpakala/mlx.embeddings` |
| `Sources/Frigate/` | This package — `FrigateEmbedder`, `FrigateLLM` |

---

## Known GPU constraints

**`container.perform` is a pure inference zone.**
Never call `MLX.Memory.*`, `Stream.*`, or any `CommandEncoder` API from inside a `container.perform` closure. The CUDA allocator is active during the closure; re-entry causes SIGSEGV (address ~0x6529) or crash at `cudaGraphLaunch`. All memory management runs after `perform` returns.

**SDPA cache size.**
`MLX_CUDA_SDPA_CACHE_SIZE=2048` is set in `FrigateEmbedder.init`. The default of 256 triggers a fatal error after 512 cache misses when sequence lengths vary across sub-batches.

**Batch and token limits.**
`FrigateEmbedder` uses 8 inputs per sub-batch and caps sequences at 512 tokens. Larger values cause `cudaMallocAsync` OOM on 24 GB cards because encoder temporary buffers accumulate until `CommandEncoder::commit()` fires.

**GPU architecture.**
Default is `sm_86` (RTX 3090). Override before building: `export CUDA_ARCH=sm_89` for RTX 4090. CUTLASS is disabled; GPU fallback uses `affine_dequantize + CublasGemm` (works on any sm_80+ without CUTLASS).

---

## TODO — MLXVLM Linux port

Vision-language models (Gemma3, Qwen2-VL, Qwen3-VL, PaliGemma, Pixtral, SmolVLM2, etc.) are fully implemented in `Sources/MLXVLM/` but **excluded on Linux** because they depend on Apple-only frameworks: `AVFoundation`, `CoreImage`, `CoreGraphics`.

The exclusion is in `Package.swift` via `vlmExcludes` — removing those excludes and providing Linux-compatible replacements is all that is needed to unlock VLM on Linux.

### What is blocked and why

| File | Dependency | Used for |
|---|---|---|
| `MediaProcessing.swift` | `AVFoundation`, `CoreImage` | Image resize, pixel buffer extraction, video frame decoding |
| `Models/Qwen2VL.swift` and similar | `CoreGraphics` (`CGSize`, `CGFloat`) | Bounding-box coordinates in vision encoders |
| `Models/Paligemma.swift` etc. | `CoreImage.CIFilterBuiltins` | Image preprocessing (normalise, crop) |
| `VLMModelFactory.swift` | Depends on all model types above | Registers all VLM model constructors |

`Sources/MLXLMCommon/LinuxCompat.swift` already provides `CGSize` and `CGFloat` stubs, so coordinate types compile. The primary blocker is image I/O and pixel manipulation in `MediaProcessing.swift`.

### Prescribed path to completion

**1. Replace `MediaProcessing.swift` with a Linux-compatible image backend.**

The file needs to:
- Load an image from a file path or `Data` blob into a float tensor (`MLXArray` of shape `[H, W, 3]`)
- Resize to a target `CGSize`
- Normalise pixel values (mean/std per channel)
- Return an `MLXArray` directly (no `CIImage`, no `CGImage`)

Candidate backends (add as a vendored source or SPM dependency):
- **`stb_image`** (C, single-header) — simplest, handles JPEG/PNG/BMP, link via a small `Sources/CStbImage/` C target
- **`libjpeg-turbo` + `libpng`** via system libraries (`apt install libjpeg-turbo8-dev libpng-dev`) — more deps but battle-tested
- **`swift-image`** or **`Swim`** — pure Swift, no system deps, covers common formats

On Apple platforms keep the existing `CoreImage` path using `#if canImport(CoreImage)`.

**2. Audit each model file for remaining Apple API calls.**

After MediaProcessing is replaced, compile with:
```bash
PATH="/usr/local/cuda/bin:$PATH" SPM_CUDA=1 swift build -c release --jobs 2 2>&1 | grep "error:"
```

Known locations to check:
- `Models/Qwen2VL.swift`, `Qwen3VL.swift` — use `CGSize` for patch grid calculations (LinuxCompat stub should cover these)
- `Models/FastVLM.swift` — uses `CoreGraphics` for tile sizing
- `Models/Gemma3.swift` — `CoreImage` for image normalisation

Wrap any remaining calls with `#if canImport(CoreImage) ... #else ... #endif`.

**3. Remove the Linux excludes from `Package.swift`.**

```swift
// Before
#if os(Linux)
let vlmExcludes: [String] = ["README.md", "MediaProcessing.swift", "Models", "VLMModelFactory.swift"]
#else
let vlmExcludes: [String] = ["README.md"]
#endif

// After (once Linux-compatible image backend exists)
let vlmExcludes: [String] = ["README.md"]
```

**4. Expose `FrigateVLM` in `Sources/Frigate/`.**

```swift
public actor FrigateVLM {
    public init(modelId: String = "mlx-community/Qwen2-VL-2B-Instruct-4bit")
    public func generate(prompt: String, imageData: Data, maxTokens: Int = 512) async throws -> AsyncStream<String>
    public func warmup() async throws
}
```

Wire it to `VLMModelFactory.shared.loadContainer(configuration:)` following the same pattern as `FrigateLLM`, ensuring all MLX.Memory.* calls stay outside `container.perform`.

**5. Add `MLXVLM` to the `Frigate` target's dependencies in `Package.swift`.**

```swift
.target(
    name: "Frigate",
    dependencies: [
        "MLX", "MLXNN", "Tokenizers",
        "MLXLMCommon", "MLXLLM", "MLXVLM", "mlx_embeddings",  // add MLXVLM
    ],
    ...
)
```

### Estimated scope

| Task | Effort |
|---|---|
| Implement `MediaProcessing.swift` Linux backend with `stb_image` | ~2–4 hours |
| Fix remaining `CoreGraphics` / `CoreImage` calls in model files | ~1–2 hours |
| Write `FrigateVLM` actor | ~1 hour |
| Integration test with Qwen2-VL-2B on RTX 3090 | ~1 hour |
