# Frigate

A fully self-contained Swift package for on-device text embeddings and LLM inference using [MLX](https://github.com/ml-explore/mlx).

All fork sources are vendored directly — no external URLs for patched libraries appear in `Package.swift`. The only external dependencies are three standard Apple packages (`swift-numerics`, `swift-collections`, `swift-crypto`).

---

## What's inside

| API | Default model | Notes |
|---|---|---|
| `FrigateEmbedder` | `mlx-community/snowflake-arctic-embed-m-v1.5` | Returns `[[Float]]` |
| `FrigateLLM` | `mlx-community/Qwen3-0.6B-4bit` | Returns `AsyncStream<String>` |
| `FrigateBoost` | local `.json` file | XGBoost tree-ensemble inference, zero runtime deps |
| `FrigateVision` | bundled ONNX region classifier | Pixel perception, re-exported from [VisionAX](../VisionAX) — **macOS only** |

HuggingFace models are downloaded on first use and cached at `~/.cache/huggingface/`. `FrigateBoost` loads a model exported with `booster.save_model("model.json")` — no `libxgboost` required at runtime.

---

## FrigateVision — pixel perception

`import FrigateVision` gives you VisionAX whole: the OpenCV region detector, Apple's text
recognition, the ONNX role classifier and the page map.

```swift
import FrigateVision

let engine = try VisionEngine()
let scene = try engine.perceive(
    image: capture,
    projection: ScreenProjection(origin: pageOrigin, pixelsPerPoint: 2),
    classifier: try? RegionClassifier.bundled(),
    lanes: [.regions, .text])
let map = scene.pageMap()          // rows, what each affords, where its name came from
```

It is a **re-export**, not a copy — VisionAX keeps its own repository, bench, harvester and
training pipeline, and this package is the door consumers walk through.

Three things worth knowing:

- **macOS only.** VisionAX brings OpenCV and ONNX Runtime as xcframeworks, which Linux
  cannot resolve, so the dependency, the target and the product all live inside
  `#if !os(Linux)` in `Package.swift`. A Linux build of Frigate never learns they exist,
  and `../VisionAX` need not be checked out on a Linux box. `ManifestPlatformTests` fails
  on macOS the moment something escapes that guard, so the Linux box is never the first to
  find out.
- **It is not in the `Frigate` umbrella.** `import Frigate` does not carry it, deliberately:
  VisionAX declares `AXNodeSnapshot`, `AXScreenElement` and `AXNodeCategory` under the same
  names an accessibility layer already uses, and every consumer of the umbrella would
  inherit that ambiguity at the use site.
- **It costs no MLX.** The wrapper depends on the VisionAX product alone, so taking
  `FrigateVision` links none of the inference stack.

Two consequences of the path dependency, both benign:

- On macOS, every consumer's `Package.resolved` gains an `opencv-spm` pin and their next
  resolve fetches the OpenCV and ONNX Runtime artifacts, whether or not they link
  `FrigateVision`.
- Frigate's own `Package.resolved` differs by host (macOS has `opencv-spm`; Linux does not).
  A plain `swift build` on Linux drops the extra pin with a warning and rewrites the file —
  which is why the Linux scripts do not pass `--force-resolved-versions`.

Because Frigate now carries a path dependency, **a consumer must take Frigate by path too**
(`.package(path: "../Frigate")`): SwiftPM refuses a package required by URL that itself
depends on a local package.

## Requirements

| | Version |
|---|---|
| Swift | 6.3+ |
| Ubuntu | 24.04 Noble (Linux) |
| CUDA | 12.x — GPU sm_86+ recommended (e.g. RTX 3090) |
| macOS | 15+ — Metal, no CUDA needed |

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
hf download mlx-community/snowflake-arctic-embed-m-v1.5

# LLM (~400 MB)
hf download mlx-community/Qwen3-0.6B-4bit
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

// XGBoost — load model, predict P(class=1) for a batch of feature vectors
let boost = try FrigateBoost(modelURL: URL(fileURLWithPath: "model.json"))
let probs: [Float] = await boost.predict(features: [
    [0.12, 0.003, 0.47, 0.06, 0.31, 0.84, 0.002, 0.001, 0.51, 0.29],
])
print(probs) // e.g. [0.731]

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
./scripts/build-metallib.sh release     # required — see below
```

Swift 6.3+ required (`brew install swiftly && swiftly install latest`).

**The metallib step is not optional on macOS.** `swift build` has no Metal stage — SwiftPM
never compiles `Sources/Cmlx/mlx-generated/metal/*.metal` — so a fresh checkout has no
`mlx.metallib` and the first GPU op dies with *"Failed to load the default metallib"*. Nothing
warns you at build time; it looks fine until inference runs. `scripts/build-metallib.sh`
compiles the shaders and installs the result beside every binary in `.build`, including inside
`.xctest` bundles, which is what lets the `FRIGATE_MLX_TESTS=1` suites run at all.

It is a no-op on Linux (no Metal backend), so it is safe to put in a build script that runs on
both. Consumers should call it rather than keeping their own copy — `Bonnie`, `Fleet`, `Sewn`,
`Thread` and `Zehn` each ship a one-line `build-metallib.sh` that delegates here:

```bash
./scripts/build-metallib.sh release --package ../Thread   # or, from Thread: ./build-metallib.sh release
```

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

### FrigateBoost

```swift
public actor FrigateBoost {
    /// Load an XGBoost model exported with `booster.save_model("model.json")`.
    public init(modelURL: URL) throws
    /// Predict P(class=1) for a batch of feature vectors.
    public func predict(features: [[Float]]) async -> [Float]
}
```

`FrigateBoost` parses the XGBoost v2 JSON format in pure Swift and walks the tree ensemble directly — no `libxgboost` binary required. The `binary:logistic` objective is supported; leaf values are summed across all trees and sigmoid is applied to produce final probabilities.

---

## Vendored sources

All fork sources are copied directly into `Sources/`. No git submodules, no external URLs for patched code.

Record the **tag**, not a branch — the absence of that record is what made the
2026-09 audit necessary. Last synced 2026-09-08.

| Directory | From | Version | Local patches |
|---|---|---|---|
| `Sources/Cmlx/mlx/` | `ml-explore/mlx` | **v0.31.1** | 4 files in `backend/cuda/` + 3 added `no_*_impl.cpp` (CUTLASS-disabled fallbacks, sm_86) |
| `Sources/Cmlx/mlx-c/` | `ml-explore/mlx-c` | v0.6.0 | none |
| `Sources/Cmlx/{json,fmt,metal-cpp}/` | pinned by mlx-swift | json 3.11.3, fmt 12.1.0 | none — these are mlx's own pins, **not** drift |
| `Sources/MLX/` … `Sources/MLXOptimizers/` | `ml-explore/mlx-swift` | **0.31.6** | none |
| `Sources/Jinja/` | `huggingface/swift-jinja` | **2.5.0** | none |
| `Sources/Hub/` … `Sources/Models/` | `huggingface/swift-transformers` | **1.3.4** | upstream adopted the fork's Linux patches; only `LocalizedStringLinuxShim.swift` added |
| `Sources/HuggingFace/` | `huggingface/swift-huggingface` | **0.10.1** | none (Xet trait off) |
| `Sources/EventSource/` | `mattt/EventSource` | **1.5.1** | `EventSource+AsyncHTTPClient.swift` dropped (AsyncHTTPClient trait off — keeps SwiftNIO out) |
| `Sources/MLXLMCommon/` … `Sources/MLXEmbedders/` | `ml-explore/mlx-swift-lm` | **3.31.4** | `LinuxCompat.swift` added; Linux `canImport` guards on `UserInput` (incl. `AudioProcessing.audioFormat`), `UserInput+Audio`, `ChatSession`, `ParoQuant/ParoQuantLoader`; `MLXEmbedders/Models/Bert.swift` sanitize chain made table-driven |
| `Sources/mlx_embeddings/` | `mzbac/mlx.embeddings` | v0.1.3 | `Bert.swift` |
| `Sources/FrigateBridge/` | This package — concrete `Downloader` / `TokenizerLoader` for mlx-swift-lm 3.x |  |  |
| `Sources/MLXAccelerate/` | This package — Linux-compatible Accelerate ops via MLX (`gaussianBlur`, `sobelGradients`, `filter2D`, `perspectiveWarp`, `spectralDistance`) |  |  |
| `Sources/Frigate/` | This package — `FrigateEmbedder`, `FrigateLLM`, `FrigateBoost` |  |  |

**mlx C++ is deliberately held at 0.31.1.** v0.32.2 exists, but every mlx-swift release through
0.31.6 defines `MLX_VERSION` as `"0.31.1"` (see `Package.swift`), so bumping the C++ core would
outrun the `mlx-c` / Swift bindings. Revisit when mlx-swift 0.32.x ships.

**mlx-swift-lm 3.x takes the downloader and tokenizer as parameters.** Upstream supplies that
wiring through its `MLXHuggingFace` macro target, which needs swift-syntax; this package wires
it by hand in `Sources/FrigateBridge/` instead, so no compiler plugin enters the Linux build.

**yyjson is an external dependency, not vendored — deliberately.** swift-transformers 1.3.x needs
it, and WhisperKit brings its own swift-transformers into the same link in at least one consumer
(Bonnie). Two copies of `yyjson.c` define the same C symbols and nothing in SwiftPM separates
them: module aliasing explicitly does not cover non-Swift symbols, renaming the target does not
rename its symbols, and even hidden (`private external`) visibility still trips ld64's duplicate
check — all three were tried. Depending on the same package URL makes SwiftPM dedupe by identity,
so every graph gets exactly one copy. The vendoring rule above is about *patched fork* sources;
this is stock upstream C at the pin swift-transformers itself uses.

**`String(localized:)` does not exist on Linux.** swift-corelibs-foundation ships no `localized:`
overload (checked on Swift 6.3 / Ubuntu 24.04, with and without `FoundationEssentials`), and
swift-transformers 1.3.x and mlx-swift-lm 3.x both use it for error text. Each affected vendored
module carries an internal `LocalizedStringLinuxShim.swift` that returns the already-interpolated
literal: `Hub`, `Tokenizers`, `Models`, `MLXLMCommon`, `MLXLLM`, `MLXVLM`.

**Three targets leave the Linux graph entirely**, alongside `FrigateVision`: `ObscurKit`,
`FluxKit` and `flux2-cli`. `ObscurDinov2` decodes a `CGImage` through CGContext/CGColorSpace and
the other two follow it in; no consumer takes them. See `imageGenTargets` in `Package.swift`.

**The CPU-only Linux build (`SPM_CUDA=0`) was broken before the 2026-09 audit** and is now fixed.
Its exclude lists had drifted: `GPU+CUDA.swift` (a portable stub that is the only definition of
`GPU`), `MLXFast.swift`/`MLXFastKernel.swift` (which the `MLXFast` shim target dereferences), and
`mlx-c/mlx/c/fast.cpp` (leaving `mlx_fast_*` undefined — invisible to a library build, caught only
when Thread linked an executable). All four are now built. `MLXAccelerate/BatchedImageOps.swift`
also needed `nonisolated(unsafe)` on two pointers, because Linux's libdispatch declares
`concurrentPerform`'s closure `@Sendable` where Darwin's does not.

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
let vlmExcludes: [String] = [
    "README.md", "MediaProcessing.swift", "Models", "VLMModelFactory.swift",
    "Gemma4AssistantRegistration.swift",
]
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
