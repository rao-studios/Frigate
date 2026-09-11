# VisionAX — the vision runtime

Frigate's **`FrigateVisionAX`** module (vended by the product of the same name) is the
pixel-perception runtime. Canny edge detection over a screenshot produces a top-down,
traversable tree of bounding boxes in **exactly the shape of Mary's AXTree**; an on-device
classifier names each box with an AX role; the page map turns that into rows a consumer
acts on.

What is not runtime lives in the [VisionAX repository](../../../VisionAX):

- **`VisionAXCore`** — the AX/A11Y data structures (`AXNodeSnapshot`, `AXWindowSnapshot`,
  `AXScreenElement`, `AXNodeCategory`, `AXTreeJSON`), the dataset schema and the role
  vocabulary. This module re-exports it, so `import FrigateVisionAX` carries those types.
- **The training pipeline** (`Training/`) that produces `Resources/Models` here.
- **The tools** (`Tools/`): the harvester that builds the training set and the bench that
  tunes the detector against it.

## Layout

| Where | What it is |
|---|---|
| `Sources/CVisionAX` | The engine. `include/visionax.h` is a pure-C header — the only public surface. Everything behind it is C++: `Engine` (the extensible engine class), `CannyRegionDetector`, `RegionTreeBuilder`, `Classifier`, the media and icon glyph banks, and `visionax.cpp`, the `extern "C"` facade. |
| `Sources/FrigateVisionAX` | The Swift face: `VisionEngine` (`perceive` → `VisionScene`), `RegionClassifier`, the page map, the media lane, text recognition. |
| `Sources/FrigateVisionAX/Resources/Models` | The bundled classifier (git-lfs), exported by VisionAX's training pipeline. |
| `Tests/FrigateVisionAXTests` | The engine's tests: synthetic screens, the tiny arithmetic classifier, real captures. |

OpenCV 4 and 5 removed the legacy C API, so "a C engine" here means **C++ internals behind
an `extern "C"` header** — the public header stays C-clean, and Swift imports it as an
ordinary module. Inference runs on ONNX Runtime's C API inside the same engine, so one
runtime serves every model that follows and Swift never sees a tensor.

The whole target is macOS-only and sits inside Frigate's `#if !os(Linux)` block: OpenCV and
ONNX Runtime arrive as xcframeworks Linux cannot resolve.

## How a region becomes a node

1. **`CannyRegionDetector`** — to gray, optional Gaussian blur, `Canny`, optional
   morphological close (seals one-pixel gaps so a box's outline reads as one loop),
   `findContours`, `boundingRect` per contour.
   `findContours`' own hierarchy is *not* used: on an edge map it describes edge
   loops, not UI containment.
2. **Filter and dedup** — drop anything under `min_width`/`min_height`, then drop a
   box that duplicates a larger kept one, by IoU or by all four edges landing within
   `merge_slack`. A stroked rectangle yields an outer and an inner contour; without
   this every box on screen would arrive doubled.
3. **`RegionTreeBuilder`** — a box's parent is the **smallest** kept box that contains
   it (within `containment_slack`); partial overlaps become siblings, never a parent.
   Siblings are sorted by Mary's roster reading rule: a row band by `midY`, then left
   to right. The tree is flattened pre-order under a walk budget with Mary's
   `AXTreeWalker` semantics — a node **at** `max_depth` is kept and its children
   withheld, and exactly `max_nodes` nodes are emitted before the cut. Either sets
   `isTruncated`.

Every knob is exposed in `CannyOptions` (the value lives in VisionAXCore, because every
dataset sample records the options it was proposed with; `.standard` is read from the C
defaults here) and on the bench's parameter strip, because defaults tuned on synthetic
rectangles over-segment real screenshots — text especially.

For reference, one 1920×1080 desktop capture at the defaults: 2,377 contours in,
978 nodes out, tree depth 5, about 40 ms, ~400 KB of JSON.

## The classifier

`VisionEngine.detectAndClassify(in:title:using:)` runs Canny, then names every region.
A model is two ONNX graphs plus a JSON sidecar:

```
region-classifier.json            the spec: vocabulary, preprocessing, thresholds
region-classifier.backbone.onnx   image  -> stride-8 feature map
region-classifier.head.onnx       features + N boxes -> N probability rows
```

**Two graphs, not one.** A 1080p screenshot yields around a thousand boxes, and two
7×7×128 RoIAlign crops each is hundreds of megabytes of activations. The backbone runs
once per image and the head runs in chunks over its output, so memory is bounded
without recomputing features. It is also the seam a GPU backbone needs: `RoiAlign` has no
CoreML kernel, so a single fused graph would be partitioned in the middle by the runtime
rather than at a boundary anyone chose.

**Swift parses the spec; C never does.** The C engine takes a struct of numbers and has
no idea what a role is called, so a retrained model can add a class without touching
C++. In exchange, Swift validates what C cannot: that every role is one Mary acts on
(`AXNodeCategory.category(role:) != .other`), and that the tensor names match what
`Classifier.cpp` binds by.

**Preprocessing is a contract.** VisionAX's `Training/visionax_train/preprocess.py` and
`Sources/CVisionAX/ClassifierPreprocess.cpp` here must produce identical tensors — same
`INTER_AREA` resize, same mean-colour padding, same per-axis box scaling, same
`floor(x + 0.5)` rounding. `RegionClassifierTests.matchesPythonExactly` runs the C++
path against probabilities Python computed and fails if they drift.

### The backbone on Metal

The backbone — one pass over the whole image, ~40% of a page read on ONNX Runtime's CPU
path — can run on MLX instead. `RegionClassifier(specURL:backbone:)` and
`RegionClassifier.bundled(backbone:)` take `.automatic` (the default), `.onnx` or `.mlx`:

- `.automatic` runs `Backbone/MLXRegionBackbone` on Metal when three things hold — an
  `mlx.metallib` sits where MLX will look for it (`Backbone/MetalLibrary`, the same five
  rungs in the same order), the spec names MLX weights (`files.backbone_mlx`), and those
  weights say they were converted from this model's own ONNX backbone (`source_sha256`
  against `sha256.backbone`) — and runs the ONNX backbone otherwise. Every check happens
  before any MLX op, because with no metallib the first GPU op aborts the process.
- `backboneDescription` says what ran, and why not the GPU: `mlx-metal`, `onnx-cpu`, or
  `onnx-cpu (no mlx.metallib beside the binary — …)`. The choice is logged once under
  `nyc.rao.frigate` / `vision`.
- `FRIGATE_VISION_BACKBONE=onnx|mlx|auto` overrides `.automatic` only — the A/B switch for a
  process that took the default, such as Mary's Sand.

Only the backbone moves. Preprocessing stays in C++ (it is the contract with Python) and so
does the RoiAlign head: `vx_classifier_prepare_image` → the backbone →
`vx_classifier_classify_features`, with the map's shape checked against the head before it
runs. The MLX network (`Backbone/RegionBackboneNet`) is the ONNX graph rebuilt layer for
layer — BatchNorm folded, fp32, NHWC — and its weights come from VisionAX's
`vxtrain export-mlx`, which refuses any graph that is not exactly this architecture.
`RegionBackboneTests` holds the split path to equality with the single call on the ONNX
backbone, and the MLX backbone to the ONNX features and probabilities it replaces
(`FRIGATE_MLX_TESTS=1`, after `scripts/build-metallib.sh`).

A binary that wants the GPU path needs the metallib beside it:
`scripts/build-metallib.sh <config> --package <consumer>` for SwiftPM-built binaries (Mary's
`sand.sh` runs it), `--app <App.app>` inside an app bundle (Mary's `make-app.sh`).

## Adding an engine capability

The seam is deliberately four steps, in this order:

1. Declare the C types and the `vx_engine_*` function in `Sources/CVisionAX/include/visionax.h`.
2. Add the method to `visionax::Engine` (`Engine.hpp`/`Engine.cpp`), with the real
   work in its own `.hpp`/`.cpp` pair beside it.
3. Wire the `extern "C"` facade in `visionax.cpp` — argument checks, `cv::Mat`
   wrapping, malloc'd outputs freed through a matching `vx_*_free`, and no exception
   crossing the boundary.
4. Add the Swift call on `VisionEngine`, returning the package's own value types.
5. If it needs a model, ship it under `Sources/FrigateVisionAX/Resources/Models` and read its
   spec in Swift — never in C.

A value type that a dataset sample or an accessibility tree has to carry belongs in
VisionAXCore instead, so the harvester and the trainer can agree on it without the engine.

## The media lane

`readMediaControls` finds a video player's transport in a screenshot — the progress
track, the row of controls, and what each glyph depicts — without knowing what site it
is looking at. What it relies on is the layout every player shares: a thin two-tone track
across the bottom of the picture, and a row of evenly sized glyphs beneath it spanning
its width, with play at the left end and full screen at the right.

```swift
let reading = try engine.readMediaControls(in: frame, previous: frameJustBefore)
reading.playback          // .playing / .paused / .unknown
reading.witnesses         // ["picture moved (0.199)"] — WHY it says that
reading.playPause?.frame  // where to click
reading.progress?.fraction
```

**It reports witnesses, not a verdict dressed as one.** Whether a video is playing is
decided from three independent observations — the picture moved between two frames, the
progress fraction advanced, the clock advanced — and `witnesses` names the ones that
actually spoke. A glyph is only the tiebreak, because a button says what pressing it
would *do*, which is one frame behind whatever just happened.

The glyph silhouettes are **drawn, not shipped**: twelve shapes in `MediaGlyphs.cpp`,
compared by mask overlap after both candidate and template are normalised to the same box.
There is no model to download and the lane works with no classifier installed at all.

Two environment variables make tuning it repeatable, and both are off by default:
`VISIONAX_MEDIA_TRACE=1` prints which gate refused each candidate, and the `readFile`
and `dumpRows` tests in `RowDumpTests` read any image off disk (`VISIONAX_MEDIA_FILE`).
Real captures live in `Tests/FrigateVisionAXTests/Fixtures/media/`; every fix so far came from
adding one and watching it fail first.

## The page map

`scene.pageMap()` turns a perception into the thing a consumer acts on: rows with a
frame, a role, what they afford, a label, and **where that label came from**.

It does not go through `roster`. The roster drops any node the classifier did not name
and any node with no words inside it, which on a page of search results is nearly all of
them — measured, a results page read as four chrome buttons. The map keeps those rows and
reports its confidence instead.

Three sources, joined before anything is classified:

| Source | What it finds |
|---|---|
| Canny regions | Anything with an edge |
| `TextLines.proposals` | Boxes with no edge at all. A result title is an anchor around a heading: no border, no fill, nothing for an edge detector to find, and perfectly legible to recognition |
| `PageGrouping` | Which rows belong together, and in what order |

**The text union runs at harvest time too** (VisionAX's `HarvestSession.record`), through
the same function. A proposal the model meets only at inference is a proposal it was never
trained on, and it answers `none` for exactly the rows that matter.

**Grouping is geometry.** Bands of boxes that share a line; bands merged with whatever
sits close under them, judged against the neighbouring gap rather than a threshold; runs
of similar bands at a regular pitch become a list, which is what makes "the first one"
mean anything. Cards, forms, toolbars and overlays fall out of the same pass.

**The label ladder** is classifier → words inside → words beside it (fields only, since a
button labels itself) → a drawn icon → `"button 3"`. Every row keeps a name and records
which rung produced it, so a consumer can tell a read name from a position.

### The icon bank

`IconGlyphs.cpp` draws twenty-two interface icons — search, close, menu, chevrons, add,
more, share, microphone, bell, account, star, heart, delete, done, settings, home, filter,
cart, download — and matches a candidate by **blurred correlation**, not silhouette
overlap.

That is a measured choice. Overlap alone scored a magnifier drawn with a 3px stroke at
0.28 against the same magnifier drawn with a 4px stroke, *below* a cross at 0.30, and read
a checkmark as a magnifier: two icons of the same shape barely overlap when their strokes
differ by a pixel, because a stroke is mostly edge. Softening each mask into a field and
correlating them scores every independently drawn icon in the test fixture between 0.79
and 0.996, with a blank button at 0.0 and the floor at 0.72 — raised from 0.55 when a
live watch page had two dozen patches of video picture named as icons.

It is a fallback rung, never an oracle: an icon under the floor keeps its position and is
still pressable as "button 4".

## Benchmarks

Measured with `Tests/FrigateVisionAXTests/BenchmarkTests.swift` over real captures taken from a
live browser, each 900×752 at 1×. M4 Max, debug build. Median of seven. Run from Frigate.
These are the ONNX CPU backbone's numbers, from before the Metal path;
`RegionBackboneTests.benchmarkRegionBackbone` measures the two backbones side by side:

    VISIONAX_BENCH=/dir/of/pngs swift test --filter benchmarkOnePipelineCall   # the package as a unit
    VISIONAX_BENCH=/dir/of/pngs swift test --filter benchmarkTheLanes          # each lane on its own

### The classifier's backbone on Metal

`RegionBackboneTests.benchmarkRegionBackbone` over the six real captures in
`Tests/FrigateVisionAXTests/Fixtures/media`. M4 Max, debug build, median of seven, nothing else
running. Each figure is the whole classify phase — preprocessing, backbone, chunked head —
with the MLX backbone against the ONNX one:

| Capture | Boxes | Size | ONNX CPU | MLX Metal | |
|---|---|---|---|---|---|
| youtube-chrome-paused-bright | 1,051 | 1150×679 | 122.5 ms | 42.2 ms | ×2.9 |
| youtube-chrome-playing | 532 | 1150×679 | 114.4 ms | 36.2 ms | ×3.2 |
| youtube-safari-bright-frame | 1,288 | 900×752 | 105.5 ms | 37.8 ms | ×2.8 |
| youtube-safari-just-started | 781 | 900×752 | 102.3 ms | 34.2 ms | ×3.0 |
| youtube-safari-paused | 681 | 900×752 | 101.3 ms | 31.6 ms | ×3.2 |
| youtube-safari-playing | 701 | 900×752 | 102.7 ms | 32.2 ms | ×3.2 |

Against the ONNX backbone, the Metal feature maps agree to within 1e-5 at 320×200, 900×752,
1280×800 and 1920×1080, and the probabilities to within 1.5e-6 with no argmax
disagreement. Resident memory is flat across a hundred reads (1,145 → 1,149 → 1,149 MB), so
MLX's buffer cache is left at its default. The ~35 ms that remains covers preprocessing, the
RoiAlign head on ONNX Runtime's CPU path and the copies across the C boundary; how it splits
between them has not been measured yet.

### One pipeline call, entry to exit

The number to quote for this package. `perceive()` records its own phases, so the
breakdown comes from inside the call rather than from invoking the lanes separately —
timed from outside, the parts add up to more than the whole, because each call rebuilds
the image buffer, the union is invisible, and the classifier appears to pay for a
conversion the detector already did. The phases always sum to the total; whatever they do
not name shows as `other`.

**A page read** — `lanes: [.regions, .text]` with a classifier, then `pageMap()`:

```
  crop + buffer      2.3ms     1%     BGRA once, shared by every lane
  text             124.0ms    56%     accurate recognition over the whole crop
  detect             3.5ms     2%     Canny → region tree
  union text         2.0ms     1%     text lines added as proposals
  classify          88.0ms    40%     ResNet18 backbone + head, 200 boxes
  icons              0.7ms     0%     the drawn bank, over wordless boxes only
  other              0.0ms     0%
  TOTAL            220.5ms   100%
  pageMap()          3.0ms            grouping, label ladder, dedup
  ────────────────────────
  ENTRY→EXIT       223.5ms            → 91 rows, 9 actionable, 63 text runs
```

**A media read** — `lanes: [.media, .text]`, no classifier:

```
  crop + buffer      2.2ms     6%
  media             23.5ms    66%     band scan, control blobs, glyph match
  clock             10.0ms    28%     the bar strip alone, magnified 3×
  other              0.0ms     0%
  TOTAL             35.8ms   100%
  ENTRY→EXIT        35.9ms            → transport found
```

Across the four captures:

| Capture | Page read | Media read |
|---|---|---|
| article | 244 ms | 50 ms |
| results | 224 ms | 47 ms |
| watch (playing) | 224 ms | 49 ms |
| watch (paused, transport up) | 212 ms | 36 ms |

**Two lanes, two orders of magnitude apart, and that is the design.** A page read is
~220 ms and is dominated by two things — accurate text at 55–60% and the classifier at
~40%. A media read is ~40 ms because it asks for neither: the glyphs are drawn rather
than learned, and the only text it reads is the clock inside a bar it has already found.
That is what lets a transport be driven interactively and re-read to prove it moved.

**Detection is not the cost, and neither is the map.** Canny is 3–30 ms depending on what
the page draws, the text union is ~2 ms, the icon bank under 1 ms on a page and 10 ms on
one full of wordless controls, and grouping plus the label ladder plus deduplication is
2–6 ms.

### Each lane on its own

Useful when tuning one of them; the totals do not compose, for the reason above.

| Stage | Article | Results | Watch | Notes |
|---|---|---|---|---|
| detect (Canny) | 6 ms | 7 ms | 40 ms | 182 / 201 / 973 nodes |
| text, fast | 33 ms | 25 ms | 16 ms | 34 / 49 / 9 runs |
| text, accurate | 150 ms | 150 ms | 83 ms | 37 / 63 / 14 runs |
| classify | 99 ms | 105 ms | 113 ms | 181 / 200 / 972 boxes |
| media, one frame | 27 ms | 30 ms | 45 ms | |
| media, two frames | 38 ms | 38 ms | 47 ms | the motion witness costs ~10 ms |
| icon bank | 3 ms | 3 ms | 4 ms | 6 / 6 / 103 boxes |

What the numbers say:

- **Accurate recognition is over half the page read**, and it is worth it: on the results
  page it found 63 runs against fast's 49 — 29% more words for the label ladder, on the
  page type the browsing lane exists for. The media lane keeps the fast pass, where the
  strip is magnified first and the answer is four digits.
- **The classifier costs about the same whatever the page.** 181 boxes and 972 boxes both
  land near 100 ms, because one backbone pass over the image dominates and the head runs
  the boxes in chunks. Adding text-line proposals roughly doubled the box count for
  almost nothing.

At 2× the pixels (the same page upscaled to 1834×1504 — an approximation of a retina
capture, not a faithful one) detection rises to 33 ms, classification to 277 ms as the
detector proposes 731 boxes instead of 200, and a page read to 545 ms.

### One whole round trip

The lane table above is what perception costs. What a person waits for is the whole act,
and no lane contains it: resolve the browser → read its shell through Accessibility →
claim the stage → look → resolve the phrase → press → wait for the shell → look again →
judge. Mary's `mary-web-probe --roundtrip "<phrase>"` timestamps the engine's own events,
so the breakdown below is measured rather than assembled from parts.

Pressing a link on a live Wikipedia article, in Safari:

```
      0ms  +   0ms  resolved Safari
    258ms  + 258ms  read the shell — Ski touring - Wikipedia
    859ms  + 601ms  looked — 58 rows, 52 named, 16 groups
   1212ms  + 354ms  clicked Slovenščina
   2677ms  +1465ms  looked again — 45 rows, 32 named, 13 groups
   2678ms  +   0ms  receipt — the page became Turno smučanje - Wikipedija
   2678ms  ── whole act
```

| Act | Whole trip | Receipt it earned |
|---|---|---|
| Press a link that navigates | 2.7 s | `navigation` — the strongest |
| Press a control that does not navigate (opens a menu) | 3.3 s | `targetChanged` |
| Press a phrase that fits two rows | 0.85 s | refused, naming both, nothing pressed |
| Search the web and open a result | 5.7 s | `navigation`, of which 1.7 s is the search page loading |

## Consuming it

A consumer names the product and imports the module:

```swift
// in the consumer's Package.swift — Mary's MaryComputerUse, the one target with the edge
.package(path: "../Frigate")
.product(name: "FrigateVisionAX", package: "Frigate")
```

```swift
import FrigateVisionAX   // the runtime, plus VisionAXCore's AX types

let engine = try VisionEngine()

// One look, in whichever lanes you want, with the map back to the screen.
let scene = try engine.perceive(
    image: capture,
    projection: ScreenProjection(origin: window.origin, pixelsPerPoint: measuredScale),
    lanes: [.media, .text],
    previous: captureJustBefore)

scene.media?.playPause.map { scene.screenPoint(of: $0) }   // global screen points
scene.hitTest(screenPoint: cursor)                          // smallest box under a point
scene.roster(pid: pid, appName: "Safari", windowTitle: title)
```

**The projection is the caller's to supply, and its scale must be MEASURED** —
`image.width / window.frame.width` after the capture, not assumed. An image does not carry
its own density, and a 2× guess on a 1× display puts every click at half the distance from
the window's corner. A region of interest is expressed as a crop, and the crop moves the
projection rather than the frames, so a sub-detection still lands correctly on screen.

```swift
// Boxes only.
let detection = try engine.detectRegions(in: screenshot, title: "Safari")

// Boxes with AX roles, when a model is bundled.
if let classifier = try RegionClassifier.bundled() {
    let named = try engine.detectAndClassify(
        in: screenshot, title: "Safari", using: classifier)
}
```

Two things a consumer should know. The resource bundle is named for the defining package
and target — **`Frigate_FrigateVisionAX.bundle`** — so a copied `.app` must carry it in
`Contents/Resources` (Mary's `make-app.sh` does). And `RegionClassifier.bundled()` looks in
the host app's own `Contents/Resources` before it touches `Bundle.module`: SwiftPM's
generated accessor calls `fatalError` when the resource bundle is not beside the
executable, so reaching for it first would TRAP inside a copied `.app` rather than return
nil. Without the bundle the classifier is simply absent, which the media lane survives and
the element lane reports by name.
