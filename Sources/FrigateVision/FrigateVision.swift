// FrigateVision — VisionAX, through Frigate.
//
// Perceive a page:  VisionEngine.perceive(image:projection:lanes:)  → VisionScene
// Map it:           scene.pageMap()                                  → PageMap
// Name roles:       RegionClassifier.bundled()
// Drive a player:   VisionEngine.readMediaControls(in:previous:)
//
// WHAT: Frigate hosts every ML surface its consumers take from one place; this is the
//       pixel-perception one. VisionAX keeps its own repository — bench, harvester,
//       corpus and training live there — and this module is the door.
// PIN:  A RE-EXPORT, NOT A COPY. Every type here is VisionAX's own; nothing is renamed,
//       wrapped or duplicated, so a consumer that once said `import VisionAX` says
//       `import FrigateVision` and changes nothing else.
//       NOT IN THE `Frigate` UMBRELLA, DELIBERATELY. VisionAX declares AXNodeSnapshot,
//       AXScreenElement and AXNodeCategory under the same names a machine layer such as
//       Mary's already uses; a consumer importing the umbrella must never inherit that
//       ambiguity. FrigateExports.swift does not name this module, and a test pins it.
//       macOS ONLY. OpenCV and ONNX Runtime arrive as xcframeworks; this target does not
//       exist when the manifest is evaluated on Linux.

@_exported import VisionAX
