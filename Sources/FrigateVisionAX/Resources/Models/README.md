# Bundled models

`region-classifier.json` (the spec), `region-classifier.backbone.onnx` and
`region-classifier.head.onnx` land here from VisionAX's training pipeline:
`uv run vxtrain export --out ../../Frigate/Sources/FrigateVisionAX/Resources/Models`, run from
`VisionAX/Training`.

The `.onnx` files are tracked with git-lfs. After a fresh clone run `git lfs pull`;
without it `RegionClassifier.bundled()` throws `.modelIsLFSPointer` rather than
handing ONNX Runtime a pointer file.

This directory ships as a SwiftPM resource bundle, so a consumer (Mary) gets the
model by depending on Frigate's `FrigateVisionAX` product — but an app bundle must copy
`Frigate_FrigateVisionAX.bundle` into `Contents/Resources` for `Bundle.module` to find it.
SwiftPM names the bundle `<package>_<target>`: the runtime target is defined in Frigate.
