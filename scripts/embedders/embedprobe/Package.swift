// swift-tools-version: 6.0
// embedprobe: FrigateEmbedder from the command line, for parity and speed runs.
import PackageDescription

let package = Package(
    name: "embedprobe",
    platforms: [.macOS(.v15)],
    dependencies: [.package(path: "../../..")],
    targets: [
        .executableTarget(
            name: "embedprobe",
            dependencies: [.product(name: "Frigate", package: "Frigate")])
    ]
)
