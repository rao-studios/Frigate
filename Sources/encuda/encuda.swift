import ArgumentParser

#if !os(iOS) && !os(tvOS) && !os(watchOS) && !os(visionOS)
@main
struct Encuda: ParsableCommand {
    static let configuration = CommandConfiguration(
        subcommands: [Compile.self, Link.self]
    )
}
#else
// encuda is a CUDA codegen build-tool, never invoked on the iOS family. The CudaBuild
// plugin still gets cross-compiled for the iOS destination by xcodebuild, so it needs a
// trivial entry point here (Process et al. are unavailable on iOS).
@main
struct Encuda: ParsableCommand {}
#endif
