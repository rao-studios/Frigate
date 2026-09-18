// Concrete `Downloader` for MLXLMCommon 3.x, backed by the vendored swift-transformers `Hub`.
//
// mlx-swift-lm 3.x deliberately dropped its hard dependency on the HuggingFace Hub and
// expects the host application to supply the concrete implementations. Upstream ships that
// wiring as the `MLXHuggingFace` macro target, which pulls in swift-syntax; this package is
// fully vendored and cross-compiles to Linux, so it wires the protocols by hand instead —
// the "integration package" shape upstream documents in MLXLMCommon's `upgrade.md`.

import Foundation
import Hub
import MLXLMCommon

/// Downloads model snapshots through `HubApi`.
public struct HubDownloader: Downloader {
    let hub: HubApi

    public init(hub: HubApi = HubDownloader.defaultHub) {
        self.hub = hub
    }

    /// `HubApi.shared` puts snapshots in ~/Documents/huggingface. When the
    /// launcher names a home for models (`HF_HOME` — Ambient sets it beside its
    /// data, out of Documents), snapshots go under `$HF_HOME/snapshots`; the
    /// Hub's cache already follows `HF_HOME` on its own.
    public static let defaultHub: HubApi = {
        guard let home = ProcessInfo.processInfo.environment["HF_HOME"], !home.isEmpty else {
            return .shared
        }
        return HubApi(downloadBase: URL(fileURLWithPath: home).appendingPathComponent("snapshots"))
    }()

    public func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        // `useLatest` has no HubApi equivalent: a snapshot always revalidates against the
        // revision unless HubApi itself is in offline mode, so honouring it is a no-op here.
        try await hub.snapshot(
            from: id,
            revision: revision ?? "main",
            matching: patterns,
            progressHandler: progressHandler)
    }
}
