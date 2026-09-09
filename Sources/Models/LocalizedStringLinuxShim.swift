// `String(localized:)` is Darwin-only: swift-corelibs-foundation ships no `localized:`
// overload (verified on Swift 6.3 / Ubuntu 24.04, with and without FoundationEssentials).
// Upstream swift-transformers 1.3.x and mlx-swift-lm 3.x both use it for user-facing error
// text, so on Linux this returns the already-interpolated literal unchanged — the strings
// are English-only in the vendored sources anyway.
//
// Kept `internal` deliberately: each vendored module carries its own copy rather than
// exporting an initializer that would collide with Foundation's on Apple platforms.
#if !canImport(Darwin)

    import Foundation

    extension String {
        init(localized value: String) { self = value }
    }

#endif
