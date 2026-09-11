//
//  MetalLibrary.swift
//  FrigateVisionAX
//
//  WHAT: Where MLX will find its compiled shaders, answered without asking MLX.
//  IN:   BackboneSelection, before any MLX op
//  PIN:  THE SAME SEARCH MLX MAKES, IN THE SAME ORDER — Frigate's vendored
//        mlx/backend/metal/device.cpp, load_default_library():
//          1. <binary dir>/mlx.metallib
//          2. <binary dir>/Resources/mlx.metallib
//          3. mlx-swift_Cmlx.bundle/…/default.metallib, beside the main bundle or in any
//             loaded bundle's resources (SWIFTPM_BUNDLE)
//          4. <binary dir>/Resources/default.metallib
//          5. default.metallib, relative to the working directory (METAL_PATH)
//        "Binary dir" is the directory of the image MLX is linked into — for SwiftPM builds
//        the same image as this module, which is why #dsohandle answers it: an executable in
//        .build/<config>/, an .xctest bundle's Contents/MacOS/, an app's Contents/MacOS/.
//        A miss here is the whole reason the backbone stays on CPU: when every rung misses,
//        MLX aborts the process on its first GPU op.
//

import Foundation

enum MetalLibrary {
    static let swiftPMBundle = "mlx-swift_Cmlx.bundle"

    /// The metallib MLX will load, or nil when every rung misses.
    static func locate() -> URL? {
        candidates().first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Every path MLX tries, in its order.
    static func candidates() -> [URL] {
        var urls: [URL] = []
        let binary = binaryDirectory
        if let binary {
            urls.append(binary.appendingPathComponent("mlx.metallib"))
            urls.append(binary.appendingPathComponent("Resources/mlx.metallib"))
        }
        let roots = [Bundle.main.bundleURL] + Bundle.allBundles.compactMap(\.resourceURL)
        for root in roots {
            let bundle = root.appendingPathComponent(swiftPMBundle)
            // NSBundle's resourceURL is Contents/Resources for a macOS-style bundle and the
            // bundle itself for a flat one; SwiftPM has produced both.
            urls.append(bundle.appendingPathComponent("Contents/Resources/default.metallib"))
            urls.append(bundle.appendingPathComponent("default.metallib"))
        }
        if let binary {
            urls.append(binary.appendingPathComponent("Resources/default.metallib"))
        }
        urls.append(
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("default.metallib"))
        return urls
    }

    /// The directory of the image this code — and MLX with it — was linked into.
    static var binaryDirectory: URL? {
        var info = Dl_info()
        guard dladdr(#dsohandle, &info) != 0, let path = info.dli_fname else { return nil }
        return URL(fileURLWithPath: String(cString: path)).deletingLastPathComponent()
    }
}
