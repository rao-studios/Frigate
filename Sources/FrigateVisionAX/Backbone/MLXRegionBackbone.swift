//
//  MLXRegionBackbone.swift
//  FrigateVisionAX
//
//  WHAT: The classifier's backbone on Metal — RegionBackboneNet behind the RegionBackbone seam.
//  IN:   <model>.backbone.safetensors (VisionAX's `vxtrain export-mlx`), chosen by
//        BackboneSelection only after every MLX-free check has passed
//  OUT:  RegionFeatures, CHW, for vx_classifier_classify_features
//  PIN:  EVERY MLX CALL RUNS INSIDE `withError`. MLX's default handler turns an error — a shape
//        mismatch, a failed eval — into fatalError, and in an app that is a crash on a page read.
//        Inside withError it is a Swift throw: a load that fails leaves the classifier on its
//        ONNX backbone (BackboneSelection says why), and a run that fails is a
//        RegionClassifierError the caller can see.
//        FP32, DELIBERATELY. The head was trained and calibrated on fp32 ONNX features and the
//        spec's confidence floor is a probability; half precision would move answers near that
//        floor for milliseconds nobody has measured yet.
//        ONE FORWARD PASS AT A TIME. MLX arrays are not Sendable and the default stream is
//        shared; a lock around the pass keeps "one classifier serves many threads" true
//        without claiming MLX itself is thread-safe.
//        WARMED AT LOAD, AND PROVEN THERE. The first dispatch of each kernel JIT-compiles it;
//        a 64×64 pass at creation keeps that cost off the first page read, and its output
//        shape is checked so a network that runs but answers the wrong shape never gets in.
//

import Foundation
import MLX
import MLXNN

enum MLXRegionBackboneError: Error, CustomStringConvertible {
    case malformed(String)

    var description: String {
        switch self {
        case .malformed(let reason): return reason
        }
    }
}

final class MLXRegionBackbone: RegionBackbone, @unchecked Sendable {
    let name = "mlx-metal"

    private let net: RegionBackboneNet
    private let lock = NSLock()
    private var completedRuns = 0

    /// Forward passes over real images so far; the warm-up is not counted.
    var runs: Int {
        lock.lock()
        defer { lock.unlock() }
        return completedRuns
    }

    init(weights url: URL) throws {
        net = try withError { () throws -> RegionBackboneNet in
            let (arrays, _) = try loadArraysAndMetadata(url: url)
            guard let smooth = arrays["smooth.weight"], smooth.ndim == 4 else {
                throw MLXRegionBackboneError.malformed(
                    "\(url.lastPathComponent) has no smooth.weight, so its width is unknown")
            }
            let channels = smooth.dim(0)
            let net = RegionBackboneNet(channels: channels)
            try net.update(parameters: ModuleParameters.unflattened(arrays), verify: [.all])
            eval(net)

            let warm = net(MLXArray.zeros([1, 64, 64, 3]))
            eval(warm)
            guard warm.shape == [1, 8, 8, channels] else {
                throw MLXRegionBackboneError.malformed(
                    "the network maps 64×64 to \(warm.shape), not [1, 8, 8, \(channels)]")
            }
            return net
        }
    }

    func features(
        image: UnsafeBufferPointer<Float>, height: Int, width: Int
    ) throws -> RegionFeatures {
        guard height > 0, width > 0, image.count == 3 * height * width else {
            throw MLXRegionBackboneError.malformed(
                "expected 3×\(height)×\(width) floats, got \(image.count)")
        }
        let pixels = Array(image)

        lock.lock()
        defer { lock.unlock() }
        return try withError { () throws -> RegionFeatures in
            let chw = MLXArray(pixels, [1, 3, height, width])
            let nhwc = net(chw.transposed(0, 2, 3, 1))
            // Back to CHW; the reshape of a transposed array is a contiguous copy, which is
            // the layout ONNX Runtime's head reads without another pass.
            let flat = nhwc.transposed(0, 3, 1, 2).reshaped([-1])
            eval(flat)
            completedRuns += 1
            return RegionFeatures(
                values: flat.asArray(Float.self),
                channels: nhwc.dim(3), height: nhwc.dim(1), width: nhwc.dim(2))
        }
    }
}
