/// MLXAccelerate — Portable CPU vector kernels.
///
/// Latency-critical single-vector paths (e.g. HNSW graph traversal computes one
/// small distance at a time) cannot amortize a GPU dispatch; these scalar
/// kernels use 8-way accumulator splitting so the compiler auto-vectorizes them
/// at -O on every platform. The vDSP_* functions mirror the subset of Apple's
/// Accelerate API used by callers that must also build on Linux.
///
/// Migrated from Totem's `Sources/Utilities/AccelerateLinux.swift`.

import Foundation

public typealias vDSP_Length = Int

/// Dot product of two Float vectors (vDSP signature compatible).
public func vDSP_dotpr(_ x: [Float], _ strideX: Int, _ y: [Float], _ strideY: Int,
                       _ result: inout Float, _ length: Int) {
    guard strideX == 1 && strideY == 1 else {
        result = 0.0
        for i in 0..<length { result += x[i * strideX] * y[i * strideY] }
        return
    }
    var s0: Float = 0, s1: Float = 0, s2: Float = 0, s3: Float = 0
    var s4: Float = 0, s5: Float = 0, s6: Float = 0, s7: Float = 0
    var i = 0
    while i &+ 8 <= length {
        s0 += x[i]   * y[i];   s1 += x[i+1] * y[i+1]
        s2 += x[i+2] * y[i+2]; s3 += x[i+3] * y[i+3]
        s4 += x[i+4] * y[i+4]; s5 += x[i+5] * y[i+5]
        s6 += x[i+6] * y[i+6]; s7 += x[i+7] * y[i+7]
        i &+= 8
    }
    while i < length { s0 += x[i] * y[i]; i &+= 1 }
    result = s0+s1+s2+s3+s4+s5+s6+s7
}

/// Dot product of two Double vectors (vDSP signature compatible).
public func vDSP_dotprD(_ x: [Double], _ strideX: Int, _ y: [Double], _ strideY: Int,
                        _ result: inout Double, _ length: Int) {
    result = 0.0
    for i in 0..<length {
        result += x[i * strideX] * y[i * strideY]
    }
}

/// Element-wise subtraction of two Double vectors (vDSP signature compatible).
public func vDSP_vsubD(_ x: [Double], _ strideX: Int, _ y: [Double], _ strideY: Int,
                       _ result: inout [Double], _ strideResult: Int, _ length: Int) {
    for i in 0..<length {
        result[i * strideResult] = x[i * strideX] - y[i * strideY]
    }
}

/// Squared Euclidean distance between two Float buffers of length `n`.
///
/// 8-accumulator split mirrors the HNSW traversal kernel proven in Totem;
/// auto-vectorizes at -O. Callers needing true Euclidean distance take
/// `sqrt` of the result.
@inline(__always)
public func squaredEuclidean(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int) -> Float {
    var s0: Float = 0, s1: Float = 0, s2: Float = 0, s3: Float = 0
    var s4: Float = 0, s5: Float = 0, s6: Float = 0, s7: Float = 0
    var i = 0
    while i &+ 8 <= n {
        let d0 = a[i]   - b[i];   let d1 = a[i+1] - b[i+1]
        let d2 = a[i+2] - b[i+2]; let d3 = a[i+3] - b[i+3]
        let d4 = a[i+4] - b[i+4]; let d5 = a[i+5] - b[i+5]
        let d6 = a[i+6] - b[i+6]; let d7 = a[i+7] - b[i+7]
        s0 += d0*d0; s1 += d1*d1; s2 += d2*d2; s3 += d3*d3
        s4 += d4*d4; s5 += d5*d5; s6 += d6*d6; s7 += d7*d7
        i &+= 8
    }
    while i < n {
        let d = a[i] - b[i]
        s0 += d*d
        i &+= 1
    }
    return s0+s1+s2+s3+s4+s5+s6+s7
}

/// Convenience overload for Swift arrays.
@inline(__always)
public func squaredEuclidean(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count, "Vectors must have same dimension")
    return a.withUnsafeBufferPointer { pa in
        b.withUnsafeBufferPointer { pb in
            squaredEuclidean(pa.baseAddress!, pb.baseAddress!, a.count)
        }
    }
}
