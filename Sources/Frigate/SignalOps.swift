/// MLXAccelerate — Signal processing helpers for the CLR seam-spectral features.
///
/// The seam strips are 1D arrays of length ≤ 32 (tile width/height).
/// A plain Swift DFT is fast enough for these sizes and avoids any dependency
/// on Accelerate (Apple-only) or complex MLX type handling.

import Foundation

// MARK: - Spectral Distance

/// Mean absolute difference of the first `bins` DFT magnitude coefficients.
///
/// Matches Python `_sdisc(a, b)`:
/// ```python
/// mean(|abs(rfft(a))[:bins] - abs(rfft(b))[:bins]|)
/// ```
/// - Parameters:
///   - a: First 1-D signal.
///   - b: Second 1-D signal (same length as `a`).
///   - bins: Number of low-frequency coefficients to compare (default 8).
/// - Returns: Mean absolute spectral distance ≥ 0.
public func spectralDistance(_ a: [Float], _ b: [Float], bins: Int = 8) -> Float {
    let magA = dftMagnitude(a, bins: bins)
    let magB = dftMagnitude(b, bins: bins)
    var sum: Float = 0
    let count = min(bins, min(magA.count, magB.count))
    for i in 0..<count { sum += abs(magA[i] - magB[i]) }
    return count > 0 ? sum / Float(count) : 0
}

// MARK: - Internal

/// Compute the first `bins` DFT magnitudes of a 1-D real signal.
///
/// Uses a direct O(n·bins) sum — correct and fast for n ≤ 64 (tile edge size).
func dftMagnitude(_ signal: [Float], bins: Int) -> [Float] {
    let n = signal.count
    guard n > 0 else { return [] }
    let count = min(bins, n / 2 + 1)
    var result = [Float](repeating: 0, count: count)
    for k in 0..<count {
        var re: Double = 0, im: Double = 0
        let twoPiKoN = -2.0 * Double.pi * Double(k) / Double(n)
        for t in 0..<n {
            let angle = twoPiKoN * Double(t)
            re += Double(signal[t]) * cos(angle)
            im += Double(signal[t]) * sin(angle)
        }
        result[k] = Float(sqrt(re * re + im * im))
    }
    return result
}
