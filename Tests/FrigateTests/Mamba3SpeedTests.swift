//  WHAT: How fast the Mamba-3 commons reads on this Mac, as Braid's reader reads with it (`Mamba3Model.read`, no
//        vocabulary projection): four 1,024-token pieces into one state, three times, the median printed. With
//        FRIGATE_MAMBA3_SPEED naming the 1.5b's folder; FRIGATE_MAMBA3_SPEED_BASELINE names a file that holds the last
//        piece's reading the first time and is held to it after, so a faster read is the same read.
//  PIN:  THE BASELINE BOUND IS 1e-3, NOT TIGHTER. The projections round to BF16, so any reordering of the float32 scan
//        (a chunk length, a fused pass) flips a few roundings: 2e-4 measured for each such change on 2026-11-19, where
//        the BF16 read itself sits 5e-3 from Braid's oracle. A wrong scan is off by orders more. How close the read is
//        to Braid's is Mamba3LiveTests' question, at float32.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

private var speedFolder: URL? {
    guard ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1",
          let path = ProcessInfo.processInfo.environment["FRIGATE_MAMBA3_SPEED"] else { return nil }
    return URL(fileURLWithPath: path)
}

@Suite("Mamba-3, read speed", .enabled(if: speedFolder != nil), .serialized)
struct Mamba3SpeedTests {
    @Test func readsFourPiecesIntoOneState() throws {
        let model = try Mamba3Loader.load(from: speedFolder!, weightDType: .bfloat16)
        let piece = 1_024, pieces = 4
        let tokens = (0 ..< piece * pieces).map { Int32(($0 * 7_919) % 120_000 + 100) }
        // Warm: the kernels made once.
        _ = model.read(MLXArray(tokens[0 ..< 64]).reshaped(1, 64), cache: model.newCache(parameters: nil), tap: 12)
        var seconds: [Double] = []
        var final: Mamba3Reading?
        for _ in 0 ..< 3 {
            let cache = model.newCache(parameters: nil)
            let started = Date()
            for i in 0 ..< pieces {
                let slice = tokens[(i * piece) ..< ((i + 1) * piece)]
                let reading = model.read(MLXArray(Array(slice)).reshaped(1, piece), cache: cache, tap: 12)
                eval(reading.tappedSum, reading.last)
                final = reading
            }
            seconds.append(Date().timeIntervalSince(started))
        }
        let median = seconds.sorted()[1]
        print(String(format: "[mamba3-speed] %d tokens in %.3f s (median of 3: %.3f / %.3f / %.3f) = %.0f tok/s",
                     piece * pieces, median, seconds[0], seconds[1], seconds[2], Double(piece * pieces) / median))
        guard let baseline = ProcessInfo.processInfo.environment["FRIGATE_MAMBA3_SPEED_BASELINE"], let final else { return }
        let url = URL(fileURLWithPath: baseline)
        let now = concatenated([final.tappedSum.asType(.float32), final.last.asType(.float32)], axis: -1)
        if FileManager.default.fileExists(atPath: url.path) {
            let kept = try loadArrays(url: url)["reading"]!
            let error = (sqrt(((now - kept) * (now - kept)).sum()) / sqrt((kept * kept).sum())).item(Float.self)
            print(String(format: "[mamba3-speed] relative error against the baseline: %.2e", error))
            #expect(error <= 1e-3)
        } else {
            try save(arrays: ["reading": now], url: url)
            print("[mamba3-speed] baseline written to \(url.path)")
        }
    }
}
