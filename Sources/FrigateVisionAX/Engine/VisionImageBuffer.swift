//
//  VisionImageBuffer.swift
//  FrigateVisionAX
//
//  WHAT: CGImage → BGRA8 bytes the engine can read; vx_edge_map → gray CGImage.
//  IN:   VisionEngine
//  OUT:  vx_image_view (borrowed), CGImage (owned)
//  PIN:  Every CGImage is REDRAWN into a known 8-bit BGRA context rather than
//        trusted — a screenshot can arrive planar, 16-bit, indexed, or with an
//        exotic color space, and the engine reads raw bytes.
//

import CVisionAX
import CoreGraphics
import Foundation
import VisionAXCore

/// Owns one contiguous BGRA8 buffer for the life of a detection call.
struct VisionImageBuffer {
    private(set) var pixels: [UInt8]
    let width: Int
    let height: Int
    let bytesPerRow: Int

    init?(image: CGImage) {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else { return nil }
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: bytesPerRow * height)

        let drew: Bool = pixels.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress,
                  let context = CGContext(
                      data: base,
                      width: width,
                      height: height,
                      bitsPerComponent: 8,
                      bytesPerRow: bytesPerRow,
                      space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                          | CGBitmapInfo.byteOrder32Little.rawValue)
            else { return false }
            // The context's origin is bottom-left, so a straight draw would flip
            // the image; CGContext.draw handles that for us because the CGImage
            // carries its own orientation — row 0 of the buffer ends up as the
            // image's TOP row, which is what the engine documents.
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drew else { return nil }

        self.pixels = pixels
        self.width = width
        self.height = height
        self.bytesPerRow = bytesPerRow
    }

    /// The view is only valid inside `body`.
    func withImageView<R>(_ body: (vx_image_view) throws -> R) rethrows -> R {
        try pixels.withUnsafeBufferPointer { buffer in
            var view = vx_image_view()
            view.data = buffer.baseAddress
            view.width = Int32(width)
            view.height = Int32(height)
            view.bytes_per_row = Int32(bytesPerRow)
            view.format = VX_PIXEL_BGRA8
            return try body(view)
        }
    }
}

extension CGImage {
    /// An 8-bit grayscale image copied out of an engine edge map.
    static func fromEdgeMap(_ map: vx_edge_map) -> CGImage? {
        guard let data = map.data, map.width > 0, map.height > 0 else { return nil }
        let count = Int(map.bytes_per_row) * Int(map.height)
        let copy = Data(bytes: data, count: count)
        guard let provider = CGDataProvider(data: copy as CFData) else { return nil }
        return CGImage(
            width: Int(map.width),
            height: Int(map.height),
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: Int(map.bytes_per_row),
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent)
    }
}
