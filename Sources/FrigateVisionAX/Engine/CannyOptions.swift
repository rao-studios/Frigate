//
//  CannyOptions.swift
//  FrigateVisionAX
//
//  WHAT: The engine's half of CannyOptions — its defaults and its C mirror.
//  IN:   CannyOptions (VisionAXCore: the value a dataset sample records)
//  OUT:  toC() → vx_canny_options
//  PIN:  `.standard` is seeded from the C default, so the two never disagree — and that is
//        why it is declared here, beside the engine, rather than in VisionAXCore.
//

import CVisionAX
import Foundation
import VisionAXCore

extension CannyOptions {
    /// The engine's own defaults, read from C so there is one source of truth.
    public static let standard = CannyOptions(vx_canny_options_default())

    init(_ c: vx_canny_options) {
        self.init(
            lowThreshold: c.low_threshold,
            highThreshold: c.high_threshold,
            apertureSize: Int(c.aperture_size),
            blurKernel: Int(c.blur_kernel),
            closeKernel: Int(c.close_kernel),
            minWidth: Int(c.min_width),
            minHeight: Int(c.min_height),
            mergeIOU: c.merge_iou,
            mergeSlack: Int(c.merge_slack),
            containmentSlack: Int(c.containment_slack),
            readingBand: Int(c.reading_band),
            maxDepth: Int(c.max_depth),
            maxNodes: Int(c.max_nodes))
    }

    func toC() -> vx_canny_options {
        var c = vx_canny_options()
        c.low_threshold = lowThreshold
        c.high_threshold = highThreshold
        c.aperture_size = Int32(apertureSize)
        c.blur_kernel = Int32(blurKernel)
        c.close_kernel = Int32(closeKernel)
        c.min_width = Int32(minWidth)
        c.min_height = Int32(minHeight)
        c.merge_iou = mergeIOU
        c.merge_slack = Int32(mergeSlack)
        c.containment_slack = Int32(containmentSlack)
        c.reading_band = Int32(readingBand)
        c.max_depth = Int32(maxDepth)
        c.max_nodes = Int32(maxNodes)
        return c
    }
}
