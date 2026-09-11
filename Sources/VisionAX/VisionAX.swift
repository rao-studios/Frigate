//
//  VisionAX.swift
//  VisionAX
//
//  WHAT: Module-level facts. The engine lives in Engine/, the tree model in Accessibility/.
//  OUT:  VisionAXBench, Mary
//

import CVisionAX
import Foundation
@_exported import VisionAXCore

public enum VisionAX {
    /// The OpenCV the engine was linked against, e.g. "4.13.0".
    public static var openCVVersion: String {
        String(cString: vx_engine_opencv_version())
    }

    /// The engine's own version, stamped into every dataset sample so a training run
    /// can tell which detector proposed its boxes.
    public static var version: String {
        String(cString: vx_engine_version())
    }

    /// The ONNX Runtime the classifier runs on, e.g. "1.24.2". Nil when the runtime
    /// could not be reached — which, since it is statically linked, means something is
    /// wrong with the build rather than with the machine.
    public static var onnxRuntimeVersion: String? {
        vx_onnxruntime_version().map { String(cString: $0) }
    }

    /// Role stamped on every detected region until a classifier names it. A non-AX
    /// prefix, like Mary's "MaryScripted", marks the node as synthesized.
    public static let regionRole = "VXRegion"

    /// Role of the root node — the whole image, the way a walked window's root is
    /// "AXWindow".
    public static let windowRole = "AXWindow"
}
