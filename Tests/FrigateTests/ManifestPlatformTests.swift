//
//  ManifestPlatformTests.swift
//  FrigateTests
//
//  WHAT: Everything macOS-only in the manifest stays inside `#if !os(Linux)`.
//  PIN:  THE LINUX BOX MUST NOT BE THE FIRST TO FIND OUT. `#if os(Linux)` is evaluated
//        where the manifest is compiled, so a macOS test run cannot execute the Linux
//        branch — it can only READ it. This reads Package.swift as text (the way Mary's
//        PackageLayeringTests does) and fails here, on a Mac, the moment VisionAX or its
//        xcframeworks escape the guard. Totem builds this package on Ubuntu with CUDA,
//        where an xcframework cannot be resolved at all.
//        AND THE UMBRELLA STAYS CLEAN. VisionAX declares AXNodeSnapshot and friends under
//        names other packages already use; `import Frigate` must never carry them.
//

import Foundation
import Testing

@Suite("Manifest platform guards")
struct ManifestPlatformTests {

    static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // FrigateTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // package root
    }

    static func manifest() throws -> [String] {
        try String(contentsOf: packageRoot.appendingPathComponent("Package.swift"),
                   encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
    }

    /// Which lines sit inside a `#if !os(Linux)` region — tracked by walking the file,
    /// so a nested `#if` cannot smuggle a line out of the guard.
    static func guardedLines(_ lines: [String]) -> Set<Int> {
        var guarded: Set<Int> = []
        var depth = 0
        var guardDepths: [Int] = []
        for (index, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("#if") {
                depth += 1
                if trimmed.contains("!os(Linux)") { guardDepths.append(depth) }
            } else if trimmed.hasPrefix("#endif") {
                guardDepths.removeAll { $0 == depth }
                depth -= 1
            } else if trimmed.hasPrefix("#else") || trimmed.hasPrefix("#elseif") {
                // The other arm of a `!os(Linux)` guard IS the Linux arm.
                guardDepths.removeAll { $0 == depth }
            }
            if !guardDepths.isEmpty { guarded.insert(index) }
        }
        return guarded
    }

    /// Nothing macOS-only may be declared where Linux will read it.
    @Test func visionStaysBehindTheLinuxGuard() throws {
        let lines = try Self.manifest()
        let guarded = Self.guardedLines(lines)
        // Comments explain the rule and must not be flagged by it.
        let tokens = ["VisionAX", "FrigateVision", "opencv", "onnxruntime"]
        var breaches: [String] = []
        for (index, line) in lines.enumerated() {
            let code = line.components(separatedBy: "//").first ?? line
            guard tokens.contains(where: { code.contains($0) }) else { continue }
            // The `var visionX: [T] = []` declarations are the seam itself: empty on
            // Linux, filled inside the guard. They carry no macOS-only reference.
            if code.contains("var vision") { continue }
            if code.contains("+ vision") { continue }
            guard !guarded.contains(index) else { continue }
            breaches.append("\(index + 1): \(line.trimmingCharacters(in: .whitespaces))")
        }
        #expect(
            breaches.isEmpty,
            """
            These manifest lines name a macOS-only dependency outside `#if !os(Linux)`:
            \(breaches.joined(separator: "\n"))

            Linux resolves this package (Totem's CUDA build) and cannot extract an
            xcframework. The dependency, the target and the product must share one guard.
            """)
    }

    /// The rule can fail — a guard around nothing passes forever.
    @Test func theGuardActuallyContainsSomething() throws {
        let lines = try Self.manifest()
        let guarded = Self.guardedLines(lines)
        let guardedCode = guarded.map { lines[$0] }.joined(separator: "\n")
        #expect(guardedCode.contains("VisionAX"), "the Linux guard names no macOS-only package")
        #expect(guardedCode.contains("FrigateVision"))
    }

    /// The umbrella never re-exports the vision module.
    @Test func theUmbrellaDoesNotCarryVisionAX() throws {
        let exports = try String(
            contentsOf: Self.packageRoot
                .appendingPathComponent("Sources/Frigate/FrigateExports.swift"),
            encoding: .utf8)
        #expect(!exports.contains("VisionAX"))
        #expect(!exports.contains("FrigateVision"))
    }
}
