//  WHAT: Holds `VENDORED-PATCHES.md`, the one list of what this tree changes in vendored MLX, to the marks in the tree
//        it is made from. A mark is `FRIGATE(<patch>)` in a comment where the change is, followed by what it does.
//  WHY:  The list was a README table kept by hand, and it had fallen behind: on 2026-11-22 a diff against the upstream
//        tags found 23 files changed or added where the table named about half, some of them with no mark at all.
//  PIN:  THE LIST IS WRITTEN, NEVER EDITED. `FRIGATE_WRITE_PATCH_LIST=1 swift test --filter VendoredPatchListTests`
//        writes it from the marks; without the variable the same test fails when the two disagree, and says how.
//  PIN:  A mark is found, an unmarked change is not. Nothing here reads upstream: after a sync, diff the trees below
//        against the new tags (`diff -rq`), and give every file that differs a mark.

import Foundation
import XCTest

final class VendoredPatchListTests: XCTestCase {
    /// The vendored MLX trees: the C++ core, its C binding, the generated kernel sources, the header copies, and
    /// mlx-swift's modules.
    static let trees = ["Sources/Cmlx/mlx", "Sources/Cmlx/mlx-c", "Sources/Cmlx/mlx-generated", "Sources/Cmlx/include",
                        "Sources/Cmlx/include-framework", "Sources/MLX", "Sources/MLXFFT", "Sources/MLXFast",
                        "Sources/MLXLinalg", "Sources/MLXNN", "Sources/MLXOptimizers", "Sources/MLXRandom"]
    static let list = "VENDORED-PATCHES.md"

    /// One mark: the patch it names, where it is, and what its comment says.
    struct Mark: Equatable {
        var patch: String
        var path: String
        var line: Int
        var says: String
    }

    func testTheListIsWhatTheMarksSay() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let marks = Self.marks(under: root)
        XCTAssertFalse(marks.isEmpty, "no mark found under \(root.path)")
        let made = Self.rendered(marks)
        let url = root.appendingPathComponent(Self.list)
        if ProcessInfo.processInfo.environment["FRIGATE_WRITE_PATCH_LIST"] == "1" {
            try Data(made.utf8).write(to: url, options: .atomic)
            return
        }
        let kept = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        guard kept != made else { return }
        let was = Set(kept.split(separator: "\n").map(String.init)), now = Set(made.split(separator: "\n").map(String.init))
        let gone = was.subtracting(now).sorted().prefix(6).map { "  listed, not in the tree: \($0)" }
        let new = now.subtracting(was).sorted().prefix(6).map { "  in the tree, not listed: \($0)" }
        XCTFail("\(Self.list) is not what the marks say. Write it again with FRIGATE_WRITE_PATCH_LIST=1.\n"
                + (gone + new).joined(separator: "\n"))
    }

    /// A comment's text continues over the comment lines that follow its mark, and ends at the comment's end or at an
    /// empty comment line.
    func testAMarkSaysItsWholeComment() {
        let source = """
        int before;
        // FRIGATE(one-thing): the first line
        // and the second.
        //
        // Not part of it.
        /* FRIGATE(another): a block
         * that goes on. */
        int after; // FRIGATE(one-thing)
        """
        XCTAssertEqual(Self.marks(in: source, path: "a.cpp"), [
            Mark(patch: "one-thing", path: "a.cpp", line: 2, says: "the first line and the second."),
            Mark(patch: "another", path: "a.cpp", line: 6, says: "a block that goes on."),
            Mark(patch: "one-thing", path: "a.cpp", line: 8, says: ""),
        ])
    }

    // MARK: - Reading the tree

    static func marks(under root: URL) -> [Mark] {
        var found: [Mark] = []
        for tree in trees {
            let base = root.appendingPathComponent(tree)
            guard let walk = FileManager.default.enumerator(at: base, includingPropertiesForKeys: [.isRegularFileKey]) else { continue }
            for case let file as URL in walk {
                guard (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true,
                      let data = try? Data(contentsOf: file), data.range(of: Data("FRIGATE(".utf8)) != nil,
                      let text = String(data: data, encoding: .utf8) else { continue }
                let path = String(file.standardizedFileURL.path.dropFirst(root.standardizedFileURL.path.count + 1))
                found += marks(in: text, path: path)
            }
        }
        return found.sorted { ($0.patch, $0.path, $0.line) < ($1.patch, $1.path, $1.line) }
    }

    static func marks(in text: String, path: String) -> [Mark] {
        let lines = text.components(separatedBy: "\n")
        var found: [Mark] = []
        for (index, line) in lines.enumerated() {
            guard let open = line.range(of: "FRIGATE("), let close = line[open.upperBound...].firstIndex(of: ")") else { continue }
            let patch = String(line[open.upperBound ..< close])
            guard !patch.isEmpty, patch.allSatisfy({ $0.isLowercase || $0.isNumber || $0 == "-" }) else { continue }
            var rest = String(line[line.index(after: close)...])
            guard rest.hasPrefix(":") else {
                found.append(Mark(patch: patch, path: path, line: index + 1, says: ""))
                continue
            }
            rest.removeFirst()
            var words = [rest]
            var ended = rest.contains("*/")
            var next = index + 1
            while !ended, next < lines.count {
                let following = lines[next].trimmingCharacters(in: .whitespaces)
                guard following.hasPrefix("//") || following.hasPrefix("*") else { break }
                let body = following.drop { $0 == "/" || $0 == "*" }.trimmingCharacters(in: .whitespaces)
                if body.isEmpty || following.hasPrefix("*/") { break }
                words.append(body)
                ended = following.contains("*/")
                next += 1
            }
            let says = words.joined(separator: " ").replacingOccurrences(of: "*/", with: "")
                .split(separator: " ", omittingEmptySubsequences: true).joined(separator: " ")
            found.append(Mark(patch: patch, path: path, line: index + 1, says: says))
        }
        return found
    }

    // MARK: - The list

    static func rendered(_ marks: [Mark]) -> String {
        var patches: [String: [Mark]] = [:]
        for mark in marks { patches[mark.patch, default: []].append(mark) }
        let files = Set(marks.map(\.path)).count
        var out = """
        # Patches to vendored MLX

        What this tree changes in the MLX it vendors (`ml-explore/mlx`, `mlx-c` and `mlx-swift`; the README's "Vendored
        sources" table names the tags). Every changed or added file carries a mark where the change is, `FRIGATE(<patch>)`,
        and this list is made from those marks: \(patches.count) patches in \(files) files.

        Written by `FRIGATE_WRITE_PATCH_LIST=1 swift test --filter VendoredPatchListTests`, never by hand; the same test,
        run without the variable, fails when the list and the tree disagree. After a sync with upstream, diff the vendored
        trees against the new tags and give every file that still differs a mark.

        """
        for patch in patches.keys.sorted() {
            out += "\n## \(patch)\n\n"
            for mark in patches[patch] ?? [] {
                out += "- `\(mark.path):\(mark.line)`" + (mark.says.isEmpty ? "" : " \(mark.says)") + "\n"
            }
        }
        return out
    }
}
