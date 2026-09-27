// Embeds a JSONL file ({"text": ...} per line) through FrigateEmbedder — the path
// Thread takes — and writes raw little-endian float32 rows.
//   embedprobe <spec> <query|document> <in.jsonl> <out.f32> [--dim N] [--max-tokens N]
// <spec> is what Profile.resolve takes: org/repo, org/repo@sha, or a directory.
import Foundation
import Frigate

/// Splits `--name N` options from the positional arguments.
func parse(_ raw: [String]) -> (positional: [String], options: [String: Int]) {
    var positional: [String] = []
    var options: [String: Int] = [:]
    var i = 0
    while i < raw.count {
        if raw[i].hasPrefix("--"), i + 1 < raw.count, let value = Int(raw[i + 1]) {
            options[raw[i]] = value
            i += 2
        } else {
            positional.append(raw[i])
            i += 1
        }
    }
    return (positional, options)
}
let (args, options) = parse(Array(CommandLine.arguments.dropFirst()))
let dim = options["--dim"]
let maxTokens = options["--max-tokens"]
guard args.count == 4, let role = FrigateEmbedder.Role(rawValue: args[1]) else {
    FileHandle.standardError.write(Data("usage: embedprobe <spec> <query|document> <in.jsonl> <out.f32> [--dim N] [--max-tokens N]\n".utf8))
    exit(2)
}
var profile = FrigateEmbedder.Profile.resolve(args[0])
if let dim { profile.outputDimension = dim }
if let maxTokens { profile.maxTokens = maxTokens }
let embedder = FrigateEmbedder(profile: profile)

let texts: [String] = try String(contentsOfFile: args[2], encoding: .utf8)
    .split(separator: "\n", omittingEmptySubsequences: true)
    .map { line in
        let object = try! JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
        return object["text"] as! String
    }
FileManager.default.createFile(atPath: args[3], contents: nil)
let out = try FileHandle(forWritingTo: URL(fileURLWithPath: args[3]))
let chunk = 64
var done = 0
var width = 0
let start = Date()
while done < texts.count {
    let slice = Array(texts[done..<min(done + chunk, texts.count)])
    let vectors = try await embedder.embed(slice, role: role)
    width = vectors.first?.count ?? width
    var flat = vectors.flatMap { $0 }
    out.write(Data(bytes: &flat, count: flat.count * MemoryLayout<Float>.size))
    done += slice.count
    if done % 1024 < chunk || done == texts.count {
        let rate = Double(done) / Date().timeIntervalSince(start)
        FileHandle.standardError.write(Data("\(done)/\(texts.count) dim=\(width) \(String(format: "%.1f", rate))/s\n".utf8))
    }
}
try out.close()
let status = embedder.status
print("dim=\(width) rows=\(texts.count) status=\(status.phase.rawValue) space=\(status.vectorSpace)")
