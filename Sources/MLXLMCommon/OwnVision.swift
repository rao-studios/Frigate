//  WHAT: A loaded language model reading a picture with the vision tower its own checkpoint carries, instead of a
//        second model loaded for vision. Each family that can do it offers one (Inkling2OwnVision in MLXLLM,
//        Qwen35OwnVision in MLXVLM), built over the text model already loaded: its weights are shared, never copied,
//        and only the tower's tensors are read from the checkpoint.
//  OUT:  The host (Craft) asks each family in turn and falls back to its own vision model when none answers.
//  PIN:  THE FORWARD IS HANDED BACK, NOT RUN. Preparing a picture (decoding, patching, the prompt's tokens) touches no
//        model; reading it runs the loaded model, so it must run where nothing else is running one: under the host's
//        generation gate, or on the parallel engine's thread between two passes (`ParallelEngine.exclusive`).

import CoreGraphics
import Foundation

public protocol OwnVision: AnyObject, Sendable {
    /// `image` and `question` made ready for the model, and the read that follows: the tower over the picture, then the
    /// language model's answer, at most `maxTokens` tokens, greedy. The returned closure is the only part that runs
    /// the model.
    func reader(image: CGImage, question: String, maxTokens: Int) async throws -> @Sendable () throws -> String
}

public enum OwnVisionCheckpoint {
    /// The safetensors files in `directory` holding tensors under `prefix`, by the checkpoint's index, or every file
    /// when it has none. Empty when the index names none: the checkpoint carries no tower.
    public static func files(in directory: URL, prefix: String) throws -> [URL] {
        let index = directory.appending(component: "model.safetensors.index.json")
        if let data = try? Data(contentsOf: index),
           let map = (try JSONSerialization.jsonObject(with: data) as? [String: Any])?["weight_map"] as? [String: String] {
            return Set(map.filter { $0.key.hasPrefix(prefix) }.values).sorted().map { directory.appending(component: $0) }
        }
        return try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
    }
}

public enum OwnVisionError: LocalizedError {
    case noPicture
    case placeholder(String)

    public var errorDescription: String? {
        switch self {
        case .noPicture: "The picture could not be read into pixels."
        case .placeholder(let what): "The prompt has no place for the picture: \(what)."
        }
    }
}
