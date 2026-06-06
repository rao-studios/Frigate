// Frigate — fully vendored MLX Swift package.
//
// Embed text:  FrigateEmbedder
// Run LLMs:    FrigateLLM
//
// All MLX fork sources (mlx-swift, swift-transformers, mlx-swift-lm, mlx.embeddings,
// swift-jinja) are vendored directly in Sources/. No fork URLs appear in Package.swift.
// Only three upstream Apple packages remain as external dependencies:
//   - swift-numerics   (Complex type for MLX)
//   - swift-collections (OrderedCollections for Hub/Jinja)
//   - swift-crypto      (Crypto for Hub downloads)

@_exported import MLX
@_exported import MLXNN
@_exported import MLXLMCommon
@_exported import MLXAccelerate
