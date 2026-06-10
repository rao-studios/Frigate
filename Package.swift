// swift-tools-version: 6.3;(experimentalCGen)
// Frigate — fully vendored MLX Swift package for embedding and LLM inference.
// All fork sources are bundled; only three upstream Apple packages are external.

import PackageDescription

let noMetalCmlxExcludes = [
    "mlx/mlx/backend/metal/allocator.cpp",
    "mlx/mlx/backend/metal/binary.cpp",
    "mlx/mlx/backend/metal/compiled.cpp",
    "mlx/mlx/backend/metal/conv.cpp",
    "mlx/mlx/backend/metal/copy.cpp",
    "mlx/mlx/backend/metal/custom_kernel.cpp",
    "mlx/mlx/backend/metal/device.cpp",
    "mlx/mlx/backend/metal/device_info.cpp",
    "mlx/mlx/backend/metal/distributed.cpp",
    "mlx/mlx/backend/metal/eval.cpp",
    "mlx/mlx/backend/metal/event.cpp",
    "mlx/mlx/backend/metal/fence.cpp",
    "mlx/mlx/backend/metal/fft.cpp",
    "mlx/mlx/backend/metal/hadamard.cpp",
    "mlx/mlx/backend/metal/indexing.cpp",
    "mlx/mlx/backend/metal/jit_kernels.cpp",
    "mlx/mlx/backend/metal/logsumexp.cpp",
    "mlx/mlx/backend/metal/matmul.cpp",
    "mlx/mlx/backend/metal/metal.cpp",
    "mlx/mlx/backend/metal/normalization.cpp",
    "mlx/mlx/backend/metal/primitives.cpp",
    "mlx/mlx/backend/metal/quantized.cpp",
    "mlx/mlx/backend/metal/reduce.cpp",
    "mlx/mlx/backend/metal/resident.cpp",
    "mlx/mlx/backend/metal/rope.cpp",
    "mlx/mlx/backend/metal/scaled_dot_product_attention.cpp",
    "mlx/mlx/backend/metal/scan.cpp",
    "mlx/mlx/backend/metal/slicing.cpp",
    "mlx/mlx/backend/metal/softmax.cpp",
    "mlx/mlx/backend/metal/sort.cpp",
    "mlx/mlx/backend/metal/ternary.cpp",
    "mlx/mlx/backend/metal/unary.cpp",
    "mlx/mlx/backend/metal/utils.cpp",
    "mlx/mlx/backend/metal/kernels",
    "mlx/mlx/backend/metal/jit",
]

let noCudaCmlxExcludes = [
    "mlx/mlx/backend/cuda/allocator.cpp",
    "mlx/mlx/backend/cuda/compiled.cpp",
    "mlx/mlx/backend/cuda/conv.cpp",
    "mlx/mlx/backend/cuda/cublas_utils.cpp",
    "mlx/mlx/backend/cuda/cudnn_utils.cpp",
    "mlx/mlx/backend/cuda/custom_kernel.cpp",
    "mlx/mlx/backend/cuda/delayload.cpp",
    "mlx/mlx/backend/cuda/device_info.cpp",
    "mlx/mlx/backend/cuda/device.cpp",
    "mlx/mlx/backend/cuda/eval.cpp",
    "mlx/mlx/backend/cuda/fence.cpp",
    "mlx/mlx/backend/cuda/indexing.cpp",
    "mlx/mlx/backend/cuda/jit_module.cpp",
    "mlx/mlx/backend/cuda/load.cpp",
    "mlx/mlx/backend/cuda/matmul.cpp",
    "mlx/mlx/backend/cuda/primitives.cpp",
    "mlx/mlx/backend/cuda/scaled_dot_product_attention.cpp",
    "mlx/mlx/backend/cuda/slicing.cpp",
    "mlx/mlx/backend/cuda/utils.cpp",
    "mlx/mlx/backend/cuda/worker.cpp",
    "mlx/mlx/backend/cuda/binary",
    "mlx/mlx/backend/cuda/conv",
    "mlx/mlx/backend/cuda/copy",
    "mlx/mlx/backend/cuda/device",
    "mlx/mlx/backend/cuda/gemms",
    "mlx/mlx/backend/cuda/quantized",
    "mlx/mlx/backend/cuda/reduce",
    "mlx/mlx/backend/cuda/steel",
    "mlx/mlx/backend/cuda/unary",
]

#if os(Linux)
    let platformExcludes: [String]
    let cxxSettings: [CXXSetting]
    let linkerSettings: [LinkerSetting]
    let mlxSwiftExcludes: [String]

    if Context.environment["SPM_CUDA"] != "0" {
        // Linux with CUDA

        platformExcludes =
            [
                "framework",
                "include-framework",
                "metal-cpp",

                "mlx/mlx/backend/no_gpu",
                "mlx/mlx/backend/cuda/no_cuda.cpp",
                "mlx/mlx/backend/cuda/gemms/cublas_gemm_batched_12_0.cpp",
                "mlx/mlx/backend/no_cpu",
                "mlx/mlx/backend/cpu/gemms/bnns.cpp",
                "mlx-conditional",
                "mlx-c/mlx/c/metal.cpp",

                "mlx/mlx/backend/cuda/delayload.cpp",
                "mlx/mlx/backend/cuda/quantized/qmm/qmm_impl_sm90_m128_n16_m1.cu",
                "mlx/mlx/backend/cuda/quantized/qmm/qmv.cu",
                "mlx/mlx/backend/cuda/quantized/qmm/qmm_impl_sm90_m128_n32_m1.cu",
                "mlx/mlx/backend/cuda/quantized/qmm/qmm_impl_sm90.cuh",
                "mlx/mlx/backend/cuda/quantized/qmm/qmm_impl_sm90_m128_n64_m2.cu",
                "mlx/mlx/backend/cuda/quantized/qmm/qmm.h",
                "mlx/mlx/backend/cuda/quantized/qmm/qmm_impl_sm90_m128_n256_m2.cu",
                "mlx/mlx/backend/cuda/quantized/qmm/qmm.cu",
                "mlx/mlx/backend/cuda/quantized/qmm/qmm_impl_sm90_m128_n128_m2.cu",
                "mlx/mlx/backend/cuda/quantized/qmm/fp_qmv.cu",

                // GPU fallback replacements vendored — exclude CUTLASS-dependent impls
                "mlx/mlx/backend/cuda/quantized/qqmm_impl.cpp",
                "mlx/mlx/backend/cuda/quantized/cublas_qqmm.cpp",
            ] + noMetalCmlxExcludes

        cxxSettings = [
            .unsafeFlags(["-I/usr/local/cuda/include"]),
            .unsafeFlags(["-I/usr/local/cuda/include/cccl"]),
            .unsafeFlags(["-I/usr/local/cudnn-frontend/include"]),
            .unsafeFlags(["-DCUDNN_FRONTEND_SKIP_JSON_LIB"]),
        ]

        linkerSettings = [
            .linkedLibrary("gfortran", .when(platforms: [.linux])),
            .linkedLibrary("blas", .when(platforms: [.linux])),
            .linkedLibrary("lapack", .when(platforms: [.linux])),
            .linkedLibrary("openblas", .when(platforms: [.linux])),
            .unsafeFlags(["-L/usr/local/cuda/lib64"]),
            .unsafeFlags(["-L/usr/local/cuda/lib64/stubs"]),
            .linkedLibrary("cudnn"),
            .linkedLibrary("cublas"),
            .linkedLibrary("cublasLt"),
            .linkedLibrary("nvrtc"),
            .linkedLibrary("cudart"),
            .linkedLibrary("cuda"),
        ]

        mlxSwiftExcludes = [
            "GPU+Metal.swift",
            "MLXArray+Metal.swift",
        ]
    } else {
        // Linux CPU-only

        platformExcludes =
            [
                "framework",
                "include-framework",
                "metal-cpp",

                "mlx/mlx/backend/gpu",
                "mlx/mlx/backend/no_cpu",
                "mlx/mlx/backend/cpu/gemms/bnns.cpp",
                "mlx-conditional",
                "mlx-c/mlx/c/metal.cpp",

                "mlx-c/mlx/c/fast.cpp",
            ] + noMetalCmlxExcludes + noCudaCmlxExcludes

        cxxSettings = []

        linkerSettings = [
            .linkedLibrary("gfortran", .when(platforms: [.linux])),
            .linkedLibrary("blas", .when(platforms: [.linux])),
            .linkedLibrary("lapack", .when(platforms: [.linux])),
            .linkedLibrary("openblas", .when(platforms: [.linux])),
        ]

        mlxSwiftExcludes = [
            "GPU+Metal.swift",
            "GPU+CUDA.swift",
            "MLXArray+Metal.swift",
            "MLXFast.swift",
            "MLXFastKernel.swift",
        ]
    }
#else
    // Apple platforms (Metal)

    let platformExcludes: [String] =
        [
            "mlx/mlx/backend/cpu/compiled.cpp",

            "mlx/mlx/backend/no_gpu",
            "mlx/mlx/backend/no_cpu",
            "mlx/mlx/backend/metal/no_metal.cpp",

            "mlx/mlx/backend/cpu/gemms/simd_fp16.cpp",
            "mlx/mlx/backend/cpu/gemms/simd_bf16.cpp",
        ] + noCudaCmlxExcludes

    let cxxSettings: [CXXSetting] = [
        .headerSearchPath("metal-cpp"),

        .define("MLX_USE_ACCELERATE"),
        .define("ACCELERATE_NEW_LAPACK"),
        .define("_METAL_"),
        .define("SWIFTPM_BUNDLE", to: "\"mlx-swift_Cmlx\""),
        .define("METAL_PATH", to: "\"default.metallib\""),
    ]

    let linkerSettings: [LinkerSetting] = [
        .linkedFramework("Foundation"),
        .linkedFramework("Metal"),
        .linkedFramework("Accelerate"),
    ]

    let mlxSwiftExcludes: [String] = [
        "GPU+CUDA.swift"
    ]
#endif

// MLXVLM uses AVFoundation, CoreImage, CoreGraphics — macOS/iOS only.
// On Linux exclude the model files and media processing; keep VLMModel.swift (protocol).
#if os(Linux)
    let vlmExcludes: [String] = ["README.md", "MediaProcessing.swift", "Models", "VLMModelFactory.swift"]
#else
    let vlmExcludes: [String] = ["README.md"]
#endif

let cmlx = Target.target(
    name: "Cmlx",
    path: "Sources/Cmlx",
    exclude: platformExcludes + [
        "vendor-README.md",

        "mlx-c/examples",
        "mlx-c/mlx/c/distributed.cpp",
        "mlx-c/mlx/c/distributed_group.cpp",

        "json",

        "fmt/test",
        "fmt/doc",
        "fmt/support",
        "fmt/src/os.cc",
        "fmt/src/fmt.cc",

        "mlx/mlx/backend/no_cpu/compiled.cpp",

        "mlx/ACKNOWLEDGMENTS.md",
        "mlx/CMakeLists.txt",
        "mlx/CODE_OF_CONDUCT.md",
        "mlx/CONTRIBUTING.md",
        "mlx/LICENSE",
        "mlx/MANIFEST.in",
        "mlx/README.md",
        "mlx/benchmarks",
        "mlx/cmake",
        "mlx/docs",
        "mlx/examples",
        "mlx/mlx.pc.in",
        "mlx/pyproject.toml",
        "mlx/python",
        "mlx/setup.py",
        "mlx/tests",

        "mlx/mlx/io/no_safetensors.cpp",
        "mlx/mlx/io/gguf.cpp",
        "mlx/mlx/io/gguf_quants.cpp",

        "mlx/mlx/backend/metal/kernels",
        "mlx/mlx/backend/metal/nojit_kernels.cpp",

        "mlx/mlx/distributed/mpi/mpi.cpp",
        "mlx/mlx/distributed/ring/ring.cpp",
        "mlx/mlx/distributed/nccl/nccl.cpp",
        "mlx/mlx/distributed/nccl/nccl_stub",
        "mlx/mlx/distributed/jaccl/jaccl.cpp",
        "mlx/mlx/distributed/jaccl/mesh.cpp",
        "mlx/mlx/distributed/jaccl/ring.cpp",
        "mlx/mlx/distributed/jaccl/utils.cpp",
    ],
    cSettings: [
        .headerSearchPath("mlx"),
        .headerSearchPath("mlx-c"),
        .headerSearchPath("mlx-generated/cuda"),
    ],
    cxxSettings: cxxSettings + [
        .headerSearchPath("mlx"),
        .headerSearchPath("mlx-c"),
        .headerSearchPath("json/single_include/nlohmann"),
        .headerSearchPath("fmt/include"),
        .define("MLX_VERSION", to: "\"0.31.1\""),
    ],
    linkerSettings: linkerSettings,
    plugins: [
        .plugin(name: "CudaBuild")
    ]
)

let package = Package(
    name: "Frigate",

    platforms: [
        .macOS("14.0"),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],

    products: [
        // Main Frigate API
        .library(name: "Frigate", targets: ["Frigate"]),

        // Linux-compatible Accelerate ops backed by MLX
        .library(name: "MLXAccelerate", targets: ["MLXAccelerate"]),

        // Re-exported MLX stack
        .library(name: "MLX", targets: ["MLX"]),
        .library(name: "MLXRandom", targets: ["MLXRandom"]),
        .library(name: "MLXNN", targets: ["MLXNN"]),
        .library(name: "MLXOptimizers", targets: ["MLXOptimizers"]),
        .library(name: "MLXFFT", targets: ["MLXFFT"]),
        .library(name: "MLXLinalg", targets: ["MLXLinalg"]),
        .library(name: "MLXFast", targets: ["MLXFast"]),

        // Transformers stack
        .library(name: "Hub", targets: ["Hub"]),
        .library(name: "Tokenizers", targets: ["Tokenizers"]),
        .library(name: "Transformers", targets: ["Tokenizers", "Generation", "Models"]),

        // LLM / embedding models
        .library(name: "MLXLLM", targets: ["MLXLLM"]),
        .library(name: "MLXVLM", targets: ["MLXVLM"]),
        .library(name: "MLXLMCommon", targets: ["MLXLMCommon"]),
        .library(name: "MLXEmbedders", targets: ["MLXEmbedders"]),
        .library(name: "mlx_embeddings", targets: ["mlx_embeddings"]),
    ],

    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-crypto.git", from: "4.2.0"),
    ],

    targets: [
        // ── C++ core ──────────────────────────────────────────────────────────
        cmlx,

        // CUDA build tool (plugin dependency)
        .executableTarget(
            name: "encuda",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .plugin(
            name: "CudaBuild",
            capability: .buildTool(),
            dependencies: [
                .target(name: "encuda")
            ]
        ),

        // ── MLX Swift wrappers ────────────────────────────────────────────────
        .target(
            name: "MLX",
            dependencies: [
                "Cmlx",
                .product(name: "Numerics", package: "swift-numerics"),
            ],
            exclude: mlxSwiftExcludes,
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXRandom",
            dependencies: ["MLX"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXFast",
            dependencies: ["MLX", "Cmlx"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXNN",
            dependencies: ["MLX"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXOptimizers",
            dependencies: ["MLX", "MLXNN"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXFFT",
            dependencies: ["MLX"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXLinalg",
            dependencies: ["MLX"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),

        // ── Jinja (swift-jinja vendored) ──────────────────────────────────────
        .target(
            name: "Jinja",
            dependencies: [
                .product(name: "OrderedCollections", package: "swift-collections")
            ]
        ),

        // ── Transformers stack (swift-transformers vendored) ──────────────────
        .target(
            name: "Hub",
            dependencies: [
                "Jinja",
                .product(name: "OrderedCollections", package: "swift-collections"),
                .product(name: "Crypto", package: "swift-crypto"),
            ],
            resources: [
                .process("Resources")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "Tokenizers",
            dependencies: ["Hub", "Jinja"]
        ),
        .target(
            name: "Generation",
            dependencies: ["Tokenizers"]
        ),
        .target(
            name: "Models",
            dependencies: ["Tokenizers", "Generation"]
        ),

        // ── LLM stack (mlx-swift-lm vendored) ────────────────────────────────
        .target(
            name: "MLXLMCommon",
            dependencies: [
                "MLX", "MLXNN", "MLXOptimizers",
                "Tokenizers", "Generation", "Models",
            ],
            exclude: ["README.md"],
            swiftSettings: [
                // mlx-swift-lm was authored against swift-tools-version:5.12;
                // compiling in Swift 6 mode surfaced Sendable errors. Keep v5
                // compatibility to avoid patching upstream source.
                .swiftLanguageMode(.v5)
            ]
        ),
        .target(
            name: "MLXLLM",
            dependencies: [
                "MLXLMCommon", "MLX", "MLXNN", "MLXOptimizers",
                "Tokenizers", "Generation", "Models",
            ],
            exclude: ["README.md"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXVLM",
            dependencies: [
                "MLXLMCommon", "MLX", "MLXNN", "MLXOptimizers",
                "Tokenizers", "Generation", "Models",
            ],
            exclude: vlmExcludes,
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXEmbedders",
            dependencies: [
                "MLX", "MLXNN", "MLXLMCommon",
                "Tokenizers", "Generation", "Models",
            ],
            exclude: ["README.md"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),

        // ── mlx_embeddings (mlx.embeddings vendored) ──────────────────────────
        .target(
            name: "mlx_embeddings",
            dependencies: [
                "MLX", "MLXFast", "MLXNN", "MLXOptimizers",
                "MLXRandom", "MLXLinalg", "MLXLMCommon",
                "Tokenizers", "Generation", "Models",
            ]
        ),

        // ── MLXAccelerate — Linux-compatible Accelerate ops via MLX ─────────
        .target(
            name: "MLXAccelerate",
            dependencies: ["MLX"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),

        // ── Frigate public API ────────────────────────────────────────────────
        .target(
            name: "Frigate",
            dependencies: [
                "MLX", "MLXNN", "Tokenizers",
                "MLXLMCommon", "MLXLLM", "mlx_embeddings",
                "MLXAccelerate",
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),

        // ── Tests ─────────────────────────────────────────────────────────────
        .testTarget(
            name: "FrigateTests",
            dependencies: ["Frigate", "MLXAccelerate"]
        ),
    ],

    cxxLanguageStandard: .gnucxx20
)
