# Patches to vendored MLX

What this tree changes in the MLX it vendors (`ml-explore/mlx`, `mlx-c` and `mlx-swift`; the README's "Vendored
sources" table names the tags). Every changed or added file carries a mark where the change is, `FRIGATE(<patch>)`,
and this list is made from those marks: 8 patches in 22 files.

Written by `FRIGATE_WRITE_PATCH_LIST=1 swift test --filter VendoredPatchListTests`, never by hand; the same test,
run without the variable, fails when the list and the tree disagree. After a sync with upstream, diff the vendored
trees against the new tags and give every file that still differs a mark.

## cuda-affine-fallback

- `Sources/Cmlx/mlx/mlx/backend/cuda/quantized/quantized.cpp:139` affine quantization on a GPU with no CUTLASS kernels: dequantize the weights, then cuBLAS GEMM. Used on GPUs without CUTLASS support (e.g. sm_86 / RTX 30xx).

## cuda-half-math

- `Sources/Cmlx/mlx/mlx/backend/cuda/device/binary_ops.cuh:7` half and bfloat16 go through float in the ops below (is_half_v, unary_ops.cuh).
- `Sources/Cmlx/mlx/mlx/backend/cuda/device/unary_ops.cuh:14` true for __half and __nv_bfloat16. cuda::std:: math ops are ambiguous for these types when GCC's cmath overloads are in scope, so each op below and in binary_ops.cuh takes them through float.

## cuda-no-cutlass

- `Sources/Cmlx/mlx/mlx/backend/cuda/gemms/no_cutlass_gemm_impl.cpp:2` added. Stands in for grouped_gemm_unaligned.cu, left out with its CUTLASS dependency.
- `Sources/Cmlx/mlx/mlx/backend/cuda/quantized/no_fp_quantize_impl.cpp:2` added. Stands in for fp_quantize.cu and qqmm_utils.cu, left out with their CUTLASS dependency.
- `Sources/Cmlx/mlx/mlx/backend/cuda/quantized/no_qmm_impl.cpp:2` added. Stubs for the qmm/ kernels, left out with their CUTLASS dependency: Sources/Cmlx/CudaBuild.json excludes them and the two sources beside them (a JSON file carries no mark of its own).

## cuda-sdpa-cache

- `Sources/Cmlx/mlx/mlx/backend/cuda/scaled_dot_product_attention.cpp:178` 2048 graphs by default, where upstream keeps 256: sub-batches of varying sequence length missed 512 times and ended in a fatal error.

## gpu-scheduling

- `Sources/Cmlx/include-framework/mlx-c-metal.h:37` not generated. How much work goes in one command buffer before it is committed (`ops` kernels or `megabytes` of inputs; 0 = the device's own) and how many committed buffers may be in flight before evaluation waits (0 = MLX's 10).
- `Sources/Cmlx/include/mlx/c/metal.h:37` not generated. How much work goes in one command buffer before it is committed (`ops` kernels or `megabytes` of inputs; 0 = the device's own) and how many committed buffers may be in flight before evaluation waits (0 = MLX's 10).
- `Sources/Cmlx/mlx-c/mlx/c/metal.cpp:57` the scheduling an engine sets around a decode step. Not generated, though the file says it is: the binding sits beside the generated ones so the same header declares it.
- `Sources/Cmlx/mlx-c/mlx/c/metal.h:37` not generated. How much work goes in one command buffer before it is committed (`ops` kernels or `megabytes` of inputs; 0 = the device's own) and how many committed buffers may be in flight before evaluation waits (0 = MLX's 10).
- `Sources/Cmlx/mlx/mlx/backend/metal/device.cpp:538` the commit limits an engine sets for a while (0 = the device's own, from its architecture or MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER).
- `Sources/Cmlx/mlx/mlx/transforms.cpp:26` how many committed command buffers may be in flight before an evaluation waits for one (MLX's own: 10). A runtime value, so an engine can let a decode step's small buffers run deeper than a prefill's large ones.
- `Sources/MLX/GPU+Scheduling.swift:12` added, not in upstream mlx-swift. The vendored MLX reads the overrides (device.cpp, transforms.cpp) and mlx-c's metal.cpp binds them.

## kernel-source-once

- `Sources/Cmlx/mlx/mlx/backend/common/metal_kernel.cpp:275` the generated source and its hash, once per kernel name. The name carries the template, the dtypes and how each input is passed, and this function's own source, header and names are fixed, so a name always maps to the same source. Writing it out on every call cost a decoded token milliseconds (Inkling-2 makes 456 calls a token, 2026-11-21).
- `Sources/Cmlx/mlx/mlx/backend/metal/custom_kernel.cpp:50` the hash its maker took, where it took one.
- `Sources/Cmlx/mlx/mlx/fast_primitives.h:576` std::hash of source_, when the maker knew it (0 = not known): a kernel's source is thousands of characters and was hashed on every call.

## saved-in-key-order

- `Sources/Cmlx/mlx/mlx/io/gguf.cpp:440` write the metadata, the tensor infos and the tensor data in key order. The maps iterate in their insertion order, which follows the caller's (Swift seeds its dictionary order per process), so the same contents saved twice gave two different files. As in safetensors.cpp. Not built in Frigate: gguf.cpp is excluded in favour of no_gguf.cpp.
- `Sources/Cmlx/mlx/mlx/io/safetensors.cpp:256` lay the tensors out in key order. The map iterates in its insertion order, which follows the caller's (Swift seeds its dictionary order per process), so the same tensors saved twice gave two different files. In key order a file's bytes depend only on its contents.

## sort-rows

- `Sources/Cmlx/mlx-generated/metal/sort.h:582` a row's partitions are n_blocks + 1 apart, as the merge kernel reads them. They were offset by the threadgroup's width, min(n_blocks + 1, 1024), so past 1,023 blocks the rows of one call overlapped. A row's offset is 64-bit here and in the kernels beside this one.
- `Sources/Cmlx/mlx-generated/sort.cpp:595` a row's partitions are n_blocks + 1 apart, as the merge kernel reads them. They were offset by the threadgroup's width, min(n_blocks + 1, 1024), so past 1,023 blocks the rows of one call overlapped. A row's offset is 64-bit here and in the kernels beside this one.
- `Sources/Cmlx/mlx/mlx/backend/metal/kernels/sort.h:582` a row's partitions are n_blocks + 1 apart, as the merge kernel reads them. They were offset by the threadgroup's width, min(n_blocks + 1, 1024), so past 1,023 blocks the rows of one call overlapped. A row's offset is 64-bit here and in the kernels beside this one.
