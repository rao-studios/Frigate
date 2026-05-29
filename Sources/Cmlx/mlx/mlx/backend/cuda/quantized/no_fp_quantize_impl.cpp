// Copyright © 2026 Apple Inc.

#include <optional>
#include <stdexcept>
#include "mlx/array.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/quantized/qqmm_utils.h"

namespace mlx::core {

void fp_quantize(
    const array&,
    array&,
    array&,
    int,
    int,
    const std::optional<array>&,
    cu::CommandEncoder&,
    const Stream&) {
  throw std::runtime_error(
      "[fp_quantize] FP8/FP4 quantization requires CUTLASS (not available on this build).");
}

void fp_dequantize(
    const array&,
    const array&,
    array&,
    int,
    int,
    const std::optional<array>&,
    cu::CommandEncoder&,
    const Stream&) {
  throw std::runtime_error(
      "[fp_dequantize] FP8/FP4 dequantization requires CUTLASS (not available on this build).");
}

void fp_quantize_dequantize(
    const array&,
    array&,
    int,
    int,
    const std::optional<array>&,
    cu::CommandEncoder&,
    const Stream&) {
  throw std::runtime_error(
      "[fp_quantize_dequantize] FP8/FP4 quantize-dequantize requires CUTLASS (not available on this build).");
}

void swizzle_scales(
    const array&,
    array&,
    cu::CommandEncoder&,
    const Stream&) {
  throw std::runtime_error(
      "[swizzle_scales] Requires CUTLASS (not available on this build).");
}

void compute_qqmm_pointers(
    array&,
    array&,
    const array&,
    const array&,
    cu::CommandEncoder&) {
  throw std::runtime_error(
      "[compute_qqmm_pointers] Requires CUTLASS (not available on this build).");
}

} // namespace mlx::core
