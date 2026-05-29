// Copyright © 2026 Apple Inc.
// Stubs for qmm/ CUDA kernels excluded due to missing CUTLASS dependency.

#include <optional>
#include <stdexcept>
#include "mlx/backend/cuda/quantized/qmm/qmm.h"

namespace mlx::core {

bool supports_qmm_sm90(
    const array&, const array&, const array&,
    const std::optional<array>&, const array&,
    bool, int, int, QuantizationMode, cu::Device&) {
  return false;
}

void qmm_sm90(
    const array&, const array&, const array&, const array&,
    array&, int, int, cu::CommandEncoder&, Stream) {
  throw std::runtime_error(
      "[qmm_sm90] Requires CUTLASS / sm90 (not available on this build).");
}

bool supports_fp_qmv(
    const array&, const array&, const array&,
    const std::optional<array>&, const array&,
    bool, int, int, QuantizationMode, cu::Device&) {
  return false;
}

void fp_qmv(
    const array&, const array&, const array&,
    array&, int, int, cu::CommandEncoder&, Stream) {
  throw std::runtime_error(
      "[fp_qmv] Requires CUTLASS (not available on this build).");
}

bool supports_qmv(
    const array&, const array&, const array&,
    const std::optional<array>&, const array&,
    bool, int, int, QuantizationMode, cu::Device&) {
  return false;
}

void qmv(
    const array&, const array&, const array&,
    const std::optional<array>&,
    array&, int, int, QuantizationMode, cu::CommandEncoder&) {
  throw std::runtime_error(
      "[qmv] Requires CUTLASS (not available on this build).");
}

} // namespace mlx::core
