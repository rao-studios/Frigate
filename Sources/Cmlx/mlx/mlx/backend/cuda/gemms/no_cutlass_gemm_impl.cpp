// Copyright © 2026 Apple Inc.

#include <stdexcept>
#include "mlx/array.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/grouped_gemm.h"

namespace mlx::core {

void cutlass_grouped_gemm_unaligned(
    bool,
    int,
    bool,
    int,
    int,
    const array&,
    const array&,
    const array&,
    array&,
    cu::CommandEncoder&) {
  throw std::runtime_error(
      "[cutlass_grouped_gemm_unaligned] Requires CUTLASS (not available on this build).");
}

void cutlass_segmented_mm(
    bool,
    int,
    bool,
    int,
    int,
    int,
    int,
    const array&,
    const array&,
    const array&,
    array&,
    cu::CommandEncoder&) {
  throw std::runtime_error(
      "[cutlass_segmented_mm] Requires CUTLASS (not available on this build).");
}

} // namespace mlx::core
