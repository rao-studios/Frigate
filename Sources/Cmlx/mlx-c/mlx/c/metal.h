/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_METAL_H
#define MLX_METAL_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup metal Metal specific operations
 */
/**@{*/

int mlx_metal_get_metallib_path(mlx_string* res);
int mlx_metal_is_available(bool* res);
int mlx_metal_set_metallib_path(const char* path);
int mlx_metal_start_capture(const char* path);
int mlx_metal_stop_capture(void);

/* FRIGATE: how much work goes in one command buffer before it is committed
 * (`ops` kernels or `megabytes` of inputs; 0 = the device's own) and how many
 * committed buffers may be in flight before evaluation waits (0 = MLX's 10). */
int mlx_set_gpu_scheduling(int ops, int megabytes, int in_flight);
/* The overrides set, 0 where none is. */
int mlx_gpu_scheduling(int* ops, int* megabytes, int* in_flight);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
