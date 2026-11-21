/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/metal.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" int mlx_metal_get_metallib_path(mlx_string* res) {
  try {
    mlx_string_set_(*res, mlx::core::metal::get_metallib_path());
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_is_available(bool* res) {
  try {
    *res = mlx::core::metal::is_available();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_set_metallib_path(const char* path) {
  try {
    mlx::core::metal::set_metallib_path(std::string(path));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_start_capture(const char* path) {
  try {
    mlx::core::metal::start_capture(std::string(path));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_stop_capture(void) {
  try {
    mlx::core::metal::stop_capture();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// FRIGATE: the scheduling an engine sets around a decode step. Not generated.
namespace mlx::core {
void set_max_active_tasks(int tasks);
int max_active_tasks_override_value();
#if defined(__APPLE__)
namespace metal {
void set_command_buffer_limits(int ops, int megabytes);
std::pair<int, int> command_buffer_limits();
} // namespace metal
#endif
} // namespace mlx::core

extern "C" int mlx_set_gpu_scheduling(int ops, int megabytes, int in_flight) {
#if defined(__APPLE__)
  mlx::core::metal::set_command_buffer_limits(ops, megabytes);
#endif
  mlx::core::set_max_active_tasks(in_flight);
  return 0;
}
extern "C" int mlx_gpu_scheduling(int* ops, int* megabytes, int* in_flight) {
#if defined(__APPLE__)
  auto [o, m] = mlx::core::metal::command_buffer_limits();
  *ops = o;
  *megabytes = m;
#else
  *ops = 0;
  *megabytes = 0;
#endif
  *in_flight = mlx::core::max_active_tasks_override_value();
  return 0;
}
