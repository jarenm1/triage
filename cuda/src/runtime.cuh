#pragma once

#include <cstddef>
#include <string>

namespace sim_cuda {

struct RuntimeInfo {
  int device_ordinal;
  std::string device_name;
  int compute_major;
  int compute_minor;
  std::size_t global_memory_bytes;
};

RuntimeInfo probe_runtime(int device_ordinal = 0);

}  // namespace sim_cuda
