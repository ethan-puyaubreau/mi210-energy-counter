#include "amdsmi_provider.hpp"

#include <amd_smi/amdsmi.h>

#include <cstdio>

namespace KokkosTools {
namespace EnergyProfiler {

namespace {

bool check(amdsmi_status_t status, const char* what) {
  if (status == AMDSMI_STATUS_SUCCESS) return true;
  const char* text = nullptr;
  amdsmi_status_code_to_string(status, &text);
  std::fprintf(stderr, "AMDSMIProvider: %s failed: %s\n", what,
               text ? text : "unknown status");
  return false;
}

}  // namespace

AMDSMIProvider::~AMDSMIProvider() { finalize(); }

bool AMDSMIProvider::initialize() {
  if (is_initialized_) return true;
  if (!check(amdsmi_init(AMDSMI_INIT_AMD_GPUS), "amdsmi_init")) return false;
  is_initialized_ = true;

  uint32_t socket_count = 0;
  if (!check(amdsmi_get_socket_handles(&socket_count, nullptr),
             "amdsmi_get_socket_handles")) {
    finalize();
    return false;
  }
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  if (!check(amdsmi_get_socket_handles(&socket_count, sockets.data()),
             "amdsmi_get_socket_handles")) {
    finalize();
    return false;
  }

  for (amdsmi_socket_handle socket : sockets) {
    uint32_t count = 0;
    if (!check(amdsmi_get_processor_handles(socket, &count, nullptr),
               "amdsmi_get_processor_handles"))
      continue;
    std::vector<amdsmi_processor_handle> handles(count);
    if (!check(amdsmi_get_processor_handles(socket, &count, handles.data()),
               "amdsmi_get_processor_handles"))
      continue;
    for (amdsmi_processor_handle handle : handles) {
      amdsmi_bdf_t bdf;
      if (!check(amdsmi_get_gpu_device_bdf(handle, &bdf),
                 "amdsmi_get_gpu_device_bdf"))
        continue;
      char name[32];
      std::snprintf(
          name, sizeof name, "%04x:%02x:%02x.%x",
          (unsigned)bdf.bdf.domain_number, (unsigned)bdf.bdf.bus_number,
          (unsigned)bdf.bdf.device_number, (unsigned)bdf.bdf.function_number);
      uint64_t energy = 0, stamp = 0;
      float resolution = 0.0f;
      if (!check(amdsmi_get_energy_count(handle, &energy, &resolution, &stamp),
                 "amdsmi_get_energy_count"))
        continue;
      devices_.push_back({handle, name, resolution});
    }
  }

  if (devices_.empty()) {
    std::fprintf(stderr,
                 "AMDSMIProvider: no GPU with a readable energy counter\n");
    finalize();
    return false;
  }
  return true;
}

void AMDSMIProvider::finalize() {
  if (!is_initialized_) return;
  devices_.clear();
  amdsmi_shut_down();
  is_initialized_ = false;
}

std::string AMDSMIProvider::get_device_name(size_t device_index) const {
  return device_index < devices_.size() ? devices_[device_index].bdf : "";
}

double AMDSMIProvider::get_counter_resolution_uj(size_t device_index) const {
  return device_index < devices_.size() ? devices_[device_index].resolution_uj
                                        : 0.0;
}

bool AMDSMIProvider::get_device_energy(size_t device_index,
                                       double& energy_joules,
                                       uint64_t& firmware_timestamp_ns) {
  if (device_index >= devices_.size()) return false;
  uint64_t energy        = 0;
  float resolution       = 0.0f;
  amdsmi_status_t status = amdsmi_get_energy_count(
      static_cast<amdsmi_processor_handle>(devices_[device_index].handle),
      &energy, &resolution, &firmware_timestamp_ns);
  if (status != AMDSMI_STATUS_SUCCESS) return false;
  energy_joules = static_cast<double>(energy) * resolution * 1e-6;
  return true;
}

}  // namespace EnergyProfiler
}  // namespace KokkosTools
