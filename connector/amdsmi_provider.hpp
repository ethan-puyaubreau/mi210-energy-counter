#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace KokkosTools {
namespace EnergyProfiler {

/**
 * @brief Reads the hardware energy counter of AMD GPUs through amdsmi.
 *
 * Only this class talks to amdsmi, so tests can swap the library for a fake
 * one. Layout follows VariorumProvider (kokkos-tools PR #302) so it can move
 * to provider/provider_amdsmi.{hpp,cpp} as is.
 */
class AMDSMIProvider {
 public:
  AMDSMIProvider() = default;
  ~AMDSMIProvider();

  bool initialize();
  void finalize();
  bool is_initialized() const { return is_initialized_; }

  size_t get_device_count() const { return devices_.size(); }
  /** @brief PCI address of the device, e.g. 0000:03:00.0 */
  std::string get_device_name(size_t device_index) const;

  /**
   * @brief Cumulative energy of one device since the driver loaded.
   * @param energy_joules counter value times its resolution
   * @param firmware_timestamp_ns timestamp amdsmi returns with the counter
   */
  bool get_device_energy(size_t device_index, double& energy_joules,
                         uint64_t& firmware_timestamp_ns);

  /** @brief Counter resolution in microjoules, read at initialize() */
  double get_counter_resolution_uj(size_t device_index) const;

 private:
  struct Device {
    void* handle;  // amdsmi_processor_handle
    std::string bdf;
    double resolution_uj;
  };

  bool is_initialized_ = false;
  std::vector<Device> devices_;
};

}  // namespace EnergyProfiler
}  // namespace KokkosTools
