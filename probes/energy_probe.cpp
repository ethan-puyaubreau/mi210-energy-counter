// energy_probe: quick look at the AMD GPU energy counter through amdsmi.
// Usage: energy_probe <sleep|awake|load> [amdsmi_gpu_index=0] [seconds=2]
//   sleep: no HIP context at all (GPU may stay in its low-power state)
//   awake: HIP context held open on the same GPU, no work
//   load : back-to-back FMA kernels on the same GPU
#include <amd_smi/amdsmi.h>
#include <hip/hip_runtime.h>

#include <strings.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

#define CHECK_SMI(call)                                                \
  do {                                                                 \
    amdsmi_status_t s_ = (call);                                       \
    if (s_ != AMDSMI_STATUS_SUCCESS) {                                 \
      std::fprintf(stderr, "%s failed: status %d\n", #call, (int)s_);  \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

#define CHECK_HIP(call)                                                          \
  do {                                                                           \
    hipError_t e_ = (call);                                                      \
    if (e_ != hipSuccess) {                                                      \
      std::fprintf(stderr, "%s failed: %s\n", #call, hipGetErrorString(e_));     \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

__global__ void burn(float* out, int iters) {
  float a = threadIdx.x * 1e-3f;
  for (int i = 0; i < iters; ++i) a = fmaf(a, 1.0001f, 0.5f);
  if (a == -1.0f) out[0] = a;  // keeps the loop from being optimized away
}

static double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[(size_t)(p * (v.size() - 1))];
}

static double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "sleep";
  int gpu = argc > 2 ? std::atoi(argv[2]) : 0;
  double duration = argc > 3 ? std::atof(argv[3]) : 2.0;
  if (mode != "sleep" && mode != "awake" && mode != "load") {
    std::fprintf(stderr, "usage: %s <sleep|awake|load> [gpu] [seconds]\n", argv[0]);
    return 2;
  }

  CHECK_SMI(amdsmi_init(AMDSMI_INIT_AMD_GPUS));
  uint32_t nsock = 0;
  CHECK_SMI(amdsmi_get_socket_handles(&nsock, nullptr));
  std::vector<amdsmi_socket_handle> sockets(nsock);
  CHECK_SMI(amdsmi_get_socket_handles(&nsock, sockets.data()));
  std::vector<amdsmi_processor_handle> gpus;
  for (amdsmi_socket_handle s : sockets) {
    uint32_t n = 0;
    CHECK_SMI(amdsmi_get_processor_handles(s, &n, nullptr));
    std::vector<amdsmi_processor_handle> handles(n);
    CHECK_SMI(amdsmi_get_processor_handles(s, &n, handles.data()));
    gpus.insert(gpus.end(), handles.begin(), handles.end());
  }
  if (gpu < 0 || gpu >= (int)gpus.size()) {
    std::fprintf(stderr, "gpu %d out of range (%zu found)\n", gpu, gpus.size());
    return 2;
  }
  amdsmi_processor_handle handle = gpus[gpu];
  amdsmi_bdf_t bdf;
  CHECK_SMI(amdsmi_get_gpu_device_bdf(handle, &bdf));
  char bus[32];
  std::snprintf(bus, sizeof bus, "%04x:%02x:%02x.%x", (unsigned)bdf.bdf.domain_number,
                (unsigned)bdf.bdf.bus_number, (unsigned)bdf.bdf.device_number,
                (unsigned)bdf.bdf.function_number);
  std::printf("mode=%s amdsmi_gpu=%d bdf=%s\n", mode.c_str(), gpu, bus);

  // HIP is only touched in awake/load, so "sleep" never opens a context.
  float* scratch = nullptr;
  hipStream_t stream = nullptr;
  if (mode != "sleep") {
    int ndev = 0;
    CHECK_HIP(hipGetDeviceCount(&ndev));
    int dev = -1;
    for (int d = 0; d < ndev; ++d) {
      char hip_bus[32];
      CHECK_HIP(hipDeviceGetPCIBusId(hip_bus, sizeof hip_bus, d));
      std::printf("hip_device=%d bdf=%s\n", d, hip_bus);
      if (strcasecmp(hip_bus, bus) == 0) dev = d;
    }
    if (dev < 0) {
      std::fprintf(stderr, "no HIP device matches %s\n", bus);
      return 1;
    }
    CHECK_HIP(hipSetDevice(dev));
    CHECK_HIP(hipMalloc(&scratch, sizeof(float)));
    CHECK_HIP(hipStreamCreate(&stream));
  }
  auto keep_busy = [&]() {
    if (mode == "load" && hipStreamQuery(stream) == hipSuccess)
      hipLaunchKernelGGL(burn, dim3(104 * 16), dim3(256), 0, stream, scratch, 1 << 18);
  };
  Clock::time_point warmup = Clock::now();
  while (seconds_since(warmup) < 1.0) {
    keep_busy();
    if (mode != "load") std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  uint64_t energy = 0, stamp = 0;
  float resolution = 0.0f;
  CHECK_SMI(amdsmi_get_energy_count(handle, &energy, &resolution, &stamp));
  std::printf("resolution_uJ=%.6f counter_J=%.3f\n", resolution, energy * resolution * 1e-6);

  // Cost of one call.
  std::vector<double> cost_ns;
  for (int i = 0; i < 10000; ++i) {
    keep_busy();
    Clock::time_point t = Clock::now();
    CHECK_SMI(amdsmi_get_energy_count(handle, &energy, &resolution, &stamp));
    cost_ns.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t).count());
  }
  std::printf("call_ns p50=%.0f p99=%.0f max=%.0f\n", percentile(cost_ns, 0.5),
              percentile(cost_ns, 0.99), percentile(cost_ns, 1.0));

  // Refresh step: tight loop, keep only the samples where the counter moved.
  struct Change { double t; uint64_t energy, stamp; };
  std::vector<Change> changes;
  uint64_t last = energy;
  long reads = 0, decreases = 0;
  Clock::time_point t0 = Clock::now();
  double first_t = seconds_since(t0);
  uint64_t first_energy = energy;
  while (seconds_since(t0) < duration) {
    keep_busy();
    CHECK_SMI(amdsmi_get_energy_count(handle, &energy, &resolution, &stamp));
    ++reads;
    if (energy < last) ++decreases;
    if (energy != last) changes.push_back({seconds_since(t0), energy, stamp});
    last = energy;
  }
  double last_t = seconds_since(t0);

  std::vector<double> step_ms, stamp_delta, step_uJ;
  for (size_t i = 1; i < changes.size(); ++i) {
    step_ms.push_back((changes[i].t - changes[i - 1].t) * 1e3);
    stamp_delta.push_back((double)(changes[i].stamp - changes[i - 1].stamp));
    step_uJ.push_back((changes[i].energy - changes[i - 1].energy) * resolution);
  }
  std::printf("reads=%ld changes=%zu decreases=%ld\n", reads, changes.size(), decreases);
  std::printf("step_ms min=%.3f p50=%.3f p99=%.3f max=%.3f\n", percentile(step_ms, 0.0),
              percentile(step_ms, 0.5), percentile(step_ms, 0.99), percentile(step_ms, 1.0));
  std::printf("fw_stamp_delta p50=%.0f (unit unknown, compare with step_ms)\n",
              percentile(stamp_delta, 0.5));
  std::printf("energy_per_step_uJ p50=%.1f\n", percentile(step_uJ, 0.5));
  std::printf("avg_power_W=%.2f\n",
              (last - first_energy) * resolution * 1e-6 / (last_t - first_t));

  if (stream) {
    CHECK_HIP(hipStreamSynchronize(stream));
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(scratch));
  }
  CHECK_SMI(amdsmi_shut_down());
  return 0;
}
