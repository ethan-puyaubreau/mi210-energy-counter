// sysfs_probe: cost of a raw pread of gpu_metrics (no amdsmi in the loop) and
// how often the energy field inside it moves. amdsmi is only used once, to find
// which 8-byte slot of gpu_metrics holds the energy accumulator.
// Usage: sysfs_probe <amdsmi_gpu_index> <path to gpu_metrics> [seconds=2]
#include <amd_smi/amdsmi.h>

#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
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

static double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[(size_t)(p * (v.size() - 1))];
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <gpu> <gpu_metrics path> [seconds]\n", argv[0]);
    return 2;
  }
  int gpu = std::atoi(argv[1]);
  std::string path = argv[2];
  double duration = argc > 3 ? std::atof(argv[3]) : 2.0;

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
  std::printf("amdsmi_gpu=%d bdf=%04x:%02x:%02x.%x\n", gpu, (unsigned)bdf.bdf.domain_number,
              (unsigned)bdf.bdf.bus_number, (unsigned)bdf.bdf.device_number,
              (unsigned)bdf.bdf.function_number);
  std::string device_dir = path.substr(0, path.rfind('/'));
  char resolved[PATH_MAX];
  if (realpath(device_dir.c_str(), resolved)) std::printf("sysfs_device=%s\n", resolved);

  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    std::perror(path.c_str());
    return 1;
  }
  unsigned char buf[4096];
  ssize_t len = pread(fd, buf, sizeof buf, 0);
  if (len < 4) {
    std::perror("pread");
    return 1;
  }
  std::printf("gpu_metrics bytes=%zd format_rev=%u content_rev=%u\n", len, buf[2], buf[3]);

  // Locate the energy accumulator: the 8-byte slot just above amdsmi's value.
  uint64_t energy = 0, stamp = 0;
  float resolution = 0.0f;
  CHECK_SMI(amdsmi_get_energy_count(handle, &energy, &resolution, &stamp));
  if (pread(fd, buf, sizeof buf, 0) != len) {
    std::perror("pread");
    return 1;
  }
  long offset = -1;
  for (long off = 0; off + 8 <= len; off += 8) {
    uint64_t v;
    std::memcpy(&v, buf + off, 8);
    if (v >= energy && v - energy < 1000000) {
      offset = off;
      break;
    }
  }
  if (offset < 0) {
    std::fprintf(stderr, "energy accumulator not found near amdsmi value %llu\n",
                 (unsigned long long)energy);
    return 1;
  }
  std::printf("energy_offset=%ld\n", offset);
  CHECK_SMI(amdsmi_shut_down());

  std::vector<double> cost_ns;
  for (int i = 0; i < 10000; ++i) {
    Clock::time_point t = Clock::now();
    if (pread(fd, buf, sizeof buf, 0) != len) {
      std::perror("pread");
      return 1;
    }
    cost_ns.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t).count());
  }
  std::printf("pread_ns p50=%.0f p99=%.0f max=%.0f\n", percentile(cost_ns, 0.5),
              percentile(cost_ns, 0.99), percentile(cost_ns, 1.0));

  std::vector<double> change_t;
  uint64_t last = 0;
  long reads = 0, decreases = 0;
  Clock::time_point t0 = Clock::now();
  while (std::chrono::duration<double>(Clock::now() - t0).count() < duration) {
    if (pread(fd, buf, sizeof buf, 0) != len) {
      std::perror("pread");
      return 1;
    }
    uint64_t v;
    std::memcpy(&v, buf + offset, 8);
    if (reads > 0 && v < last) ++decreases;
    if (reads > 0 && v != last)
      change_t.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    last = v;
    ++reads;
  }
  std::vector<double> step_ms;
  for (size_t i = 1; i < change_t.size(); ++i) step_ms.push_back(change_t[i] - change_t[i - 1]);
  std::printf("reads=%ld changes=%zu decreases=%ld\n", reads, change_t.size(), decreases);
  std::printf("step_ms min=%.3f p50=%.3f p99=%.3f max=%.3f\n", percentile(step_ms, 0.0),
              percentile(step_ms, 0.5), percentile(step_ms, 0.99), percentile(step_ms, 1.0));
  close(fd);
  return 0;
}
