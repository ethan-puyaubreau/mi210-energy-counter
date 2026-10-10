/**
 * @file kp_amdsmi_energy.cpp
 * @brief Experimental KokkosP connector: AMD GPU energy counter via amdsmi.
 *
 * A background thread reads the hardware energy counter of one GPU every
 * KP_AMDSMI_PERIOD_MS (default 10 ms). Regions and kernels are timestamped
 * on the same clock. At finalize the run is written in the trace v1 format
 * of energy-dashboard-for-kokkos (DATA_SPEC.md): events.csv,
 * power_samples.csv and metadata.json.
 *
 * Environment:
 *   KP_AMDSMI_DEVICE     amdsmi GPU index to read (default 0)
 *   KP_AMDSMI_PERIOD_MS  sampling period in ms (default 10)
 *   KP_AMDSMI_OUTPUT     output directory (default <host>-<pid>-amdsmi)
 *
 * Not for upstream as is: kokkos-tools PR #350 owns the daemon.
 */

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "amdsmi_provider.hpp"

using KokkosTools::EnergyProfiler::AMDSMIProvider;
using Clock = std::chrono::steady_clock;

namespace {

struct Event {
  uint64_t id;
  uint64_t parent_id;
  std::string name;
  const char* category;
  int64_t start_ns;
  int64_t end_ns;
};

struct Sample {
  int64_t time_ns;  // middle of the amdsmi call
  double energy_joules;
};

AMDSMIProvider g_provider;
size_t g_device = 0;
int g_period_ms = 10;
std::string g_output_dir;
int64_t g_epoch_offset_ns = 0;  // system_clock - steady_clock at init

std::vector<Event> g_events;
std::vector<uint64_t> g_region_stack;  // ids of open regions
std::vector<Sample> g_samples;         // written by the sampler thread only
std::vector<int64_t> g_read_cost_ns;
long g_read_failures = 0;
std::atomic<bool> g_running{false};
std::thread g_sampler;

/** @brief Monotonic time, shifted to nanoseconds since the UNIX epoch. */
int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
             .count() +
         g_epoch_offset_ns;
}

void take_sample() {
  double energy           = 0.0;
  uint64_t firmware_stamp = 0;
  int64_t before          = now_ns();
  bool ok = g_provider.get_device_energy(g_device, energy, firmware_stamp);
  int64_t after = now_ns();
  if (!ok) {
    ++g_read_failures;
    return;
  }
  g_samples.push_back({before + (after - before) / 2, energy});
  g_read_cost_ns.push_back(after - before);
}

void sampler_loop() {
  auto period = std::chrono::milliseconds(g_period_ms);
  auto next   = Clock::now() + period;
  while (g_running.load()) {
    std::this_thread::sleep_until(next);
    take_sample();
    next += period;
    if (next < Clock::now()) next = Clock::now() + period;  // missed ticks
  }
}

uint64_t open_event(const char* name, const char* category) {
  uint64_t id     = g_events.size() + 1;
  uint64_t parent = g_region_stack.empty() ? 0 : g_region_stack.back();
  g_events.push_back({id, parent, name, category, now_ns(), 0});
  return id;
}

void close_event(uint64_t id) {
  if (id >= 1 && id <= g_events.size()) g_events[id - 1].end_ns = now_ns();
}

int64_t percentile(std::vector<int64_t> values, double p) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  return values[static_cast<size_t>(p * (values.size() - 1))];
}

std::string csv_quote(const std::string& text) {
  if (text.find_first_of(",\"\n") == std::string::npos) return text;
  std::string quoted = "\"";
  for (char c : text) {
    if (c == '"') quoted += '"';
    quoted += c;
  }
  return quoted + "\"";
}

std::string json_escape(const std::string& text) {
  std::string escaped;
  for (char c : text) {
    if (c == '"' || c == '\\') escaped += '\\';
    if (static_cast<unsigned char>(c) >= 0x20) escaped += c;
  }
  return escaped;
}

std::string program_name() {
  std::ifstream cmdline("/proc/self/cmdline");
  std::string argv0;
  std::getline(cmdline, argv0, '\0');
  return argv0.substr(argv0.find_last_of('/') + 1);
}

std::string host_name() {
  char host[256] = {};
  gethostname(host, sizeof host - 1);
  return host;
}

void write_trace() {
  mkdir(g_output_dir.c_str(), 0755);

  std::ofstream events(g_output_dir + "/events.csv");
  events << "id,parent_id,name,category,start_ns,end_ns\n";
  for (const Event& e : g_events) {
    int64_t end = e.end_ns ? e.end_ns : e.start_ns;  // never closed
    events << e.id << ',' << e.parent_id << ',' << csv_quote(e.name) << ','
           << e.category << ',' << e.start_ns << ',' << end << '\n';
  }

  // power_watts of sample i is the mean power over (i-1, i]; the first
  // sample reuses the second one so the series has no artificial step.
  std::ofstream power(g_output_dir + "/power_samples.csv");
  power << "timestamp_ns,domain,device_id,power_watts,energy_joules\n";
  char line[160];
  for (size_t i = 0; i < g_samples.size(); ++i) {
    size_t a  = i == 0 ? 0 : i - 1;
    size_t b  = i == 0 ? std::min<size_t>(1, g_samples.size() - 1) : i;
    double dt = (g_samples[b].time_ns - g_samples[a].time_ns) * 1e-9;
    double watts =
        dt > 0 ? (g_samples[b].energy_joules - g_samples[a].energy_joules) / dt
               : 0.0;
    std::snprintf(line, sizeof line, "%lld,GPU,%zu,%.3f,%.6f\n",
                  static_cast<long long>(g_samples[i].time_ns), g_device, watts,
                  g_samples[i].energy_joules);
    power << line;
  }

  std::ofstream meta(g_output_dir + "/metadata.json");
  meta << "{\n"
       << "  \"spec_version\": \"1.0\",\n"
       << "  \"app_name\": \"" << json_escape(program_name()) << "\",\n"
       << "  \"hostname\": \"" << json_escape(host_name()) << "\",\n"
       << "  \"kokkos_backend\": \"HIP\",\n"
       << "  \"start_epoch_ns\": "
       << (g_samples.empty() ? 0 : g_samples.front().time_ns) << ",\n"
       << "  \"connector\": \"kp_amdsmi_energy (experimental)\",\n"
       << "  \"gpu_bdf\": \"" << g_provider.get_device_name(g_device) << "\",\n"
       << "  \"amdsmi_device_index\": " << g_device << ",\n"
       << "  \"counter_resolution_uj\": "
       << g_provider.get_counter_resolution_uj(g_device) << ",\n"
       << "  \"sample_period_ms\": " << g_period_ms << ",\n"
       << "  \"samples\": " << g_samples.size() << ",\n"
       << "  \"read_failures\": " << g_read_failures << ",\n"
       << "  \"read_cost_ns_p50\": " << percentile(g_read_cost_ns, 0.5) << ",\n"
       << "  \"read_cost_ns_p99\": " << percentile(g_read_cost_ns, 0.99)
       << "\n}\n";

  std::printf("KokkosP AMDSMI energy: %zu events, %zu samples -> %s\n",
              g_events.size(), g_samples.size(), g_output_dir.c_str());
}

}  // namespace

extern "C" void kokkosp_init_library(const int /*loadSeq*/,
                                     const uint64_t /*interfaceVer*/,
                                     const uint32_t /*devInfoCount*/,
                                     void* /*deviceInfo*/) {
  g_epoch_offset_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count() -
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                          Clock::now().time_since_epoch())
                          .count();

  if (const char* v = std::getenv("KP_AMDSMI_DEVICE")) g_device = std::atoi(v);
  if (const char* v = std::getenv("KP_AMDSMI_PERIOD_MS"))
    g_period_ms = std::max(1, std::atoi(v));
  if (const char* v = std::getenv("KP_AMDSMI_OUTPUT"))
    g_output_dir = v;
  else
    g_output_dir = host_name() + "-" + std::to_string(getpid()) + "-amdsmi";

  if (!g_provider.initialize()) return;
  if (g_device >= g_provider.get_device_count()) {
    std::fprintf(stderr,
                 "KokkosP AMDSMI energy: device %zu out of range (%zu)\n",
                 g_device, g_provider.get_device_count());
    g_provider.finalize();
    return;
  }
  std::printf("KokkosP AMDSMI energy: GPU %zu (%s), every %d ms\n", g_device,
              g_provider.get_device_name(g_device).c_str(), g_period_ms);

  g_events.reserve(1 << 16);
  g_samples.reserve(1 << 16);
  g_read_cost_ns.reserve(1 << 16);
  take_sample();
  g_running = true;
  g_sampler = std::thread(sampler_loop);
}

extern "C" void kokkosp_finalize_library() {
  if (!g_provider.is_initialized()) return;
  g_running = false;
  if (g_sampler.joinable()) g_sampler.join();
  take_sample();  // so the samples bracket every event
  write_trace();
  g_provider.finalize();
}

extern "C" void kokkosp_push_profile_region(const char* name) {
  g_region_stack.push_back(open_event(name, "USER_REGION"));
}

extern "C" void kokkosp_pop_profile_region() {
  if (g_region_stack.empty()) return;
  close_event(g_region_stack.back());
  g_region_stack.pop_back();
}

extern "C" void kokkosp_begin_parallel_for(const char* name, const uint32_t,
                                           uint64_t* kID) {
  *kID = open_event(name, "PARALLEL_FOR");
}
extern "C" void kokkosp_end_parallel_for(const uint64_t kID) {
  close_event(kID);
}

extern "C" void kokkosp_begin_parallel_reduce(const char* name, const uint32_t,
                                              uint64_t* kID) {
  *kID = open_event(name, "PARALLEL_REDUCE");
}
extern "C" void kokkosp_end_parallel_reduce(const uint64_t kID) {
  close_event(kID);
}

extern "C" void kokkosp_begin_parallel_scan(const char* name, const uint32_t,
                                            uint64_t* kID) {
  *kID = open_event(name, "PARALLEL_SCAN");
}
extern "C" void kokkosp_end_parallel_scan(const uint64_t kID) {
  close_event(kID);
}
