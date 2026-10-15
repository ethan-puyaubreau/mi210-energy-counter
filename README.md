# The AMD GPU energy counter on an MI210, and ArborX DBSCAN measured with it

The [SMC 2025 poster](https://github.com/ethan-puyaubreau/smc2025-gpu-energy-poster) measured
GPU energy on an NVIDIA H100 through NVML, which reports a power value every 100 ms that
covers only the last 25 ms. That rules out attributing energy to kernels shorter than about
10 ms. AMD GPUs expose a cumulative energy counter instead (`amdsmi_get_energy_count`).
This repository characterizes that counter on one AMD Instinct MI210, provides an
experimental Kokkos Tools connector that reads it, and repeats the poster's ArborX DBSCAN
case study with it: same input, same parameters, 64 runs of each implementation.

## Results

### The counter

`analysis/probes_summary.py` on `data/probes-mi210-2026-10-15/`
(full output in [`analysis/expected_probes.txt`](analysis/expected_probes.txt)):

| Probe run | amdsmi call, back to back (p50 / p99) | amdsmi call, 10 ms apart (p50 / p99) | Counter step (p50) | Mean power |
| --- | --- | --- | --- | --- |
| idle, no HIP context | 262 / 510 µs | 532 / 893 µs | 1.170 ms | 57.8 W |
| idle, HIP context open | 275 / 587 µs | 557 / 948 µs | 1.129 ms | 57.8 W |
| FMA kernels back to back | 288 / 600 µs | 469 / 881 µs | 1.181 ms | 143.6 W |
| two readers at once | 315–319 / 750–752 µs | 459–463 / 772–794 µs | 1.178–1.187 ms | 59.9 W |

- The resolution is 15.3 µJ exactly, and the counter readable without root.
- It never decreased, over any probe run nor over the 140,033 samples of the 128 study runs. It counts from the
  moment the driver loads: it read about 33.9 MJ on 2026-10-09 and 4.4 MJ after the node
  rebooted on 2026-10-14.
- Read directly from `/sys/class/drm/card2/device/gpu_metrics` (energy accumulator at byte
  offset 24 of the v1.3 table), the value changes every 1.000 ms (0.947 to 1.053 ms) and a
  read costs 42 µs at the median, 237 µs at p99. Our reading: the amdgpu driver refreshes the
  metrics table from the firmware at most once per millisecond, on the first read after the
  previous copy expired. The ~1.13–1.19 ms step seen through amdsmi is that 1 ms plus the
  wait for the next ~0.3 ms call, and a call 10 ms after the previous one always pays the
  refresh. We did not read the driver source to confirm this.
- Two reads 30 s apart with nothing else on the GPU give the same idle power (57.8 W) as
  reading continuously, so reading the counter does not measurably keep the GPU awake.
- Two concurrent readers are not serialized: the median cost grows by about 20%, p99
  by about half (510 to 750 µs, back to back).

Consequence for a profiler: reading the counter at every Kokkos region or kernel boundary
costs 0.3 to 0.6 ms per read, more than many kernels last. The counter has to be read by a
sampling thread. At a 10 ms period that thread spends about 5% of one core in amdsmi.

### ArborX DBSCAN

`analysis/dbscan_counter.py` on `data/dbscan-mi210-2026-10-09/`
(full output in [`analysis/expected_dbscan.txt`](analysis/expected_dbscan.txt)).
The `DBSCANCalculation` region of each run; energy is the counter difference, interpolated
linearly between the two samples around each region boundary.

| Implementation | Runs | Time (median) | Energy (median) | Mean power | Time quartiles | Energy quartiles |
| --- | --- | --- | --- | --- | --- | --- |
| `fdbscan` | 64 | 6.948 s | 1465.2 J | 210.9 W | 6.942–6.953 s | 1464.2–1466.9 J |
| `fdbscan-densebox` | 64 | 4.309 s | 846.3 J | 196.5 W | 4.305–4.314 s | 845.8–847.5 J |

- `fdbscan-densebox` takes 38.0% less time and 42.2% less energy, with 6.8% lower mean power.
  Bootstrap 95% intervals over the runs: 37.9 to 38.0% for time, 42.2 to 42.3% for energy.
- No run is more than 30% slower than its implementation's median; means equal medians to
  within 0.2%. The two implementations alternated run by run (order in `runs.csv`).
- As on the H100, the faster implementation is also the more frugal one, and the energy gap
  is wider than the time gap.
- `energy-dashboard-for-kokkos analyze` on the median runs gives 1464.3 J and 845.7 J
  (`data/dbscan-mi210-2026-10-09/dashboard/`). In `fdbscan`, the `HalfTraversal` kernel
  holds 80% of the region's energy.
- The amdsmi read in the connector, 10 ms apart, cost 473 µs at the median of the per-run
  p50s and 792 µs at the median of the per-run p99s.

### What this does not show

- **AMD against NVIDIA.** The H100 figures of the poster (2.69 s and 777 J for `fdbscan`,
  2.19 s and 580 J for `fdbscan-dense`) come from NVML power sampling, ArborX `5054ad0`
  (July 2025) and the Kokkos of the time. These runs use the energy counter, ArborX `953dd61`
  (October 2026) and Kokkos 5.0.2. The scope of what each sensor measures was not compared
  either. The wider gap between the two implementations here (38% against 19% in time) may
  come from the GPU or from ArborX changes since July 2025.
- **Absolute accuracy.** The counter was not compared with an external power meter. The
  power-trapezoid figure printed next to the counter energy uses power derived from the same
  counter: it checks the arithmetic, not the sensor.
- **One machine state.** The DBSCAN study ran on 2026-10-09 with an idle power of about
  41 W. After a reboot on 2026-10-14 the same GPU idled at about 58 W with its clocks at
  their maximum (1700 MHz compute, 1600 MHz memory), performance level `auto`. The probe
  data are from that later state; we did not find the cause.

## Contents

| Path | Content |
| --- | --- |
| `connector/` | `libkp_amdsmi_energy.so`, an experimental Kokkos Tools connector |
| `probes/` | `energy_probe.cpp`, `sysfs_probe.cpp` and `run_probes.sh` |
| `study/` | `run_study.sh` (DBSCAN protocol), `log_neighbors.sh`, the ArborX patch |
| `analysis/` | The two analysis scripts and their expected output |
| `data/probes-mi210-2026-10-15/` | Raw probe samples: `*-call-cost.csv`, `*-counter-changes.csv`, `sysfs-*.csv`, probe logs, `environment.txt` |
| `data/dbscan-mi210-2026-10-09/` | `environment.txt`, `runs.csv` (order and exit codes), `neighbors.csv`, `per_run.csv`, one trace per run |

Each run directory holds a trace in the v1 format of
[energy-dashboard-for-kokkos](https://github.com/ethan-puyaubreau/energy-dashboard-for-kokkos)
(`events.csv`, `power_samples.csv`, `metadata.json`) and the benchmark output (`run.log`).
`energy_joules` is the counter; `power_watts` of a sample is the mean power since the
previous one.

`neighbors.csv` logs, every 15 s, the load of the second MI210 and of the H100 that share the
node. It started during the study and covers runs 11 to 128: both stayed idle.

## Connector

`connector/amdsmi_provider.{hpp,cpp}` is the only code that calls amdsmi. It follows the
provider layout of [kokkos/kokkos-tools#302](https://github.com/kokkos/kokkos-tools/pull/302)
so it can move there; `kp_amdsmi_energy.cpp` adds a sampling thread and the Kokkos Tools
callbacks. It is experimental: the sampling daemon upstream is the subject of
[kokkos/kokkos-tools#350](https://github.com/kokkos/kokkos-tools/pull/350).

| Variable | Default | Meaning |
| --- | --- | --- |
| `KP_AMDSMI_DEVICE` | `0` | amdsmi GPU index |
| `KP_AMDSMI_PERIOD_MS` | `10` | sampling period |
| `KP_AMDSMI_OUTPUT` | `<host>-<pid>-amdsmi` | output directory |

```bash
connector/build.sh
KOKKOS_TOOLS_LIBS=$PWD/connector/libkp_amdsmi_energy.so ./app
```

## Reproduce

The figures above, from the raw data (Python 3.8 or later, standard library only):

```bash
python analysis/probes_summary.py data/probes-mi210-2026-10-15
python analysis/dbscan_counter.py data/dbscan-mi210-2026-10-09
```

The output must match `analysis/expected_*.txt`; the
[Reproduce workflow](.github/workflows/reproduce.yml) checks it on every push.

The measurements themselves need the hardware. On illyad, both from Slurm jobs:

```bash
srun -p illyad -c 4 -t 20 probes/run_probes.sh data/probes-<date>
srun -p illyad -c 8 -t 240 study/run_study.sh data/dbscan-<date> 64
```

`run_study.sh` expects Kokkos and ArborX built as below, the input file in the home directory,
and the connector built with `connector/build.sh`.

## Setup

- Node: illyad, OACISS, University of Oregon. Two AMD Instinct MI210 (gfx90a, 104 CUs, IFWI
  020.040.000.036), one NVIDIA H100 PCIe. The MI210 at PCI address `0000:03:00.0` was
  measured (amdsmi index 0, HIP device 0).
- RHEL 8.10, amdgpu 6.14.14, ROCm 7.1.1, amd-smi 26.2.0.
- Kokkos 5.0.2 (`dfceb77`), HIP backend, `Kokkos_ARCH_AMD_GFX90A`, C++20.
- ArborX `953dd61` with [`study/dbscan_regions_head.patch`](study/dbscan_regions_head.patch),
  which adds the poster's `DBSCANCalculation` region around the call to `ArborX::dbscan`.
- hipcc (AMD clang 20) with `--gcc-toolchain` set to GCC 13.3 for every component.
- Input: `hacc_497M.arborx`, 497,838,606 3D points from a HACC cosmology simulation (6 GB,
  not included). Parameters as in the poster: `--core-min-size 2 --eps 0.015`, binary input,
  float precision.

## References

- A. Prokopenko, D. Lebrun-Grandié, D. Arndt. *Fast tree-based algorithms for DBSCAN for
  low-dimensional data on GPUs.* ICPP 2023, pp. 503–512.
  [arXiv:2103.05162](https://arxiv.org/abs/2103.05162). The `fdbscan` and
  `fdbscan-densebox` algorithms.
- [ArborX](https://github.com/arborx/ArborX), [Kokkos](https://github.com/kokkos/kokkos),
  [Kokkos Tools](https://github.com/kokkos/kokkos-tools).
- [AMD SMI](https://github.com/ROCm/amdsmi), `amdsmi_get_energy_count` in `amd_smi/amdsmi.h`.
- E. Puyaubreau, D. Arndt, J. Bludau, D. Lebrun-Grandié. *Understanding GPU energy dynamics
  in HPC applications.* Poster, SMC 2025.
  [Repository](https://github.com/ethan-puyaubreau/smc2025-gpu-energy-poster).

## License

The code (`connector/`, `probes/`, `study/`, `analysis/`) is under the [MIT license](LICENSE),
except `study/dbscan_regions_head.patch`, a change to an ArborX file under ArborX's
BSD-3-Clause license. The data in `data/` are under the
[Creative Commons Attribution 4.0 International license](data/LICENSE).

## Acknowledgments

This research used resources on the Frank cluster at the University of Oregon.
