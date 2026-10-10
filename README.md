# amdsmi-counter

Experimental KokkosP connector that reads the AMD GPU hardware energy counter
(`amdsmi_get_energy_count`) from a background thread, and the ArborX DBSCAN
study run with it on one AMD Instinct MI210 of illyad (OACISS, University of
Oregon). Not upstream code: the daemon belongs to kokkos-tools PR #350, and
`connector/amdsmi_provider.*` is meant to move to `provider/` as in PR #302.

## Layout

| Path | Content |
| --- | --- |
| `connector/` | `libkp_amdsmi_energy.so`: `AMDSMIProvider` (the only amdsmi user) and the KokkosP callbacks |
| `probes/` | `energy_probe.cpp`, `sysfs_probe.cpp`: counter resolution, refresh step, call cost |
| `dbscan_regions_head.patch` | `DBSCANCalculation` region of the poster, ported to ArborX `953dd61` |
| `run_study.sh` | Poster protocol on the MI210, both implementations alternating |
| `analysis/dbscan_counter.py` | Region time and energy from the counter, medians, bootstrap |
| `results/` | One directory per study: `environment.txt`, `runs.csv`, one trace per run |

## Connector

Output is the trace v1 format of
[energy-dashboard-for-kokkos](https://github.com/ethan-puyaubreau/energy-dashboard-for-kokkos)
(`events.csv`, `power_samples.csv`, `metadata.json`). `energy_joules` is the
counter; `power_watts` of a sample is the mean power since the previous one.

| Variable | Default | Meaning |
| --- | --- | --- |
| `KP_AMDSMI_DEVICE` | `0` | amdsmi GPU index |
| `KP_AMDSMI_PERIOD_MS` | `10` | sampling period |
| `KP_AMDSMI_OUTPUT` | `<host>-<pid>-amdsmi` | output directory |

```bash
connector/build.sh
KOKKOS_TOOLS_LIBS=$PWD/connector/libkp_amdsmi_energy.so ./app
```

## Study

```bash
srun -p illyad -c 8 -t 240 ./run_study.sh results/mi210-<date> 64
python3 analysis/dbscan_counter.py results/mi210-<date>
```

This research used resources on the Frank cluster at the University of Oregon.
