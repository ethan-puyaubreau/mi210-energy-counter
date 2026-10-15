"""Energy counter characterization from the raw probe CSVs (probes/run_probes.sh).

For each probe run: amdsmi call cost back to back and 10 ms apart, the steps
between two changes of the counter (host clock and firmware timestamp), any
decrease of the counter, and the mean power over the run. Then the raw
gpu_metrics read (no amdsmi) and the idle power read 30 s apart.

Usage: python probes_summary.py <probe data dir>
Standard library only.
"""

import csv
import re
import sys
from pathlib import Path

RUNS = ("sleep", "awake", "load", "concurrent-1", "concurrent-2")


def pct(values, p):
    """Same convention as the C++ probes: sorted[floor(p * (n - 1))]."""
    values = sorted(values)
    return values[int(p * (len(values) - 1))]


def rows(path):
    return list(csv.DictReader(open(path, newline="")))


def main():
    data = Path(sys.argv[1])
    resolution_uj = float(re.search(r"resolution_uJ=([\d.]+)", (data / "sleep.txt").read_text()).group(1))
    print(f"counter resolution: {resolution_uj} uJ")
    print("run           call us p50/p99  10ms-apart us p50/p99  changes  step ms p50 [min, max]  "
          "fw step ms p50  decreases  mean W")
    for run in RUNS:
        cost = rows(data / f"{run}-call-cost.csv")
        tight = [float(r["call_ns"]) / 1e3 for r in cost if r["spacing"] == "tight"]
        spaced = [float(r["call_ns"]) / 1e3 for r in cost if r["spacing"] == "10ms"]
        changes = [(float(r["host_s"]), int(r["energy_count"]), int(r["firmware_timestamp"]))
                   for r in rows(data / f"{run}-counter-changes.csv")]
        steps = [(b[0] - a[0]) * 1e3 for a, b in zip(changes, changes[1:])]
        fw_steps = [(b[2] - a[2]) / 1e6 for a, b in zip(changes, changes[1:])]
        decreases = sum(b[1] < a[1] for a, b in zip(changes, changes[1:]))
        (t0, e0, _), (t1, e1, _) = changes[0], changes[-1]
        watts = (e1 - e0) * resolution_uj * 1e-6 / (t1 - t0)
        print(f"{run:13s} {pct(tight, .5):7.0f} /{pct(tight, .99):5.0f}   "
              f"{pct(spaced, .5):9.0f} /{pct(spaced, .99):5.0f}        {len(changes):6d}  "
              f"{pct(steps, .5):6.3f} [{min(steps):.3f}, {max(steps):.3f}]   "
              f"{pct(fw_steps, .5):12.3f}  {decreases:9d}  {watts:6.1f}")

    pread = [float(r["pread_ns"]) / 1e3 for r in rows(data / "sysfs-pread-cost.csv")]
    times = [float(r["host_ms"]) for r in rows(data / "sysfs-energy-changes.csv")]
    steps = [b - a for a, b in zip(times, times[1:])]
    print(f"raw gpu_metrics pread: p50 {pct(pread, .5):.1f} us, p99 {pct(pread, .99):.1f} us; "
          f"energy field changes every {pct(steps, .5):.3f} ms p50 [{min(steps):.3f}, {max(steps):.3f}]")
    idle = float((data / "idle_30s.txt").read_text().split("=")[1])
    print(f"idle power from two reads 30 s apart: {idle:.1f} W")
    return 0


if __name__ == "__main__":
    sys.exit(main())
