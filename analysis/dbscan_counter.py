"""Time and energy of ArborX's DBSCANCalculation region from kp_amdsmi_energy
traces, with the indicators of the SMC 2025 poster re-analysis
(smc2025-gpu-energy-poster/analysis/dbscan_medians.py): medians, quartiles,
bootstrap 95% interval of the reduction, runs more than 30% slower than the
median, means.

Energy comes from the hardware counter: cumulative energy_joules interpolated
linearly at the region boundaries, then subtracted. The poster's method
(trapezoid of power_watts over the samples inside the region) is printed next
to it as a cross-check.

Runs whose DBSCANCalculation overlaps activity on the neighbor GPUs
(neighbors.csv from log_neighbors.sh, if present) are flagged.

Usage: python dbscan_counter.py <study dir with runs.csv>
Standard library only.
"""

import csv
import json
import random
import statistics
import sys
from pathlib import Path

REGION = "DBSCANCalculation"
BOOTSTRAP = 10_000
POSTER_H100 = {  # medians over 64 runs, SMC 2025 poster re-analysis (NVML)
    "fdbscan": (2.69, 777.0),
    "fdbscan-densebox": (2.19, 580.0),
}


def counter_at(samples, t):
    """Cumulative energy (J) at time t, linear between the two nearest samples."""
    for (t0, e0, _), (t1, e1, _) in zip(samples, samples[1:]):
        if t0 <= t <= t1:
            return e0 + (e1 - e0) * (t - t0) / (t1 - t0) if t1 > t0 else e0
    raise ValueError(f"t={t} outside the sampled interval")


def measure(trace_dir):
    events = list(csv.DictReader(open(trace_dir / "events.csv", newline="")))
    matches = [e for e in events if e["name"] == REGION]
    if len(matches) != 1:
        raise ValueError(f"{trace_dir}: {len(matches)} {REGION} events")
    start, end = int(matches[0]["start_ns"]), int(matches[0]["end_ns"])
    samples = [
        (int(r["timestamp_ns"]), float(r["energy_joules"]), float(r["power_watts"]))
        for r in csv.DictReader(open(trace_dir / "power_samples.csv", newline=""))
    ]
    joules = counter_at(samples, end) - counter_at(samples, start)
    inside = [(t, p) for t, _, p in samples if start <= t <= end]
    trapezoid = sum((t1 - t0) * (p0 + p1) / 2
                    for (t0, p0), (t1, p1) in zip(inside, inside[1:])) / 1e9
    meta = json.load(open(trace_dir / "metadata.json"))
    seconds = (end - start) / 1e9
    return {"start_ns": start, "end_ns": end, "seconds": seconds, "joules": joules,
            "watts": joules / seconds, "joules_trapezoid": trapezoid,
            "read_cost_ns_p50": meta["read_cost_ns_p50"],
            "read_cost_ns_p99": meta["read_cost_ns_p99"]}


def neighbor_activity(study):
    """(time_ns, busy) samples of the neighbor GPUs, busy if any utilization > 0."""
    path = study / "neighbors.csv"
    if not path.exists():
        return None
    rows = []
    for r in csv.DictReader(open(path, newline="")):
        utils = [r["mi210_1_gfx_pct"], r["h100_util_pct"]]
        rows.append((int(r["time_epoch_ns"]),
                     any(u.strip().replace(".", "", 1).isdigit() and float(u) > 0 for u in utils)))
    return rows


def reduction(a, b, key):
    """Relative reduction of the median of key from runs a to runs b."""
    return 1 - statistics.median(r[key] for r in b) / statistics.median(r[key] for r in a)


def main():
    study = Path(sys.argv[1])
    runs = {"fdbscan": [], "fdbscan-densebox": []}
    for row in csv.DictReader(open(study / "runs.csv", newline="")):
        if row["exit_code"] != "0":
            print(f"skip run {row['index']}: exit code {row['exit_code']}")
            continue
        run = measure(study / row["trace_dir"])
        run.update(index=int(row["index"]), impl=row["impl"],
                   kfd_busy_before=int(row["kfd_busy_before"]))
        runs[row["impl"]].append(run)

    neighbors = neighbor_activity(study)
    for run in runs["fdbscan"] + runs["fdbscan-densebox"]:
        # A neighbor sample within 15 s of the region (the logging period) counts;
        # None when no sample falls there (log not running yet).
        window = [busy for t, busy in neighbors or []
                  if run["start_ns"] - 15e9 <= t <= run["end_ns"] + 15e9]
        run["neighbor_busy"] = any(window) if window else None

    every = sorted(runs["fdbscan"] + runs["fdbscan-densebox"], key=lambda r: r["index"])
    with open(study / "per_run.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(every[0]))
        writer.writeheader()
        writer.writerows(every)

    a, b = runs["fdbscan"], runs["fdbscan-densebox"]
    print(f"{REGION} on {study.name}, energy from the hardware counter")
    for impl, impl_runs in runs.items():
        s, j, w = (statistics.median(r[k] for r in impl_runs) for k in ("seconds", "joules", "watts"))
        jt = statistics.median(r["joules_trapezoid"] for r in impl_runs)
        ref = POSTER_H100[impl]
        print(f"{impl:17s} median of {len(impl_runs)} runs: {s:.3f} s  {j:.1f} J  {w:.1f} W"
              f"   (power trapezoid {jt:.1f} J; H100 poster {ref[0]:.2f} s, {ref[1]:.0f} J)")
    for impl, impl_runs in runs.items():
        (s1, _, s3), (j1, _, j3) = (statistics.quantiles([r[k] for r in impl_runs], n=4)
                                    for k in ("seconds", "joules"))
        print(f"{impl:17s} quartiles: {s1:.3f} to {s3:.3f} s  {j1:.1f} to {j3:.1f} J")
    print(f"densebox vs fdbscan: {reduction(a, b, 'seconds'):.1%} less time, "
          f"{reduction(a, b, 'joules'):.1%} less energy, {reduction(a, b, 'watts'):.1%} less mean power")
    rng = random.Random(2025)
    for key, what in (("seconds", "time"), ("joules", "energy")):
        boot = sorted(reduction(rng.choices(a, k=len(a)), rng.choices(b, k=len(b)), key)
                      for _ in range(BOOTSTRAP))
        print(f"bootstrap 95% interval, reduction in {what}: "
              f"{boot[int(0.025 * BOOTSTRAP)]:.1%} to {boot[int(0.975 * BOOTSTRAP) - 1]:.1%}")

    for impl, impl_runs in runs.items():
        med = statistics.median(r["seconds"] for r in impl_runs)
        slow = [r for r in impl_runs if r["seconds"] > 1.3 * med]
        if not slow:
            print(f"{impl:17s} no run is more than 30% slower than the median")
            continue
        print(f"{impl:17s} {len(slow)} slow runs, order {[r['index'] for r in slow]}: "
              f"{min(r['seconds'] for r in slow):.2f} to {max(r['seconds'] for r in slow):.2f} s, "
              f"mean power {statistics.mean(r['watts'] for r in slow):.0f} W")
        rest = [r for r in impl_runs if r not in slow]
        x, y = (rest, b) if impl == "fdbscan" else (a, rest)
        print(f"  without them: {reduction(x, y, 'seconds'):.1%} less time, "
              f"{reduction(x, y, 'joules'):.1%} less energy")
    for impl, impl_runs in runs.items():
        print(f"{impl:17s} mean of all runs: {statistics.mean(r['seconds'] for r in impl_runs):.3f} s  "
              f"{statistics.mean(r['joules'] for r in impl_runs):.1f} J")

    busy = [r["index"] for r in every if r["kfd_busy_before"]]
    print(f"runs that waited for GPU 0: {busy or 'none'}")
    if neighbors is None:
        print("neighbor GPUs: not logged")
    else:
        covered = [r["index"] for r in every if r["neighbor_busy"] is not None]
        flagged = [r["index"] for r in every if r["neighbor_busy"]]
        print(f"neighbor GPUs logged around {len(covered)} of {len(every)} runs "
              f"({min(covered)} to {max(covered)}); busy during: {flagged or 'none'}")
    print(f"amdsmi read cost at the sampling period: median p50 "
          f"{statistics.median(r['read_cost_ns_p50'] for r in every) / 1e3:.0f} us, median p99 "
          f"{statistics.median(r['read_cost_ns_p99'] for r in every) / 1e3:.0f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
