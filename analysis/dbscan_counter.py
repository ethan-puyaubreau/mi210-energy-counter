"""Per-run time and energy of the DBSCANCalculation region from kp_amdsmi_energy
traces, medians per implementation, bootstrap of the gap, slow-run detection.

Energy comes from the hardware counter: cumulative energy_joules interpolated
linearly at the region boundaries, then subtracted. No power integration.

Usage: python dbscan_counter.py <study dir with runs.csv> [region name]
Standard library only.
"""

import csv
import json
import random
import statistics
import sys
from pathlib import Path

POSTER_H100 = {  # medians over 64 runs, SMC 2025 poster re-analysis (NVML)
    "fdbscan": (2.69, 777.0),
    "fdbscan-densebox": (2.19, 580.0),
}


def counter_at(samples, t):
    """Cumulative energy (J) at time t, linear between the two nearest samples."""
    for (t0, e0), (t1, e1) in zip(samples, samples[1:]):
        if t0 <= t <= t1:
            return e0 + (e1 - e0) * (t - t0) / (t1 - t0) if t1 > t0 else e0
    raise ValueError(f"t={t} outside the sampled interval")


def region_energy(trace_dir, region):
    events = list(csv.DictReader(open(trace_dir / "events.csv", newline="")))
    matches = [e for e in events if e["name"] == region]
    if len(matches) != 1:
        raise ValueError(f"{trace_dir}: {len(matches)} '{region}' events")
    start, end = int(matches[0]["start_ns"]), int(matches[0]["end_ns"])
    samples = [
        (int(r["timestamp_ns"]), float(r["energy_joules"]))
        for r in csv.DictReader(open(trace_dir / "power_samples.csv", newline=""))
    ]
    energy = counter_at(samples, end) - counter_at(samples, start)
    meta = json.load(open(trace_dir / "metadata.json"))
    return (end - start) * 1e-9, energy, meta


def bootstrap_gap(a, b, rounds=10000, seed=1):
    """95% interval of median(b)/median(a) - 1 over resampled runs."""
    rng = random.Random(seed)
    gaps = sorted(
        statistics.median(rng.choices(b, k=len(b)))
        / statistics.median(rng.choices(a, k=len(a)))
        - 1
        for _ in range(rounds)
    )
    return gaps[int(0.025 * rounds)], gaps[int(0.975 * rounds)]


def main():
    study = Path(sys.argv[1])
    region = sys.argv[2] if len(sys.argv) > 2 else "DBSCANCalculation"
    runs = list(csv.DictReader(open(study / "runs.csv", newline="")))

    rows = []
    for run in runs:
        if run["exit_code"] != "0":
            print(f"skip run {run['index']}: exit code {run['exit_code']}")
            continue
        seconds, joules, meta = region_energy(study / run["trace_dir"], region)
        rows.append({
            "index": int(run["index"]),
            "impl": run["impl"],
            "seconds": seconds,
            "joules": joules,
            "watts": joules / seconds,
            "kfd_busy_before": run["kfd_busy_before"],
            "read_cost_ns_p50": meta["read_cost_ns_p50"],
            "read_cost_ns_p99": meta["read_cost_ns_p99"],
            "samples": meta["samples"],
        })

    with open(study / "per_run.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_impl = {}
    for row in rows:
        by_impl.setdefault(row["impl"], []).append(row)

    print(f"region {region}, {len(rows)} runs, {study}")
    for impl, impl_rows in by_impl.items():
        t = statistics.median(r["seconds"] for r in impl_rows)
        e = statistics.median(r["joules"] for r in impl_rows)
        w = statistics.median(r["watts"] for r in impl_rows)
        h100 = POSTER_H100.get(impl)
        ref = f"   (H100 poster: {h100[0]:.2f} s, {h100[1]:.0f} J)" if h100 else ""
        print(f"{impl:18s} n={len(impl_rows):3d}  median {t:.3f} s  {e:.1f} J  {w:.1f} W{ref}")
        slow = [r for r in impl_rows if r["seconds"] > 1.25 * t]
        if slow:
            print(f"  slow runs (>1.25x median): order {[r['index'] for r in slow]}")
        busy = [r["index"] for r in impl_rows if r["kfd_busy_before"] != "0"]
        if busy:
            print(f"  GPU busy before runs {busy}")

    if {"fdbscan", "fdbscan-densebox"} <= by_impl.keys():
        a, b = by_impl["fdbscan"], by_impl["fdbscan-densebox"]
        for key, label in (("seconds", "time"), ("joules", "energy"), ("watts", "power")):
            va, vb = [r[key] for r in a], [r[key] for r in b]
            gap = statistics.median(vb) / statistics.median(va) - 1
            lo, hi = bootstrap_gap(va, vb)
            print(f"densebox vs fdbscan {label:6s}: {gap:+.1%}  (95% bootstrap {lo:+.1%} .. {hi:+.1%})")

    costs = [r["read_cost_ns_p50"] for r in rows]
    costs99 = [r["read_cost_ns_p99"] for r in rows]
    print(f"amdsmi read cost at the sampling period: median p50 {statistics.median(costs) / 1e3:.0f} us,"
          f" median p99 {statistics.median(costs99) / 1e3:.0f} us")


if __name__ == "__main__":
    main()
