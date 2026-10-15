#!/bin/bash
# ArborX DBSCAN energy study on one MI210 of illyad, poster protocol:
# hacc_497M (binary, now the default), --core-min-size 2 --eps 0.015, N runs per implementation,
# sleep 0.025*k+1 s after each run of the k-th pair. Implementations alternate run by run so a
# drift of the machine state hits both equally; the order is in runs.csv.
# Usage: run_study.sh <out dir> [runs per impl=64]
set -uo pipefail
OUT=$1
N=${2:-64}
HERE=$(cd "$(dirname "$0")" && pwd)
EXE=~/amd-study/build/arborx/benchmarks/cluster/ArborX_Benchmark_DBSCAN.exe
DATA=~/hacc_497M.arborx
LIB=$HERE/../connector/libkp_amdsmi_energy.so
export PATH=/opt/rocm-7.1.1/bin:$PATH  # srun keeps the PATH of orthus
command -v rocm-smi > /dev/null || { echo "rocm-smi not found" >&2; exit 1; }
export HIP_VISIBLE_DEVICES=0  # BDF 0000:03:00.0, amdsmi index 0
export KP_AMDSMI_DEVICE=0 KP_AMDSMI_PERIOD_MS=${KP_AMDSMI_PERIOD_MS:-10}

mkdir -p "$OUT"
{
  echo "date: $(date -Is)"
  echo "host: $(hostname), slurm job: ${SLURM_JOB_ID:-none}"
  echo "rocm: $(readlink -f /opt/rocm)"
  amd-smi static --asic --vbios --driver -g 0 | grep -E "MARKET_NAME|VERSION:|BUILD_DATE|PART_NUMBER|NAME: (Mi|amdgpu)"
  echo "kokkos: $(git -C ~/amd-study/src/kokkos rev-parse HEAD)"
  echo "arborx: $(git -C ~/amd-study/src/arborx rev-parse HEAD) (+ DBSCANCalculation region patch)"
  echo "connector: $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo uncommitted)"
  echo "command: $EXE --impl <impl> --filename $DATA --core-min-size 2 --eps 0.015"
} > "$OUT/environment.txt"

"$HERE/log_neighbors.sh" "$OUT/neighbors.csv" &
trap 'kill %1' EXIT

echo "index,impl,start_iso,exit_code,kfd_busy_before,trace_dir" > "$OUT/runs.csv"
i=0
for rep in $(seq 1 "$N"); do
  for impl in fdbscan fdbscan-densebox; do
    i=$((i + 1))
    busy=0
    for attempt in $(seq 1 20); do
      # "PID <n> is using <k> DRM device(s):" then the device list on the next line
      pids=$(rocm-smi --showpidgpus 2>&1)
      if ! awk '/is using/ {getline; for (f = 1; f <= NF; f++) if ($f == "0") busy = 1}
                END {exit !busy}' <<< "$pids"; then break; fi
      busy=$attempt
      { date -Is; echo "$pids"; } >> "$OUT/gpu_busy.log"
      sleep 30
    done
    dir=$(printf "%s/run_%03d_%s" "$OUT" "$i" "$impl")
    start=$(date -Is)
    KOKKOS_TOOLS_LIBS=$LIB KP_AMDSMI_OUTPUT=$dir \
      "$EXE" --impl "$impl" --filename "$DATA" --core-min-size 2 --eps 0.015 \
      > "$dir.log" 2>&1
    code=$?
    mv "$dir.log" "$dir/run.log" 2>/dev/null
    echo "$i,$impl,$start,$code,$busy,$(basename "$dir")" >> "$OUT/runs.csv"
    echo "run $i/$((2 * N)) $impl exit=$code kfd_busy=$busy"
    sleep "$(echo "0.025 * $rep + 1" | bc)"
  done
done
