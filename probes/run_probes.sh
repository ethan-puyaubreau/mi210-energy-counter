#!/bin/bash
# Characterize the energy counter of one MI210 on illyad: resolution, refresh
# step, call cost (back to back and 10 ms apart), raw sysfs read cost, two
# concurrent readers, and idle power read 30 s apart. Raw CSVs go to <out dir>.
# Usage (inside a Slurm job on illyad): run_probes.sh <out dir>
set -euo pipefail
OUT=$(realpath -m "$1")
HERE=$(cd "$(dirname "$0")" && pwd)
source /opt/rh/gcc-toolset-13/enable
export ROCM_PATH=/opt/rocm-7.1.1 PATH=/opt/rocm-7.1.1/bin:$PATH
TC="--gcc-toolchain=/opt/rh/gcc-toolset-13/root/usr"
mkdir -p "$OUT"
cd "$HERE"

hipcc $TC -std=c++17 -O2 --offload-arch=gfx90a energy_probe.cpp \
  -L$ROCM_PATH/lib -lamd_smi -Wl,-rpath,$ROCM_PATH/lib -o energy_probe
g++ -std=c++17 -O2 -I$ROCM_PATH/include sysfs_probe.cpp \
  -L$ROCM_PATH/lib -lamd_smi -Wl,-rpath,$ROCM_PATH/lib -o sysfs_probe

metrics=""
for d in /sys/class/drm/card*/device; do
  if [[ "$(readlink -f "$d")" == *0000:03:00.0 ]]; then metrics="$d/gpu_metrics"; fi
done

{
  echo "date: $(date -Is)"
  echo "host: $(hostname), slurm job: ${SLURM_JOB_ID:-none}"
  echo "rocm: $ROCM_PATH, $(amd-smi version 2>/dev/null)"
  amd-smi static --asic --vbios --driver -g 0
  echo "gpu_metrics: $metrics"
  echo "GPU processes before the probes:"
  rocm-smi --showpidgpus 2>&1 | grep -v -E "^$|====|WARNING"
} > "$OUT/environment.txt"

# Idle power from two counter reads 30 s apart, nothing else touching the GPU.
read_joules() { amd-smi metric -g 0 --energy | awk '/TOTAL_ENERGY_CONSUMPTION/ {print $2}'; }
t0=$(date +%s.%N); e0=$(read_joules); sleep 30; t1=$(date +%s.%N); e1=$(read_joules)
echo "idle_30s_W=$(echo "($e1 - $e0) / ($t1 - $t0)" | bc -l)" > "$OUT/idle_30s.txt"

for mode in sleep awake load; do
  ./energy_probe "$mode" 0 2 "$OUT/$mode" > "$OUT/$mode.txt"
done
./sysfs_probe 0 "$metrics" 2 "$OUT/sysfs" > "$OUT/sysfs.txt"
./energy_probe sleep 0 3 "$OUT/concurrent-1" > "$OUT/concurrent-1.txt" &
./energy_probe sleep 0 3 "$OUT/concurrent-2" > "$OUT/concurrent-2.txt"
wait
rm -f energy_probe sysfs_probe
