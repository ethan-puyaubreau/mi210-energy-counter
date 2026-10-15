#!/bin/bash
# Log the load of the GPUs next to the measured one (MI210 #1, H100) every
# 15 s. Never queries GPU 0, so it does not compete with the connector.
# Usage: log_neighbors.sh <csv out>
export PATH=/opt/rocm-7.1.1/bin:$PATH
OUT=$1
[ -f "$OUT" ] || echo "time_epoch_ns,mi210_1_gfx_pct,mi210_1_power_w,h100_util_pct,h100_power_w" > "$OUT"
while true; do
  # socket_power is counted from the end: quoted lists before it contain commas
  amd=$(amd-smi metric -g 1 --usage --power --csv 2>/dev/null |
        awk -F, 'NR == 2 {print $2 "," $(NF - 5)}')
  nv=$(nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  echo "$(date +%s%N),$amd,$nv" >> "$OUT"
  sleep 15
done
