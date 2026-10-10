#!/bin/bash
# Build libkp_amdsmi_energy.so with the same toolchain as Kokkos and ArborX.
set -euo pipefail
cd "$(dirname "$0")"
source /opt/rh/gcc-toolset-13/enable
ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.1.1}
g++ -std=c++17 -O2 -g -fPIC -shared -Wall -Wextra -I$ROCM_PATH/include \
  kp_amdsmi_energy.cpp amdsmi_provider.cpp \
  -L$ROCM_PATH/lib -lamd_smi -Wl,-rpath,$ROCM_PATH/lib -lpthread \
  -o libkp_amdsmi_energy.so
ldd libkp_amdsmi_energy.so | grep -E "stdc\+\+|amd_smi"
