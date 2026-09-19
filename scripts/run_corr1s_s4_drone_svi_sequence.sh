#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo '[1/2] Corr1S-S4 Grid64 - CVOGL_DroneAerial'
bash scripts/run_corr1s_s4_drone_naturalrng.sh
echo '[2/2] Corr1S-S4 Grid64 - CVOGL_SVI'
bash scripts/run_corr1s_s4_svi_naturalrng.sh
echo 'Corr1S-S4 Grid64 COMPLETE'
