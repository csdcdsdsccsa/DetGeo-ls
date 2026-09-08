#!/usr/bin/env bash
# Wait for the currently running E8 screen session, then run all five PE/LE experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
while screen -ls 2>/dev/null | grep -q 'trogeo_e8_stage2cls05'; do
  sleep 30
done
test -f saved_models/trogeo_ms_e8_h2_ind_3scale_stage2cls05_swin_t_drone_seed2024_model_best.pth.tar
bash scripts/run_trogeo_query_pe_3scale_all.sh
