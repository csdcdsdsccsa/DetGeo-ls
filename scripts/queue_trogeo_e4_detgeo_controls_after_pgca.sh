#!/usr/bin/env bash
# Do not begin controls until the active PGCA C->A->B screen has succeeded.
set -euo pipefail
cd "$(dirname "$0")/.."
while screen -ls 2>/dev/null | grep -q '[.]trogeo_e4_pgca_abc'; do
  sleep 60
done

for name in \
  trogeo_ms_e4_pgca_c_direct_bias_swin_t_drone_seed2024 \
  trogeo_ms_e4_pgca_a_conv_bias_swin_t_drone_seed2024 \
  trogeo_ms_e4_pgca_b_dynamic_lambda_swin_t_drone_seed2024; do
  test -f "saved_models/${name}_model_best.pth.tar" || {
    echo "PGCA queue did not complete successfully; missing ${name}_model_best.pth.tar" >&2
    exit 1
  }
done

exec bash scripts/run_trogeo_e4_detgeo_controls.sh
