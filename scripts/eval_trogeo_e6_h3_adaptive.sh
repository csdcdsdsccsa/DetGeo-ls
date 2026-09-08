#!/usr/bin/env bash
# E6 inference-only adaptive fusion: tau_i = 0.7 - 0.3 * confidence_balance.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_ms_det_variant h3_adaptive
  --trogeo_backbone swin_t --standard_rng --seed 2024 --beta 1.0 --print_freq 50
  --pretrain saved_models/trogeo_ms_e4_h2_ind_swin_t_drone_seed2024_model_best.pth.tar)

PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --val \
  --savename trogeo_ms_e6_h3_adaptive_swin_t_drone_seed2024_val
