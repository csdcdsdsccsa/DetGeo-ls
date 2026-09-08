#!/usr/bin/env bash
# Validation-only H3 threshold selection.  Run one test afterwards with the
# selected threshold; do not select it using the test set.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_ms_det_variant h3_ind
  --trogeo_backbone swin_t --standard_rng --seed 2024 --beta 1.0 --print_freq 50)
CHECKPOINT="saved_models/trogeo_ms_e4_h2_ind_swin_t_drone_seed2024_model_best.pth.tar"

for THRESHOLD in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
  TAG="${THRESHOLD/./p}"
  PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --h3_iou_threshold "$THRESHOLD" \
    --pretrain "$CHECKPOINT" --val \
    --savename "trogeo_ms_e5_h3_ind_iou${TAG}_swin_t_drone_seed2024_val"
done
