#!/usr/bin/env bash
# E8: E7 architecture with only Stage2 confidence loss weight reduced to 0.5.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t --standard_rng
  --seed 2024 --beta 1.0 --print_freq 50 --trogeo_ms_det_variant h2_ind_3scale_stage2cls05)
NAME="trogeo_ms_e8_h2_ind_3scale_stage2cls05_swin_t_drone_seed2024"

PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 --savename "$NAME"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" \
  --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
rm -f "saved_models/${NAME}_checkpoint.pth.tar"
