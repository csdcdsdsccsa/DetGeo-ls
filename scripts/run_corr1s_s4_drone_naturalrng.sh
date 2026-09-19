#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
NAME="corr1s_s4_currentpe_distance_naturalrng_swin_t_drone_seed2024"
COMMON=(--gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t
  --trogeo_ms_det_variant corr1s_s4 --trogeo_position_mode current --trogeo_click_map_mode distance
  --dadpe_mode none --amr_pe_mode none --standard_rng --seed 2024 --beta 1.0 --print_freq 50)
PYTHONPATH=. "$PYTHON_BIN" tools/test_corr1s_s4.py --data_name CVOGL_DroneAerial
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --max_epoch 25 --lr 1e-4 --savename "$NAME"
PYTHONPATH=. "$PYTHON_BIN" train.py "${COMMON[@]}" --pretrain "saved_models/${NAME}_model_best.pth.tar" --test --savename "${NAME}_test"
rm -f "saved_models/${NAME}_checkpoint.pth.tar"
