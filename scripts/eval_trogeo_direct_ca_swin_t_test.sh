#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
PYTHONPATH=. "$PYTHON_BIN" train.py \
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 \
  --data_root data --data_name CVOGL_DroneAerial --trogeo_direct_ca --trogeo_backbone swin_t --standard_rng \
  --seed 2024 --beta 1.0 --pretrain saved_models/trogeo_direct_ca_swin_t_drone_seed2024_model_best.pth.tar \
  --test --savename trogeo_direct_ca_swin_t_drone_seed2024_test --print_freq 50
