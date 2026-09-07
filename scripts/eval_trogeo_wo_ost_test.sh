#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
PYTHONPATH=. "$PYTHON_BIN" train.py \
  --gpu 0 --num_workers 24 --batch_size 6 --emb_size 768 --img_size 1024 \
  --data_root data --data_name CVOGL_DroneAerial --trogeo_wo_ost --standard_rng \
  --seed 2024 --beta 1.0 --pretrain saved_models/trogeo_wo_ost_drone_seed2024_model_best.pth.tar \
  --test --savename trogeo_wo_ost_drone_seed2024_test --print_freq 50
