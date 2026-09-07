#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
PYTHONPATH=. "$PYTHON_BIN" train.py \
  --gpu 0 --num_workers 24 --max_epoch 25 --lr 1e-4 --batch_size 7 \
  --emb_size 768 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial \
  --trogeo_direct_ca --trogeo_backbone resnet50 --standard_rng --seed 2024 --beta 1.0 \
  --savename trogeo_direct_ca_resnet50_drone_seed2024 --print_freq 50
