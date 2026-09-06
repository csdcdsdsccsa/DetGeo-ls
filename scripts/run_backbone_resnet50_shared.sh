#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
PYTHONPATH=. "$PYTHON_BIN" train.py --gpu 0 --num_workers 6 --max_epoch 25 --lr 1e-4 --batch_size 8 --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --backbone_exp resnet50_shared --savename detgeo_resnet50_shared --seed 13 --beta 1.0 --standard_rng --print_freq 50 > logs/detgeo_resnet50_shared.log 2>&1
