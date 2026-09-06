#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
# Checkpoint is after epoch 15; resume epochs 16--24 and retain the historical best at epoch <=15.
PYTHONPATH=. "$PYTHON_BIN" train.py --gpu 0 --num_workers 24 --max_epoch 25 --lr 1e-4 --batch_size 8 --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --savename standard_b_e4_b2_seed13 --seed 13 --beta 1.0 --b_variant b2 --standard_rng --resume saved_models/standard_b_e4_b2_seed13_checkpoint.pth.tar --resume_best_accu 0.438787 --print_freq 50 > logs/standard_b_e4_b2_seed13_resume.log 2>&1
