#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
PYTHONPATH=. "$PYTHON_BIN" train.py --gpu 0 --num_workers 16 --batch_size 8 --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --savename standard_c_hisym_pae_inline_seed13_test --seed 13 --beta 1.0 --hisym_pae_inline_init --standard_rng --pretrain saved_models/standard_c_hisym_pae_inline_seed13_model_best.pth.tar --test --print_freq 50 > logs/standard_c_hisym_pae_inline_seed13_test.log 2>&1
