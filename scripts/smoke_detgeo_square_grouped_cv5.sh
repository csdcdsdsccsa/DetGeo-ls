#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
GPU="${GPU:-0}"
PYTHONPATH=. "$PYTHON_BIN" tools/check_cvogl_grouped_cv5.py --data_root data
PYTHONPATH=. "$PYTHON_BIN" tools/check_detgeo_square_cv5_loader.py

run_one_epoch() {
  local dataset="$1" name="$2"
  local split_dir="data/${dataset}/cv5_grouped"
  PYTHONPATH=. "$PYTHON_BIN" train.py --gpu "$GPU" --num_workers 24 --max_epoch 1 --lr 1e-4 \
    --batch_size 8 --emb_size 512 --img_size 1024 --data_root data --data_name "$dataset" \
    --train_pth "${split_dir}/fold1_train.pth" --val_pth "${split_dir}/fold1_val.pth" \
    --savename "$name" --seed 13 --standard_rng --beta 1.0 --print_freq 50
  rm -f "saved_models/${name}_checkpoint.pth.tar" "saved_models/${name}_model_best.pth.tar"
}

run_one_epoch CVOGL_DroneAerial detgeo_square_groupcv5_smoke_drone_seed13
run_one_epoch CVOGL_SVI detgeo_square_groupcv5_smoke_svi_seed13
echo 'DetGeo Square Grouped CV5 smoke COMPLETE'
