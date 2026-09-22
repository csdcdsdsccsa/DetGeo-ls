#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
GPU="${GPU:-0}"
SPLIT_DIR="data/CVOGL_DroneAerial/cv5_grouped"
test -f "${SPLIT_DIR}/manifest.json"
for FOLD in 1 2 3 4 5; do
  NAME="detgeo_square_groupcv5_drone_f${FOLD}_seed13"
  TRAIN_PTH="${SPLIT_DIR}/fold${FOLD}_train.pth"
  VAL_PTH="${SPLIT_DIR}/fold${FOLD}_val.pth"
  test -f "$TRAIN_PTH"; test -f "$VAL_PTH"
  if [[ -e "saved_models/${NAME}_model_best.pth.tar" ]]; then echo "Refusing to overwrite existing checkpoint: ${NAME}" >&2; exit 1; fi
  PYTHONPATH=. "$PYTHON_BIN" train.py --gpu "$GPU" --num_workers 24 --max_epoch 25 --lr 1e-4 \
    --batch_size 8 --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial \
    --train_pth "$TRAIN_PTH" --val_pth "$VAL_PTH" --savename "$NAME" --seed 13 --standard_rng \
    --beta 1.0 --print_freq 50
  test -f "saved_models/${NAME}_model_best.pth.tar"
  rm -f "saved_models/${NAME}_checkpoint.pth.tar"
done
