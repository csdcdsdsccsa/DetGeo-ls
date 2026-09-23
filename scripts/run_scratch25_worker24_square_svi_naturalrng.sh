#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
GPU="${GPU:-0}"
NAME="scratch25_worker24_square_svi_naturalrng_seed13"
BEST="saved_models/${NAME}_model_best.pth.tar"

if [[ -e "$BEST" ]]; then
  echo "Refusing to overwrite existing best checkpoint: $BEST" >&2
  exit 1
fi

mkdir -p logs

# Original DetGeo and official CVOGL_SVI splits.  --standard_rng deliberately
# selects the ordinary DataLoader trajectory without RNG alignment helpers.
PYTHONPATH=. "$PYTHON_BIN" train.py \
  --gpu "$GPU" \
  --num_workers 24 \
  --max_epoch 25 \
  --lr 1e-4 \
  --batch_size 8 \
  --emb_size 512 \
  --img_size 1024 \
  --data_root data \
  --data_name CVOGL_SVI \
  --savename "$NAME" \
  --seed 13 \
  --beta 1.0 \
  --standard_rng \
  --print_freq 50 \
  > "logs/${NAME}.log" 2>&1

test -f "$BEST"

# Evaluate the validation-selected checkpoint once on the untouched official test split.
PYTHONPATH=. "$PYTHON_BIN" train.py \
  --gpu "$GPU" \
  --num_workers 24 \
  --batch_size 8 \
  --emb_size 512 \
  --img_size 1024 \
  --data_root data \
  --data_name CVOGL_SVI \
  --savename "${NAME}_test" \
  --seed 13 \
  --beta 1.0 \
  --standard_rng \
  --pretrain "$BEST" \
  --test \
  --print_freq 50 \
  > "logs/${NAME}_test.log" 2>&1
