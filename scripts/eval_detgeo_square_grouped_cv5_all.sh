#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
GPU="${GPU:-0}"
EVAL_LOG_DIR="results/detgeo_square_grouped_cv5_eval"
mkdir -p "$EVAL_LOG_DIR"

preflight_all_checkpoints() {
  local short fold checkpoint
  for short in drone svi; do
    for fold in 1 2 3 4 5; do
      checkpoint="saved_models/detgeo_square_groupcv5_${short}_f${fold}_seed13_model_best.pth.tar"
      if [[ ! -f "$checkpoint" ]]; then echo "Official-test preflight failed: missing $checkpoint" >&2; exit 1; fi
    done
  done
  echo 'All ten DetGeo Square best checkpoints exist; official evaluation is permitted.'
}

evaluate_dataset() {
  local dataset="$1" short="$2"
  local fold name split_dir checkpoint
  for fold in 1 2 3 4 5; do
    name="detgeo_square_groupcv5_${short}_f${fold}_seed13"
    split_dir="data/${dataset}/cv5_grouped"
    checkpoint="saved_models/${name}_model_best.pth.tar"
    PYTHONPATH=. "$PYTHON_BIN" train.py --gpu "$GPU" --num_workers 24 --batch_size 8 --emb_size 512 --img_size 1024 \
      --data_root data --data_name "$dataset" --train_pth "${split_dir}/fold${fold}_train.pth" \
      --val_pth "${split_dir}/fold${fold}_val.pth" --seed 13 --standard_rng --beta 1.0 \
      --pretrain "$checkpoint" --val --savename "${name}_cvval" 2>&1 | tee "${EVAL_LOG_DIR}/${name}_cvval.log"
    PYTHONPATH=. "$PYTHON_BIN" train.py --gpu "$GPU" --num_workers 24 --batch_size 8 --emb_size 512 --img_size 1024 \
      --data_root data --data_name "$dataset" --train_pth "${split_dir}/fold${fold}_train.pth" \
      --val_pth "${split_dir}/fold${fold}_val.pth" --test_pth "data/${dataset}/${dataset}_test.pth" \
      --seed 13 --standard_rng --beta 1.0 --pretrain "$checkpoint" --test --savename "${name}_cvtest" \
      2>&1 | tee "${EVAL_LOG_DIR}/${name}_cvtest.log"
  done
}

preflight_all_checkpoints
evaluate_dataset CVOGL_DroneAerial drone
evaluate_dataset CVOGL_SVI svi
PYTHONPATH=. "$PYTHON_BIN" tools/summarize_detgeo_square_grouped_cv5.py --log_dir "$EVAL_LOG_DIR" --output_dir results
if [[ -f results/grouped_cv5_summary.json ]]; then
  PYTHONPATH=. "$PYTHON_BIN" tools/compare_fullmodel_detgeo_grouped_cv5.py
else
  echo 'Full Model grouped_cv5_summary.json not found; skip paired comparison.'
fi
