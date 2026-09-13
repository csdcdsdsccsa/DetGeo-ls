#!/usr/bin/env bash
# Validation-only Oracle head-selection diagnostic for the completed FG-Res run.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
VARIANT="h2_ind_fg_amhcsfi_res"
NAME="trogeo_ms_e4_${VARIANT}_standardrng_swin_t_drone_seed2024_rerun1"
CKPT="saved_models/${NAME}_model_best.pth.tar"

mkdir -p results
if [[ ! -f "$CKPT" ]]; then
  echo "Missing validation-best checkpoint: $CKPT" >&2
  exit 1
fi

PYTHONPATH=. "$PYTHON_BIN" tools/test_hqs_oracle_diag.py
PYTHONPATH=. "$PYTHON_BIN" train.py \
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 \
  --data_root data --data_name CVOGL_DroneAerial \
  --trogeo_backbone swin_t --trogeo_ms_det_variant "$VARIANT" \
  --standard_rng --seed 2024 --beta 1.0 --print_freq 50 \
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0 \
  --pretrain "$CKPT" --val --hqs_oracle_diag \
  --hqs_oracle_csv "results/hqs_oracle_fg_amhcsfi_res_val.csv" \
  --savename "fg_amhcsfi_res_hqs_oracle_val"
