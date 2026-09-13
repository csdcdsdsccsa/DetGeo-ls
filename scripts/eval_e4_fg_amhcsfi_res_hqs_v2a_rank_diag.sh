#!/usr/bin/env bash
# Validation-only HQS-v2a rank separability and threshold-sweep diagnostic.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/detgeo/bin/python}"
VARIANT="h2_ind_fg_amhcsfi_res"
NAME="trogeo_ms_e4_${VARIANT}_hqs_v2a_seed2024"
CKPT="saved_models/${NAME}_model_best.pth.tar"

[[ -f "$CKPT" ]] || { echo "Missing HQS-v2a best checkpoint: $CKPT" >&2; exit 1; }
mkdir -p results

PYTHONPATH=. "$PYTHON_BIN" train.py \
  --gpu 0 --num_workers 24 --batch_size 7 --emb_size 768 --img_size 1024 \
  --data_root data --data_name CVOGL_DroneAerial --trogeo_backbone swin_t \
  --trogeo_ms_det_variant "$VARIANT" --standard_rng --seed 2024 --beta 1.0 --print_freq 50 \
  --coarse_loss_weight 0.2 --coarse_sigma 1.5 --fine_sigma 3.0 \
  --hqs_v2a --hqs_rank_epsilon 0.03 --hqs_rank_threshold 0.0 \
  --hqs_v2a_rank_diag --hqs_v2a_rank_diag_prefix results/hqs_v2a_rank_diag \
  --pretrain "$CKPT" --val --savename hqs_v2a_rank_diag_val
